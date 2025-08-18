import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
import json
import gc
from typing import List, Dict, Tuple, Optional, Any, Set
import os
import copy

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Utilities and Aggregation
# ==============================================================================

class LayerAggregator:
    """Accumulates probability mass per expert."""
    def __init__(self, moe_layers: List[int], num_experts: int):
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        # Track probability mass assigned to each expert per layer (on CPU to save VRAM)
        self.mass = {L: torch.zeros(num_experts, dtype=torch.float64) for L in moe_layers}
        self.tokens_seen = 0

    # Requirement 1: Logging function
    def update_from_router_logits(
        self,
        router_logits: List[Optional[torch.Tensor]],
        last_pos: int
    ):
        """
        Update stats based on router logits at the last position.
        """
        self.tokens_seen += 1
        for L, layer_logits in enumerate(router_logits):
            if layer_logits is None or L not in self.moe_layers:
                continue
            
            # Handle different shapes ([seq_len, num_experts] or [batch, seq_len, num_experts])
            try:
                if layer_logits.dim() == 2:
                    seq_len = layer_logits.shape[0]
                    target_pos = min(last_pos, seq_len - 1)
                    v = layer_logits[target_pos]
                elif layer_logits.dim() == 3:
                    # Assuming batch size 1 during profiling
                    seq_len = layer_logits.shape[1]
                    target_pos = min(last_pos, seq_len - 1)
                    v = layer_logits[0, target_pos]
                else:
                    continue
            except IndexError:
                continue
            
            # Calculate probabilities (softmax). Use float32 for stability.
            probs = torch.softmax(v.to(torch.float32), dim=-1)
            
            # Update mass
            self.mass[L] += probs.to(torch.float64).cpu()

# ==============================================================================
# 2. MoE Profiler (Handles Requirements 1-4)
# ==============================================================================

class MoEProfiler:
    """
    Profiles MoE expert usage for specific tasks and generates pruning plans.
    """
    def __init__(self, model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        print(f"Loading model: {model_name}...")
        # trust_remote_code might be necessary for some architectures
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval()
        print("Model loaded.")
        
        self._detect_config()
        self.aggregator = LayerAggregator(self.moe_layers, self.num_experts)

    def _detect_config(self):
        """Detects MoE configuration by inspecting the model config and structure."""
        cfg = self.model.config
        self.num_experts = getattr(cfg, 'num_routed_experts', getattr(cfg, 'num_local_experts', None))
        self.top_k = getattr(cfg, 'num_experts_per_tok', 8)
        
        if self.num_experts is None:
            raise ValueError("Could not detect the number of experts in the model configuration.")

        # Probe to find which layers are MoE
        print("Probing model structure to identify MoE layers...")
        try:
            probe_inputs = self.tokenizer("probe", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                probe_out = self.model(
                    **probe_inputs, output_router_logits=True, return_dict=True
                )
            self.moe_layers = [L for L, x in enumerate(probe_out.router_logits) if x is not None]
        except Exception as e:
            raise RuntimeError(f"Failed to probe model for MoE layers: {e}")
            
        self.total_experts = len(self.moe_layers) * self.num_experts

        print(f"\nModel Configuration Detected:")
        print(f"  • MoE Layers Detected: {len(self.moe_layers)}")
        print(f"  • Experts per Layer: {self.num_experts}")
        print(f"  • Experts per Token (Top-K): {self.top_k}")

    # Requirement 2: Prompt processing function
    def profile_task(self, prompts: List[str], task_name: str = "Task", max_new_tokens: int = 64):
        """
        Profile expert usage for a given task by running generation step-by-step.
        """
        print(f"\n{'='*70}\nPROFILING: {task_name} ({len(prompts)} prompts)\n{'='*70}")
        
        for i, prompt in enumerate(prompts):
            if (i + 1) % max(1, len(prompts) // 5) == 0 or i == 0:
                print(f"  Progress: {i + 1}/{len(prompts)}")
            
            self._generate_and_log(prompt, max_new_tokens)
            
            # Memory management during long profiling runs
            if (i + 1) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"Profiling complete. Total tokens analyzed: {self.aggregator.tokens_seen}")

    def _generate_and_log(self, prompt: str, max_new_tokens: int):
        """Step-by-step generation for a single prompt, logging router probs."""
        # Apply chat template if available
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_text = prompt
        
        inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=True).to(self.model.device)
        seq = inputs.input_ids
        attn = inputs.attention_mask

        # Determine EOS tokens
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        # Handle specific tokens (e.g., Qwen <|im_end|>)
        try:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id >= 0:
                eos_ids.add(im_end_id)
        except Exception:
            pass

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(
                    input_ids=seq,
                    attention_mask=attn,
                    output_router_logits=True,
                    return_dict=True,
                    use_cache=False # Disable cache for easier position tracking
                )
                
                # Router logits at the last input position govern the NEXT token
                last_pos = seq.shape[1] - 1
                
                # Log the routing decisions for this step
                self.aggregator.update_from_router_logits(
                    out.router_logits, last_pos
                )

                # Get next token (greedy decoding for deterministic profiling)
                logits = out.logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1)
                
                # Ensure shape is [B, 1] for concatenation (assuming B=1)
                if next_id.dim() == 1:
                    next_id = next_id.unsqueeze(0)
                if next_id.dim() == 0:
                    next_id = next_id.view(1, 1)
                
                next_id_item = int(next_id[0, 0].item())

                # Stop on EOS
                if next_id_item in eos_ids:
                    break
                
                # Update sequence for next iteration
                seq = torch.cat([seq, next_id], dim=1)
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=attn.device, dtype=attn.dtype)],
                    dim=1,
                )

    # Requirement 3 & 4: Analysis and identifying unused experts
    def generate_uniform_pruning_plan(self, coverage: float = 0.99):
        """
        Analyzes the profiling data and generates a UNIFORM pruning plan (Union method).
        """
        print(f"\n{'='*70}\nGENERATING UNIFORM PRUNING PLAN (Target Coverage: {coverage*100:.2f}%)\n{'='*70}")

        if self.aggregator.tokens_seen == 0:
            print("No tokens profiled. Cannot generate plan."); return None

        plan = self._build_uniform_plan(self.aggregator, coverage, self.top_k)
        
        self._report_plan(plan)
        return plan

    def _compute_layer_requirements(self, mass_vec: torch.Tensor, coverage: float, top_k: int) -> Set[int]:
        """
        Determine experts required for a single layer to meet target coverage.
        Ensures at least 'top_k' experts are identified.
        """
        total = float(mass_vec.sum().item())
        if total <= 0.0:
            # If a layer was never used (unlikely), keep the first top_k experts for stability.
            return set(range(min(top_k, len(mass_vec))))
        
        # Sort experts by mass (descending)
        order = torch.argsort(mass_vec, descending=True)
        
        # If coverage is 1.0, we look for non-zero mass
        if coverage >= 1.0:
            non_zero_mask = mass_vec[order] > 0
            keep_n = max(top_k, int(non_zero_mask.sum().item()))
        else:
            # Calculate cumulative sum and find the target index
            cumsum = torch.cumsum(mass_vec[order], dim=0)
            target = coverage * total
            idx = int(torch.searchsorted(cumsum, torch.tensor(target)).item())
            # Ensure we keep at least top_k experts
            keep_n = max(top_k, idx + 1)
        
        keep_indices = order[:keep_n].cpu().tolist()
        return set(keep_indices)

    def _build_uniform_plan(self, agg: LayerAggregator, coverage: float, top_k: int) -> Dict[str, Any]:
        """
        Build a uniform pruning plan: identify the UNION of experts required
        across all layers.
        """
        required_experts_union = set()
        per_layer_requirements_count = {}

        # 1. Identify required experts per layer
        for L in agg.moe_layers:
            required_indices = self._compute_layer_requirements(agg.mass[L], coverage, top_k)
            required_experts_union.update(required_indices)
            per_layer_requirements_count[L] = len(required_indices)

        # 2. Create the uniform plan
        uniform_keep_indices = sorted(list(required_experts_union))
        uniform_keep_count = len(uniform_keep_indices)

        # 3. (Requirement 4) Identify globally unused experts (zero mass in ALL layers)
        globally_unused = set(range(agg.num_experts))
        for L in agg.moe_layers:
             # Experts with non-zero mass
             used_in_layer = (agg.mass[L] > 0).nonzero(as_tuple=True)[0].tolist()
             # An expert is globally unused only if it's unused in every layer
             # Here we remove the used experts from the set of all experts
             # Note: The logic in the original attempt was slightly flawed for "globally unused".
             # We redefine globally unused as those NOT in the union when coverage=1.0.
             
        # Recalculate union for 100% coverage to find truly unused experts
        if coverage < 1.0:
            unused_100_coverage = set(range(agg.num_experts))
            for L in agg.moe_layers:
                required_100 = self._compute_layer_requirements(agg.mass[L], 1.0, top_k)
                unused_100_coverage.intersection_update(set(range(agg.num_experts)) - required_100)
        else:
            unused_100_coverage = set(range(agg.num_experts)) - required_experts_union


        plan = {
            "strategy": "uniform_union",
            "coverage_target": coverage,
            "uniform_keep_indices": uniform_keep_indices,
            "uniform_keep_count": uniform_keep_count,
            "original_num_experts": agg.num_experts,
            "per_layer_requirements_count": per_layer_requirements_count,
            "globally_unused_experts_count_100_coverage": len(unused_100_coverage)
        }
        return plan

    def _report_plan(self, plan: Dict[str, Any]):
        """Generate a human-readable report of the pruning plan. (Requirement 3)"""
        keep_count = plan["uniform_keep_count"]
        original_count = plan["original_num_experts"]
        savings_ratio = 1.0 - (keep_count / original_count)
        
        print("\n--- Pruning Plan Summary ---")
        print(f"Strategy: Uniform (Union Method) - Maximizes compatibility")
        print(f"Original Experts per Layer: {original_count}")
        print(f"Experts Kept per Layer:     {keep_count}")
        print(f"💡 Pruning Rate (VRAM Savings Estimate for MoE layers): {savings_ratio*100:.2f}%")
        print(f"Experts with Zero Activations (at 100% coverage): {plan['globally_unused_experts_count_100_coverage']}")

        print("\n--- Per-Layer Requirements Analysis ---")
        print(f"This shows how many experts each layer needed *individually* for {plan['coverage_target']*100:.2f}% coverage.")
        print("The uniform plan keeps the UNION of these requirements.")
        
        requirements = list(plan['per_layer_requirements_count'].values())
        if not requirements: return

        avg_req = sum(requirements) / len(requirements)
        
        print(f"Min required: {min(requirements)}")
        print(f"Max required: {max(requirements)}")
        print(f"Avg required: {avg_req:.1f}")

# ==============================================================================
# 3. MoE Pruner (Requirement 5: Architectural Modification)
# ==============================================================================

class MoEPruner:
    """
    Applies a uniform pruning plan to a Hugging Face MoE model.
    Physically removes unused experts and updates the model configuration.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, pruning_plan: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.plan = pruning_plan
        self.config = model.config

    def prune(self):
        """
        Executes the pruning process in-place based on the uniform plan.
        """
        if self.plan is None:
            print("Pruning plan is missing. Aborting."); return self.model

        experts_to_keep_indices = self.plan["uniform_keep_indices"]
        new_num_experts = self.plan["uniform_keep_count"]
        original_num_experts = self.plan["original_num_experts"]

        if new_num_experts >= original_num_experts:
            print("No significant pruning possible."); return self.model

        print(f"\n{'='*70}\nSTARTING UNIFORM HARD PRUNING\n{'='*70}")
        print(f"Keeping {new_num_experts}/{original_num_experts} experts across all MoE layers.")

        # We need to access the layers. Assuming standard transformer structure (Llama/Qwen/Mixtral)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
             layers = self.model.model.layers
        else:
            raise RuntimeError("Could not identify the decoder layers (expected model.model.layers).")

        # Identify MoE layer indices based on the profiling results
        moe_layer_indices = sorted([int(k) for k in self.plan["per_layer_requirements_count"].keys()])
        
        # Convert keep indices to tensor for slicing
        keep_indices_tensor = torch.tensor(experts_to_keep_indices, dtype=torch.long).to(self.model.device)

        for layer_idx in moe_layer_indices:
            print(f"Processing Layer {layer_idx}...")
            layer = layers[layer_idx]

            # Identify the MoE block within the layer
            moe_block = self._find_moe_block(layer)
            if moe_block is None:
                print(f"Warning: Could not find MoE block in layer {layer_idx}. Skipping.")
                continue
                
            # 1. Prune the experts (MLPs) - Handles both Fused and ModuleList
            self._prune_experts(moe_block, experts_to_keep_indices, keep_indices_tensor, original_num_experts)

            # 2. Prune the router (gate)
            gate, gate_name = self._find_router(moe_block, original_num_experts)
            if gate is None:
                 raise RuntimeError(f"Could not find router in layer {layer_idx}. Pruning aborted.")

            self._prune_router(moe_block, gate, gate_name, keep_indices_tensor, new_num_experts)
            
            # 3. Update internal attributes if present (important for some architectures)
            if hasattr(moe_block, 'num_experts'):
                moe_block.num_experts = new_num_experts
            if hasattr(moe_block, 'num_routed_experts'):
                    moe_block.num_routed_experts = new_num_experts


        # 4. Update the global configuration
        self._update_config(new_num_experts)

        print(f"\n{'='*70}\nPRUNING COMPLETE.\n{'='*70}")
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.model

    def _find_moe_block(self, layer):
        """Heuristically finds the MoE block within a decoder layer."""
        # Common names (Qwen uses 'mlp', Mixtral uses 'block_sparse_moe')
        potential_names = ['mlp', 'block_sparse_moe', 'feed_forward']
        for name in potential_names:
            if hasattr(layer, name):
                module = getattr(layer, name)
                # Check if it looks like an MoE block (e.g., has a gate or experts)
                if hasattr(module, 'gate') or hasattr(module, 'experts') or hasattr(module, 'w1'):
                     return module
        return None

    def _find_router(self, moe_block, num_experts):
        """Heuristically finds the router (gate) within the MoE block."""
        # Common names
        potential_names = ['gate', 'router']
        for name in potential_names:
            if hasattr(moe_block, name):
                module = getattr(moe_block, name)
                # Check if it's a Linear layer mapping to the number of experts
                if isinstance(module, nn.Linear) and module.out_features == num_experts:
                    return module, name
        return None, None

    def _prune_experts(self, moe_block, keep_indices_list, keep_indices_tensor, original_num_experts):
        """Prunes the experts, handling both ModuleList and Fused Tensor implementations."""

        # Case 1: Experts stored as a ModuleList (e.g., Mixtral)
        if hasattr(moe_block, 'experts') and isinstance(moe_block.experts, nn.ModuleList):
            print("  Detected ModuleList implementation (e.g., Mixtral). Pruning...")
            new_experts = nn.ModuleList()
            for idx in keep_indices_list:
                # Use deepcopy to ensure clean transfer
                new_experts.append(copy.deepcopy(moe_block.experts[idx]))
            moe_block.experts = new_experts
            return

        # Case 2: Fused/Concatenated Weights (e.g., Qwen MoE)
        # Look for typical weight names (w1, w2, w3 or gate_proj, up_proj, down_proj)
        pruned_weights = False
        weight_names = ['w1', 'w2', 'w3', 'gate_proj', 'up_proj', 'down_proj']
        
        # Determine the container (might be moe_block or moe_block.experts)
        container = moe_block
        if hasattr(moe_block, 'experts') and not isinstance(moe_block.experts, nn.ModuleList):
             container = moe_block.experts

        for name in weight_names:
            if hasattr(container, name):
                param = getattr(container, name)
                if isinstance(param, nn.Parameter):
                    weight = param.data
                    # Assume the first dimension is the expert dimension
                    if weight.shape[0] == original_num_experts:
                        print(f"  Detected Fused Tensor implementation (e.g., Qwen). Pruning {name}...")
                        new_weight = weight.index_select(0, keep_indices_tensor)
                        # Update the parameter in place
                        param.data = new_weight
                        pruned_weights = True
        
        if not pruned_weights:
            raise RuntimeError(f"Could not identify expert weights structure (ModuleList or Fused Tensors) in {type(container).__name__}.")

    def _prune_router(self, moe_block, gate, gate_name, keep_indices_tensor, new_num_experts):
        """Creates a new Linear layer for the router with reduced dimensions."""

        # Slice the weights and bias along the output dimension (dim 0)
        new_gate_weight = gate.weight.data.index_select(0, keep_indices_tensor)
        new_gate_bias = gate.bias.data.index_select(0, keep_indices_tensor) if gate.bias is not None else None

        # Create the new gate module
        new_gate = nn.Linear(gate.in_features, new_num_experts, bias=new_gate_bias is not None)
        new_gate.weight.data = new_gate_weight
        if new_gate_bias is not None:
            new_gate.bias.data = new_gate_bias

        # Replace the old gate in the MoE block, ensuring device/dtype consistency
        setattr(moe_block, gate_name, new_gate.to(gate.weight.device, dtype=gate.weight.dtype))

    def _update_config(self, new_num_experts):
        """Updates the model configuration to reflect the new number of experts."""
        print("Updating global model configuration...")
        # Update common configuration keys
        if hasattr(self.config, "num_routed_experts"):
            self.config.num_routed_experts = new_num_experts
        if hasattr(self.config, "num_local_experts"):
            self.config.num_local_experts = new_num_experts
            
        # Adjust Top-K if it exceeds the new number of experts
        if self.config.num_experts_per_tok > new_num_experts:
            print(f"Adjusting Top-K (num_experts_per_tok) from {self.config.num_experts_per_tok} to {new_num_experts}.")
            self.config.num_experts_per_tok = new_num_experts


    def save_pruned_model(self, output_dir: str):
        """Saves the pruned model and tokenizer to the specified directory."""
        print(f"Saving pruned model and tokenizer to {output_dir}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            # Ensure the config reflects the changes before saving
            self.config.save_pretrained(output_dir)
            print("✅ Model saved successfully.")
            print(f"You can now load this model from '{output_dir}' using standard HF tools.")
        except Exception as e:
            print(f"Error saving model: {e}")

# ==============================================================================
# 4. Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    # --- IMPORTANT ---
    # To run Qwen3-30B, you need substantial VRAM (e.g., A100 80GB).
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
    
    # Example using a smaller MoE model for testing the mechanism if resources are limited.
    # Qwen1.5-MoE-A2.7B uses Fused Tensors, similar to Qwen3-30B.
    #MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat" 
    
    TASK_NAME = "Python Coding Generation"
    MAX_NEW_TOKENS = 128  # Analyze generation length
    PRUNING_COVERAGE = 1.0 # Keep experts covering 100% of the probability mass (all used experts)
    OUTPUT_DIR = f"./{MODEL_NAME.replace('/', '_')}-Pruned-Python"

    # Define prompts representative of the target task
    prompts = [
        "Write a Python function to calculate the Fibonacci sequence recursively.",
        "How do I use list comprehensions in Python? Provide an example.",
        "Explain the difference between '==' and 'is' in Python with code snippets.",
        "Write a Python script to read a CSV file using pandas and calculate the average.",
        "What are decorators in Python and how can I implement a simple timing decorator?",
        "Generate a class definition for a Binary Search Tree in Python.",
        "How to handle exceptions in Python using try, except, else, and finally?",
        "Write a function that uses `argparse` in a Python script.",
        "Explain Python's Global Interpreter Lock (GIL).",
        "Provide an example of using `asyncio` for asynchronous programming in Python."
    ]

    # Use torch.bfloat16 if supported (Ampere+ GPUs) for better performance and stability
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16

    # --- Pipeline Execution ---
    try:
        # --- Step 1: Initialize Profiler (Loads the model) ---
        profiler = MoEProfiler(model_name=MODEL_NAME, device_map="auto", dtype=DTYPE)

        # --- Step 2: Profile the Task ---
        profiler.profile_task(prompts=prompts, task_name=TASK_NAME, max_new_tokens=MAX_NEW_TOKENS)

        # --- Step 3: Analyze and Generate Pruning Plan ---
        plan = profiler.generate_uniform_pruning_plan(coverage=PRUNING_COVERAGE)

        if plan:
            # --- Step 4: Apply Pruning ---
            # The profiler holds the loaded model, so we pass it to the pruner.
            pruner = MoEPruner(model=profiler.model, tokenizer=profiler.tokenizer, pruning_plan=plan)
            
            if plan["uniform_keep_count"] < plan["original_num_experts"]:
                pruned_model = pruner.prune()

                # --- Step 5: Save the Pruned Model ---
                pruner.save_pruned_model(OUTPUT_DIR)
                
                # --- Step 6: Verification (Optional) ---
                print(f"\nVerifying pruned model loading from {OUTPUT_DIR}...")
                try:
                    # Load the pruned model to ensure structural integrity
                    model_v = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto", torch_dtype=DTYPE)
                    tokenizer_v = AutoTokenizer.from_pretrained(OUTPUT_DIR)
                    print("Verification successful: Model loaded correctly.")
                    
                    # Test inference
                    test_prompt = "Write a python function to sort a list."
                    test_input = tokenizer_v(test_prompt, return_tensors="pt").to(model_v.device)
                    output = model_v.generate(**test_input, max_new_tokens=50)
                    print("\nTest Output:\n", tokenizer_v.decode(output[0], skip_special_tokens=True))
                    
                except Exception as e:
                    print(f"Verification FAILED: {e}")

            else:
                print("\nSkipping pruning as the task utilizes nearly all experts at the target coverage.")

    except Exception as e:
        print(f"\nAn error occurred during the pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        if "out of memory" in str(e).lower():
            print("\nHint: The model might be too large for your available VRAM/RAM.")

    print("\n✅ Script finished!")
    gc.collect()
    torch.cuda.empty_cache()