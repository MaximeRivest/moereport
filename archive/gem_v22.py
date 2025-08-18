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
            if v.numel() > 0:
                # Check for numerical stability
                if not torch.isfinite(v).all():
                    print(f"Warning: Non-finite values in router logits Layer {L}. Skipping token.")
                    continue
                
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
        self.model_name = model_name
        print(f"Loading model: {model_name}...")
        # trust_remote_code is essential for Qwen3 MoE architectures
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
        """Detects MoE configuration robustly across various architectures."""
        cfg = self.model.config
        self.num_experts = None
        self.top_k = None

        # 1. Check nested configurations (Standardized approach in newer models)
        potential_moe_keys = ['moe_config', 'ffn_config', 'router_config']
        for key in potential_moe_keys:
            if hasattr(cfg, key) and getattr(cfg, key) is not None:
                moe_cfg = getattr(cfg, key)
                # Handle dictionary or object-like access
                if isinstance(moe_cfg, dict):
                    if self.num_experts is None:
                        self.num_experts = moe_cfg.get('num_experts', moe_cfg.get('num_routed_experts'))
                    if self.top_k is None:
                        self.top_k = moe_cfg.get('num_experts_per_tok', moe_cfg.get('moe_top_k'))
                else:
                    if self.num_experts is None:
                        self.num_experts = getattr(moe_cfg, 'num_experts', getattr(moe_cfg, 'num_routed_experts', None))
                    if self.top_k is None:
                        self.top_k = getattr(moe_cfg, 'num_experts_per_tok', getattr(moe_cfg, 'moe_top_k', None))

        # 2. Check top-level attributes (Common in many existing models, including Qwen3-30B-A3B)
        if self.num_experts is None:
            self.num_experts = getattr(cfg, 'num_experts', None)
        if self.num_experts is None:
            self.num_experts = getattr(cfg, 'num_routed_experts', None)
        if self.num_experts is None:
            self.num_experts = getattr(cfg, 'num_local_experts', None)
            
        if self.top_k is None:
             self.top_k = getattr(cfg, 'num_experts_per_tok', None)

        # 3. Final check and fallback
        if self.num_experts is None:
            print("Loaded Model Config Keys:")
            try: print(list(cfg.to_dict().keys()))
            except: pass
            raise ValueError(f"Could not detect the number of experts for {self.model_name}. Checked top-level and nested configs.")

        if self.top_k is None:
            self.top_k = 8 # Default fallback
            print(f"Warning: Could not detect Top-K (num_experts_per_tok). Defaulting to {self.top_k}.")

        # Probe to find which layers are MoE
        print("Probing model structure to identify MoE layers...")
        try:
             # Ensure the model configuration allows outputting router logits
            if hasattr(self.model.config, 'output_router_logits'):
                self.model.config.output_router_logits = True

            probe_inputs = self.tokenizer("probe test", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                probe_out = self.model(
                    **probe_inputs, output_router_logits=True, return_dict=True, use_cache=False
                )
            
            if not hasattr(probe_out, 'router_logits') or probe_out.router_logits is None:
                 raise ValueError("Model did not return 'router_logits' during probe.")

            router_logits_output = probe_out.router_logits

            # Handle potential variations in output structure (List vs Tuple of Tuples)
            if isinstance(router_logits_output, (list, tuple)):
                 # Standard structure: list where index corresponds to layer index, None if not MoE
                self.moe_layers = [L for L, x in enumerate(router_logits_output) if x is not None]
            else:
                raise TypeError(f"Unexpected type for router_logits: {type(router_logits_output)}")

        except Exception as e:
            raise RuntimeError(f"Failed to probe model for MoE layers: {e}")
            
        if not self.moe_layers:
            raise ValueError("No MoE layers were detected during probing.")

        self.total_experts = len(self.moe_layers) * self.num_experts

        print(f"\nModel Configuration Detected:")
        print(f"  • Total Layers: {getattr(cfg, 'num_hidden_layers', 'N/A')}")
        print(f"  • MoE Layers Detected: {len(self.moe_layers)}")
        print(f"  • Experts per Layer: {self.num_experts}")
        print(f"  • Experts per Token (Top-K): {self.top_k}")

    # Requirement 2: Prompt processing function
    def profile_task(self, prompts: List[str], task_name: str = "Task", max_new_tokens: int = 64):
        """
        Profile expert usage for a given task by running generation step-by-step.
        """
        print(f"\n{'='*70}\nPROFILING: {task_name} ({len(prompts)} prompts)\n{'='*70}")
        
        # Ensure the model is configured to output logits during the run
        if hasattr(self.model.config, 'output_router_logits'):
            self.model.config.output_router_logits = True

        for i, prompt in enumerate(prompts):
            if (i + 1) % max(1, len(prompts) // 5) == 0 or i == 0:
                print(f"  Progress: {i + 1}/{len(prompts)}")
            
            self._generate_and_log(prompt, max_new_tokens)
            
            # Memory management during long profiling runs
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"Profiling complete. Total tokens analyzed: {self.aggregator.tokens_seen}")

    def _get_eos_ids(self):
        """Helper to identify EOS token IDs robustly."""
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_ids.update(self.tokenizer.eos_token_id)
            else:
                eos_ids.add(self.tokenizer.eos_token_id)

        # Handle specific tokens (e.g., Qwen <|im_end|>)
        try:
            # Qwen3 often uses 151645 (<|im_end|>) as EOS for chat
            if 151645 in self.tokenizer.get_vocab():
                eos_ids.add(151645)
        except Exception:
            pass
        return eos_ids

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
        
        # Set a reasonable max length for inputs
        inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        seq = inputs.input_ids
        attn = inputs.attention_mask

        eos_ids = self._get_eos_ids()

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(
                    input_ids=seq,
                    attention_mask=attn,
                    output_router_logits=True,
                    return_dict=True,
                    use_cache=False # Disable cache for easier position tracking and memory management
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
                
                # Ensure shape is [B, 1] for concatenation
                if next_id.dim() == 1:
                    next_id = next_id.unsqueeze(1)
                elif next_id.dim() == 0:
                    next_id = next_id.view(1, 1)
                
                # Check batch size consistency (assuming batch size 1)
                if next_id.shape[0] != seq.shape[0]:
                     next_id = next_id[0].view(1, 1)

                next_id_item = int(next_id[0, 0].item())

                # Stop on EOS
                if next_id_item in eos_ids:
                    break
                
                # Update sequence for next iteration
                seq = torch.cat([seq, next_id], dim=1)
                attn = torch.cat(
                    [attn, torch.ones((seq.shape[0], 1), device=attn.device, dtype=attn.dtype)],
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
        if mass_vec.numel() == 0:
            return set()
            
        total = float(mass_vec.sum().item())
        epsilon = 1e-9 # Threshold for numerical stability
        
        if total <= epsilon:
            # If a layer was never used, keep the first top_k experts for stability.
            return set(range(min(top_k, len(mass_vec))))
        
        # Sort experts by mass (descending)
        order = torch.argsort(mass_vec, descending=True)
        
        # If coverage is 1.0, we look for non-zero mass
        if coverage >= 1.0:
            non_zero_mask = mass_vec[order] > epsilon
            keep_n = max(top_k, int(non_zero_mask.sum().item()))
        else:
            # Calculate cumulative sum and find the target index
            cumsum = torch.cumsum(mass_vec[order], dim=0)
            target = coverage * total
            # Handle potential floating point inaccuracies
            target_tensor = torch.tensor(target, dtype=cumsum.dtype)
            idx = int(torch.searchsorted(cumsum, target_tensor).item())
            # Ensure we keep at least top_k experts
            keep_n = max(top_k, idx + 1)
        
        # Ensure keep_n does not exceed the number of experts
        keep_n = min(keep_n, len(mass_vec))
        
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

        # 3. (Requirement 4) Identify globally unused experts
             
        # Recalculate union for 100% coverage to find truly unused experts
        if coverage < 1.0:
            # We need the union of experts required if coverage was 1.0
            union_100 = set()
            for L in agg.moe_layers:
                required_100 = self._compute_layer_requirements(agg.mass[L], 1.0, top_k)
                union_100.update(required_100)
            unused_100_coverage = set(range(agg.num_experts)) - union_100
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
        
        if original_count == 0:
            savings_ratio = 0.0
        else:
            savings_ratio = 1.0 - (keep_count / original_count)
        
        print("\n--- Pruning Plan Summary ---")
        print(f"Strategy: Uniform (Union Method) - Maximizes compatibility")
        print(f"Original Experts per Layer: {original_count}")
        print(f"Experts Kept per Layer:     {keep_count}")
        print(f"💡 Pruning Rate (VRAM Savings Estimate for MoE layers): {savings_ratio*100:.2f}%")
        print(f"Experts with Zero Activations (at 100% coverage): {plan['globally_unused_experts_count_100_coverage']}")

        print("\n--- Per-Layer Requirements Analysis ---")
        
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

        # Identify decoder layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
             layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
             layers = self.model.layers
        else:
            raise RuntimeError("Could not identify the decoder layers (expected model.layers or model.model.layers).")

        # Identify MoE layer indices
        moe_layer_indices = sorted([int(k) for k in self.plan["per_layer_requirements_count"].keys()])
        
        # Convert keep indices to tensor for slicing
        keep_indices_tensor = torch.tensor(experts_to_keep_indices, dtype=torch.long).to(self.model.device)

        for layer_idx in moe_layer_indices:
            if layer_idx >= len(layers): continue

            print(f"Processing Layer {layer_idx}...")
            layer = layers[layer_idx]

            # Identify the MoE block within the layer
            moe_block = self._find_moe_block(layer)
            if moe_block is None:
                print(f"Warning: Could not find MoE block in layer {layer_idx}. Skipping.")
                continue
                
            # 1. Prune the experts (MLPs)
            self._prune_experts(moe_block, experts_to_keep_indices, keep_indices_tensor, original_num_experts)

            # 2. Prune the router (gate)
            gate, gate_name_path = self._find_router(moe_block, original_num_experts)
            
            if gate is not None:
                self._prune_router(moe_block, gate, gate_name_path, keep_indices_tensor, new_num_experts)
            else:
                # Handle cases where a router might not be present if only shared experts exist
                if self._handle_shared_expert_check(moe_block):
                     print(f"  Router not found, but shared expert detected. Proceeding.")
                else:
                    raise RuntimeError(f"Could not find router/gate in layer {layer_idx}. Pruning aborted.")

            # 3. Update internal attributes
            self._update_moe_block_attributes(moe_block, new_num_experts)


        # 4. Update the global configuration
        self._update_config(new_num_experts)

        print(f"\n{'='*70}\nPRUNING COMPLETE.\n{'='*70}")
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.model

    def _find_moe_block(self, layer):
        """Heuristically finds the MoE block within a decoder layer."""
        potential_names = ['mlp', 'block_sparse_moe', 'feed_forward']
        for name in potential_names:
            if hasattr(layer, name):
                module = getattr(layer, name)
                # Check for MoE indicators (gate, router, experts, fused weights, shared expert)
                if hasattr(module, 'gate') or hasattr(module, 'router') or hasattr(module, 'experts') or hasattr(module, 'w1') or hasattr(module, 'shared_expert'):
                     return module
        return None

    def _find_router(self, moe_block, num_experts, potential_names=['gate', 'router']):
        """Heuristically finds the router (gate) within the MoE block. Handles wrapped routers."""
        for name in potential_names:
            if hasattr(moe_block, name):
                module = getattr(moe_block, name)
                
                # Case 1: Direct nn.Linear
                if isinstance(module, nn.Linear) and module.out_features == num_experts:
                    return module, name
                
                # Case 2: Wrapped Linear (e.g., custom Router class containing the Linear layer)
                for internal_name in ['layer', 'dense', 'linear']:
                    if hasattr(module, internal_name) and isinstance(getattr(module, internal_name), nn.Linear):
                        internal_linear = getattr(module, internal_name)
                        if internal_linear.out_features == num_experts:
                            # Return the actual Linear layer and its full path
                            return internal_linear, f"{name}.{internal_name}"
                            
        return None, None

    def _handle_shared_expert_check(self, moe_block):
        """Checks if the block contains shared expert components."""
        return hasattr(moe_block, 'shared_expert')

    def _prune_experts(self, moe_block, keep_indices_list, keep_indices_tensor, original_num_experts):
        """Prunes the experts, handling both ModuleList and Fused Tensor implementations."""

        # Case 1: Experts stored as a ModuleList (e.g., Mixtral)
        if hasattr(moe_block, 'experts') and isinstance(moe_block.experts, nn.ModuleList):
            print("  Detected ModuleList implementation (e.g., Mixtral). Pruning...")
            new_experts = nn.ModuleList()
            for idx in keep_indices_list:
                new_experts.append(copy.deepcopy(moe_block.experts[idx]))
            moe_block.experts = new_experts
            return

        # Case 2: Fused/Concatenated Weights (e.g., Qwen MoE)
        pruned_weights = False
        weight_names = ['w1', 'w2', 'w3', 'gate_proj', 'up_proj', 'down_proj']
        
        # Determine the container
        container = moe_block
        if hasattr(moe_block, 'experts') and not isinstance(moe_block.experts, nn.ModuleList) and isinstance(moe_block.experts, nn.Module):
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
             if self._handle_shared_expert_check(moe_block):
                 print("  No routed expert weights found (shared expert detected).")
             else:
                print(f"Warning: Could not identify routed expert weights structure in {type(container).__name__}.")

    def _prune_router(self, moe_block, gate, gate_name_path, keep_indices_tensor, new_num_experts):
        """Creates a new Linear layer for the router with reduced dimensions. Handles nested paths."""
        print(f"  Pruning router: {gate_name_path}...")
        
        # Navigate to the parent module if the path is nested (e.g., "router.layer")
        parent_module = moe_block
        path_parts = gate_name_path.split('.')
        gate_name = path_parts[-1]
        
        for part in path_parts[:-1]:
             parent_module = getattr(parent_module, part)

        # Slice the weights and bias along the output dimension (dim 0)
        new_gate_weight = gate.weight.data.index_select(0, keep_indices_tensor)
        new_gate_bias = gate.bias.data.index_select(0, keep_indices_tensor) if gate.bias is not None else None

        # Create the new gate module
        new_gate = nn.Linear(gate.in_features, new_num_experts, bias=new_gate_bias is not None)
        new_gate.weight.data = new_gate_weight
        if new_gate_bias is not None:
            new_gate.bias.data = new_gate_bias

        # Replace the old gate in the parent module
        setattr(parent_module, gate_name, new_gate.to(gate.weight.device, dtype=gate.weight.dtype))

    def _update_moe_block_attributes(self, moe_block, new_num_experts):
        """Updates internal attributes of the MoE block and its router."""
        # Update expert counts
        if hasattr(moe_block, 'num_experts'):
            moe_block.num_experts = new_num_experts
        if hasattr(moe_block, 'num_routed_experts'):
            moe_block.num_routed_experts = new_num_experts
            
        # Update Top-K on the block itself
        if hasattr(moe_block, 'top_k') and moe_block.top_k > new_num_experts:
             moe_block.top_k = new_num_experts

        # Update Top-K on the router module if it exists (e.g. Qwen3MoeRouter)
        for router_name in ['gate', 'router']:
             if hasattr(moe_block, router_name):
                  router_module = getattr(moe_block, router_name)
                  if hasattr(router_module, 'top_k') and router_module.top_k > new_num_experts:
                       router_module.top_k = new_num_experts

    def _update_config(self, new_num_experts):
        """Updates the global model configuration robustly."""
        print("Updating global model configuration...")
        
        cfg = self.config
        
        # 1. Update nested configurations
        potential_moe_keys = ['moe_config', 'ffn_config', 'router_config']
        for key in potential_moe_keys:
            if hasattr(cfg, key) and getattr(cfg, key) is not None:
                moe_cfg = getattr(cfg, key)
                # Update expert counts in nested config
                for expert_key in ['num_experts', 'num_routed_experts']:
                    if isinstance(moe_cfg, dict):
                        if expert_key in moe_cfg: moe_cfg[expert_key] = new_num_experts
                    else:
                        if hasattr(moe_cfg, expert_key): setattr(moe_cfg, expert_key, new_num_experts)
                
                # Adjust Top-K in nested config
                for tk_key in ['num_experts_per_tok', 'moe_top_k']:
                    current_tk = None
                    if isinstance(moe_cfg, dict):
                        current_tk = moe_cfg.get(tk_key)
                    else:
                        current_tk = getattr(moe_cfg, tk_key, None)

                    if current_tk is not None and current_tk > new_num_experts:
                        print(f"Adjusting Top-K ({key}.{tk_key}) to {new_num_experts}.")
                        if isinstance(moe_cfg, dict):
                            moe_cfg[tk_key] = new_num_experts
                        else:
                            setattr(moe_cfg, tk_key, new_num_experts)

        # 2. Update top-level configuration keys
        if hasattr(cfg, "num_experts"):
             cfg.num_experts = new_num_experts
        if hasattr(cfg, "num_routed_experts"):
            cfg.num_routed_experts = new_num_experts
        if hasattr(cfg, "num_local_experts"):
            cfg.num_local_experts = new_num_experts
            
        # Adjust Top-K at top level
        if hasattr(cfg, "num_experts_per_tok") and cfg.num_experts_per_tok > new_num_experts:
            print(f"Adjusting Top-K (num_experts_per_tok) to {new_num_experts}.")
            cfg.num_experts_per_tok = new_num_experts


    def save_pruned_model(self, output_dir: str):
        """Saves the pruned model and tokenizer to the specified directory."""
        print(f"\nSaving pruned model and tokenizer to {output_dir}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            # Ensure the model's internal config attribute is updated before saving
            self.model.config = self.config
            
            # Save model, tokenizer, and config. Use smaller shard size for large models.
            self.model.save_pretrained(output_dir, max_shard_size="4GB")
            self.tokenizer.save_pretrained(output_dir)
            
            print("✅ Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

# ==============================================================================
# 4. Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    # Model: Qwen3-30B-A3B (Requires significant VRAM, e.g., A100 80GB)
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
    
    # If you want to test with a smaller model first:
    # MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat" 
    
    TASK_NAME = "Python Coding Generation"
    MAX_NEW_TOKENS = 64  # Reduced for faster execution on large models
    PRUNING_COVERAGE = 1.0 # Keep experts covering 100% of the probability mass (all used experts)
    # PRUNING_COVERAGE = 0.99 # Keep experts covering 99% of the probability mass

    OUTPUT_DIR = f"./{MODEL_NAME.replace('/', '_')}-Pruned-Python"

    # Define prompts representative of the target task (Reduced count for faster execution)
    prompts = [
        "Write a Python function to calculate the Fibonacci sequence recursively.",
        "How do I use list comprehensions in Python? Provide an example.",
        "Explain the difference between '==' and 'is' in Python with code snippets.",
        "Write a Python script to read a CSV file using pandas and calculate the average.",
        "What are decorators in Python and how can I implement a simple timing decorator?",
    ]

    # Determine dtype
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            DTYPE = torch.bfloat16
            print("Using bfloat16.")
        else:
            DTYPE = torch.float16
            print("Using float16 (bfloat16 not supported).")
    else:
        DTYPE = torch.float32
        print("CUDA not available, using float32 on CPU (Warning: This will be extremely slow).")


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
            pruner = MoEPruner(model=profiler.model, tokenizer=profiler.tokenizer, pruning_plan=plan)
            
            if plan["uniform_keep_count"] < plan["original_num_experts"]:
                pruned_model = pruner.prune()

                # --- Step 5: Save the Pruned Model ---
                pruner.save_pruned_model(OUTPUT_DIR)
                
                # --- Step 6: Verification (Optional) ---
                print(f"\nVerifying pruned model loading from {OUTPUT_DIR}...")
                
                # Clean up memory before loading the verification model
                del profiler, pruner, pruned_model
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    # Load the pruned model to ensure structural integrity
                    model_v = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto", torch_dtype=DTYPE, trust_remote_code=True)
                    tokenizer_v = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
                    print("Verification successful: Model loaded correctly.")
                    
                    # Test inference
                    test_prompt = "Write a python function to sort a list."
                    try:
                        messages = [{"role": "user", "content": test_prompt}]
                        formatted_text = tokenizer_v.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        formatted_text = test_prompt

                    test_input = tokenizer_v(formatted_text, return_tensors="pt").to(model_v.device)
                    
                    # Generate output (using greedy decoding for verification)
                    output = model_v.generate(**test_input, max_new_tokens=100, do_sample=False, eos_token_id=list(profiler._get_eos_ids()))
                    
                    # Decode only the generated part
                    input_len = test_input.input_ids.shape[1]
                    generated_output = output[0][input_len:]
                    print("\nTest Output:\n", tokenizer_v.decode(generated_output, skip_special_tokens=True))
                    
                except Exception as e:
                    print(f"Verification FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                print("\nSkipping pruning as the task utilizes nearly all experts at the target coverage.")

    except Exception as e:
        print(f"\nAn error occurred during the pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            print("\nHint: The model might be too large for your available VRAM/RAM. Try reducing MAX_NEW_TOKENS or the number of prompts.")

    print("\n✅ Script finished!")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()