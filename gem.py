"""
MoE Expert Analysis and Optimization Suite (Multi-GPU Hybrid Memory Management)
===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import warnings
import gc
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.activations import ACT2FN
from safetensors import safe_open
from functools import partial
from huggingface_hub import snapshot_download

# Requires 'accelerate' library
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    # Import specific cache types for the shim
    from transformers.cache_utils import DynamicCache, Cache
except ImportError:
    print("Error: 'accelerate' or 'transformers' libraries not found or outdated. Please upgrade: pip install --upgrade accelerate transformers")
    init_empty_weights = None
    load_checkpoint_and_dispatch = None
    DynamicCache = None
    Cache = None

warnings.filterwarnings('ignore')

# ==============================================================================
# Compatibility Shim
# ==============================================================================

# ---- Transformers cache compatibility shim (Integrated from the working script) ----
# Fixes compatibility issues when models (like DeepSeek V3.*) use older/internal cache APIs.
if DynamicCache is not None and not hasattr(DynamicCache, "get_usable_length"):
    print("Applying Transformers DynamicCache compatibility shim (get_usable_length)...")
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        # Return the number of tokens already in the cache for this layer.
        # Do NOT try to "reserve room" by subtracting new_seq_length; that can go negative on prefill.
        if hasattr(self, "get_seq_length"):
            try:
                return int(self.get_seq_length(layer_idx))
            except TypeError:
                # Older APIs used no layer argument
                return int(self.get_seq_length())
        # very old fallback
        return int(getattr(self, "seen_tokens", 0))

    DynamicCache.get_usable_length = _get_usable_length
    try:
        if Cache is not None and not hasattr(Cache, "get_usable_length"):
            Cache.get_usable_length = _get_usable_length
    except Exception:
        pass
# ---- end shim ----

# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class TokenExpertLog:
    token_id: int
    token_text: str
    position: int # Position in the generation sequence (0-indexed)
    layer_experts: Dict[int, List[Tuple[int, float]]]
    layer_groups: Dict[int, List[int]] # Added group tracking

@dataclass
class PromptCompletionLog:
    prompt: str
    prompt_idx: int
    generated_text: str
    token_logs: List[TokenExpertLog]
    total_tokens: int

# ==============================================================================
# Helper Classes for Initialization
# ==============================================================================

class GenericMLP(nn.Module):
    """A generic MLP structure (SwiGLU). Used for Hot Experts."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # Determine intermediate size robustly
        self.intermediate_size = getattr(config, 'moe_intermediate_size', None) or \
                                 getattr(config, 'intermediate_size', config.hidden_size * 4)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        act_fn_name = getattr(config, 'hidden_act', 'silu')
        self.act_fn = ACT2FN.get(act_fn_name)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class ExpertPlaceholder(nn.Module):
    """Temporary placeholder used during initialization to skip loading expert weights."""
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        # Dummy parameter so accelerate recognizes the module during dispatch
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x):
        raise RuntimeError("ExpertPlaceholder forward called. Initialization likely failed.")

# ==============================================================================
# Disk Offloading Mechanism (Functional JIT Loading)
# ==============================================================================

class DiskOffloadExpert(nn.Module):
    """Loads weights JIT from safetensors directly to the correct GPU and executes functionally."""
    def __init__(self, config, parameter_map):
        super().__init__()
        self.config = config
        self.parameter_map = parameter_map
        self.act_fn = ACT2FN[getattr(config, 'hidden_act', 'silu')]
        # Register a buffer so the module tracks the correct device after initialization
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def functional_mlp_forward(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]):
        # Assumes SwiGLU structure keys.
        gate_weight = weights.get('gate_proj.weight')
        up_weight = weights.get('up_proj.weight')
        down_weight = weights.get('down_proj.weight')

        if gate_weight is None or up_weight is None or down_weight is None:
             raise RuntimeError("Missing required weights for functional MLP forward.")

        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        activation = self.act_fn(gate) * up
        output = F.linear(activation, down_weight)
        return output

    def forward(self, x: torch.Tensor):
        # The input tensor 'x' dictates the target device and dtype (Crucial for Multi-GPU)
        target_device = x.device
        target_dtype = x.dtype
        weights = {}
        
        files_to_open = defaultdict(list)
        for param_name, (file_path, tensor_key) in self.parameter_map.items():
            # We only need the core weights for the functional forward pass, ignore quantization artifacts
            if "weight" in param_name and "scale" not in param_name and "inv" not in param_name:
                 files_to_open[file_path].append((param_name, tensor_key))

        try:
            # 1. Load weights from Safetensors directly to the target GPU device
            for file_path, keys in files_to_open.items():
                # Load directly onto the specific GPU where computation is happening
                with safe_open(file_path, framework="pt", device=str(target_device)) as f:
                    for param_name, tensor_key in keys:
                        weights[param_name] = f.get_tensor(tensor_key).to(target_dtype)
            
            # 2. Execute functional forward pass
            output = self.functional_mlp_forward(x, weights)
            return output
        
        finally:
            # 3. Explicitly delete weights to free GPU memory immediately
            del weights

# ==============================================================================
# Model Patcher (Handles Multi-GPU Optimization Strategy)
# ==============================================================================

class ModelPatcher:
    """
    Manages the initialization, dispatching, and patching of the MoE model.
    """
    def __init__(self, model_name: str, model: nn.Module, dtype: torch.dtype):
        self.model_name = model_name
        self.model = model
        self.dtype = dtype
        self.config = model.config
        
        print(f"Locating/downloading model snapshot for {model_name}...")
        # This is the directory path needed for accelerate
        self.repo_path = snapshot_download(model_name)
        self._expert_mapping = None
        self._dense_mapping = None
        self.MLPClass = GenericMLP

    def _analyze_checkpoint_structure(self):
        """Analyzes safetensors index to map experts and dense parameters."""
        if self._expert_mapping is not None:
            return 

        print("Analyzing checkpoint structure...")
        index_file = os.path.join(self.repo_path, "model.safetensors.index.json")
        if not os.path.exists(index_file):
             raise FileNotFoundError("Indexed checkpoint (model.safetensors.index.json) required for this workflow.")

        with open(index_file, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})

        expert_mapping = defaultdict(lambda: defaultdict(dict))
        dense_mapping = {}

        # Unified pattern (DeepSeek V3, Qwen, Mixtral)
        expert_pattern = r"model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.e?(\d+)\.(.+)"

        for tensor_key, file_name in weight_map.items():
            file_path = os.path.join(self.repo_path, file_name)
            match = re.match(expert_pattern, tensor_key)
            if match:
                layer_idx, expert_idx, param_name = int(match.group(1)), int(match.group(2)), match.group(3)
                # Store all keys, including quantization scales, in the map for DiskOffloadExpert to filter later
                expert_mapping[(layer_idx, expert_idx)][param_name] = (file_path, tensor_key)
            else:
                dense_mapping[tensor_key] = (file_path, tensor_key)
        
        self._expert_mapping = expert_mapping
        self._dense_mapping = dense_mapping
        print("✓ Checkpoint structure analyzed.")

    def _replace_experts_with_placeholders(self):
        """Replaces expert modules with placeholders in the meta model."""
        print("Replacing experts with temporary placeholders...")
        base_model = self.model.model if hasattr(self.model, 'model') else self.model
        count = 0
        if hasattr(base_model, 'layers'):
            for block in base_model.layers:
                moe_block = getattr(block, 'mlp', None) or getattr(block, 'block_sparse_moe', None)
                if moe_block is not None and hasattr(moe_block, 'experts'):
                    
                    # Robustly determine the number of experts
                    if isinstance(moe_block.experts, (nn.ModuleList, nn.ModuleDict)):
                        num_experts = len(moe_block.experts)
                    elif isinstance(moe_block.experts, ExpertPlaceholder):
                        continue # Already replaced
                    else:
                        # Fallback if 'experts' attribute exists but isn't initialized as a container on meta device
                        num_experts = getattr(self.config, 'n_routed_experts', None) or getattr(self.config, 'num_experts', 0)
                    
                    if num_experts > 0:
                        # Replace the entire container with a single placeholder
                        moe_block.experts = ExpertPlaceholder(num_experts)
                        count += 1
        print(f"✓ Replaced experts in {count} layers.")

    def load_and_dispatch_skeleton(self, device_map="auto"):
        """
        Prepares the model and uses accelerate to load the dense skeleton across GPUs.
        """
        self._analyze_checkpoint_structure()
        
        # 1. Replace experts so accelerate ignores their weights
        self._replace_experts_with_placeholders()
        
        # 2. Prepare for dispatch
        print(f"Loading and dispatching dense skeleton across GPUs (device_map='{device_map}')...")
        
        # Define modules that should not be split across GPUs (Decoder layers)
        no_split_modules = [
            "DeepseekV3DecoderLayer", "DeepseekV2DecoderLayer", "LlamaDecoderLayer", "Qwen2MoeDecoderLayer", "MixtralDecoderLayer"
        ]

        # 3. Accelerate handles the loading and distribution
        # Pass the directory path (self.repo_path).
        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=self.repo_path, # Pass the directory path
            device_map=device_map,
            dtype=self.dtype,
            no_split_module_classes=no_split_modules
        )
        print("✓ Dense skeleton dispatched.")


    def apply_hybrid_optimization(self, hot_expert_set: Dict[int, Set[int]]):
        """
        Restores the expert structure, applying hybrid optimization across the distributed GPUs.
        """
        print("\nApplying Hybrid Expert Optimization Strategy across GPUs...")
        
        base_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        pinned_count = 0
        disk_loaded_count = 0

        if hasattr(base_model, 'layers'):
            for layer_idx, block in enumerate(base_model.layers):
                hot_indices = hot_expert_set.get(layer_idx, set())
                
                moe_block = getattr(block, 'mlp', None) or getattr(block, 'block_sparse_moe', None)

                # Ensure we are dealing with the placeholder inserted earlier
                if moe_block is None or not hasattr(moe_block, 'experts') or not isinstance(moe_block.experts, ExpertPlaceholder):
                    continue

                # Determine the target GPU device for this specific layer
                # We infer this from the router/gate which was already placed by accelerate.
                target_device = None
                if hasattr(moe_block, 'gate'):
                     try:
                         # Get the device of the router parameters
                         target_device = next(moe_block.gate.parameters()).device
                     except StopIteration:
                         pass
                
                # Fallback using shared experts if present (e.g., DeepSeek)
                if target_device is None and hasattr(moe_block, 'shared_experts') and moe_block.shared_experts:
                    try:
                        target_device = next(moe_block.shared_experts.parameters()).device
                    except StopIteration:
                        pass

                if target_device is None:
                    # Final fallback: use the device of the placeholder itself
                    target_device = moe_block.experts.dummy_param.device


                num_experts = moe_block.experts.num_experts
                
                # Reconstruct the expert container (ModuleList or ModuleDict)
                model_type = getattr(self.config, 'model_type', '')
                is_module_dict = 'Qwen' in model_type or 'Mixtral' in model_type
                
                new_experts = nn.ModuleDict() if is_module_dict else nn.ModuleList()

                # Populate the container with the hybrid strategy
                for expert_idx in range(num_experts):
                    # expert_key = f'e{expert_idx}' if is_module_dict else expert_idx
                    param_map = self._expert_mapping.get((layer_idx, expert_idx))
                    
                    if not param_map: continue

                    if expert_idx in hot_indices:
                        # --- Hot Expert: Initialize and Pin to the specific GPU ---
                        # 1. Initialize the MLP structure on the target device
                        expert_mlp = self.MLPClass(self.config).to(target_device).to(self.dtype)
                        
                        # 2. Load weights from disk directly to the target GPU
                        tensors = {}
                        files_to_open_hot = defaultdict(list)
                        for param_name, (file_path, tensor_key) in param_map.items():
                            # We only load the actual weights, ignoring quantization artifacts for the BF16 MLP
                            if "weight" in param_name and "scale" not in param_name and "inv" not in param_name:
                                 files_to_open_hot[file_path].append((param_name, tensor_key))

                        for file_path, keys in files_to_open_hot.items():
                            # Load directly onto the specific GPU
                            with safe_open(file_path, framework="pt", device=str(target_device)) as f:
                                for param_name, tensor_key in keys:
                                    tensors[param_name] = f.get_tensor(tensor_key).to(self.dtype)

                        # Load the collected tensors (strict=True ensures we only load what the GenericMLP expects)
                        expert_mlp.load_state_dict(tensors, strict=True)
                        
                        # 3. Add to the container
                        if is_module_dict:
                            new_experts[f'e{expert_idx}'] = expert_mlp
                        else:
                            new_experts.append(expert_mlp)
                        pinned_count += 1
                    else:
                        # --- Cold Expert: Use DiskOffloadExpert Placeholder ---
                        disk_expert = DiskOffloadExpert(self.config, param_map)
                        # Ensure the placeholder itself (its buffer) is moved to the target device
                        disk_expert.to(target_device) 
                        
                        if is_module_dict:
                            new_experts[f'e{expert_idx}'] = disk_expert
                        else:
                            new_experts.append(disk_expert)
                        disk_loaded_count += 1

                # Replace the ExpertPlaceholder with the finalized expert container
                moe_block.experts = new_experts

        print(f"Optimization Summary:")
        print(f"  - Experts Pinned across GPUs (Hot): {pinned_count}")
        print(f"  - Experts Loaded JIT from Disk (Cold): {disk_loaded_count}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==============================================================================
# Core MoE Expert Logger (Refactored for KV Caching and Group Tracking)
# ==============================================================================
class MoEExpertLogger:
    
    def __init__(self, model_name: str, model: nn.Module, tokenizer: AutoTokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer

        # Determine the primary input device robustly
        try:
            if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
                self.primary_device = model.get_input_embeddings().weight.device
            else:
                self.primary_device = next(model.parameters()).device
        except Exception:
            self.primary_device = model.device
        
        # Storage and Initialization
        self.prompt_logs: List[PromptCompletionLog] = []
        self.expert_activation_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.expert_probability_mass: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # Added group tracking
        self.group_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.group_mass: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        
        # step_buffer: Used to store the output of the hooks for the *current* forward pass.
        self.step_buffer = {} 
        self.moe_layers = []
        self._register_hooks()
        self.config = self._extract_model_config()
        
        # Calculate experts per group based on config
        n_routed = self.config.get('n_routed_experts', 0)
        n_group = self.config.get('n_group', 1)
        self.experts_per_group = n_routed // n_group if n_group > 0 and n_routed > 0 else n_routed

        print(f"✓ Logger initialized. Input device: {self.primary_device}")

    def _extract_model_config(self) -> Dict[str, Any]:
        # Integrated detailed configuration extraction
        cfg = self.model.config
        config = {}
        
        # Core MoE parameters
        config['n_routed_experts'] = getattr(cfg, 'n_routed_experts', None) or getattr(cfg, 'num_experts', None)
        config['num_experts_per_tok'] = getattr(cfg, 'num_experts_per_tok', 8)
        
        # DeepSeek specific routing parameters (with fallbacks)
        config['n_group'] = getattr(cfg, 'n_group', 8)
        config['topk_group'] = getattr(cfg, 'topk_group', 4)
        config['routed_scaling_factor'] = getattr(cfg, 'routed_scaling_factor', 2.5)
        config['norm_topk_prob'] = getattr(cfg, 'norm_topk_prob', True)

        if config['n_routed_experts'] is None:
             # Fallback if config parsing fails
             config['n_routed_experts'] = 256 if len(self.moe_layers) > 0 else 0 
        
        config['total_experts'] = len(self.moe_layers) * (config['n_routed_experts'] or 0)
        return config

    # The hook OVERWRITES the step_buffer entry for the layer
    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            # Generic hook function (DeepSeek/Qwen/Mixtral)
            cls_name = module.__class__.__name__

            if 'MoEGate' in cls_name:
                if isinstance(output, tuple) and len(output) >= 2:
                    # Move captured data to CPU immediately to save VRAM
                    self.step_buffer[layer_idx] = (output[0].detach().cpu(), output[1].detach().cpu())
                return

            if ('Router' in cls_name and 'Mixtral' in cls_name) or ('Gate' in cls_name and 'Qwen' in cls_name):
                router_logits = output.detach().cpu()
                top_k = self.config.get('num_experts_per_tok')
                probs = torch.softmax(router_logits.to(torch.float32), dim=-1)
                topk_weight, topk_idx = torch.topk(probs, top_k, dim=-1)
                if getattr(self.model.config, 'norm_topk_prob', False):
                     topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
                self.step_buffer[layer_idx] = (topk_idx, topk_weight)
        return hook

    def _register_hooks(self):
        # (Updated to use _make_hook)
        print("Registering hooks...")
        found = 0
        base_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(base_model, 'layers'):
            for i, layer in enumerate(base_model.layers):
                for name, module in layer.named_modules():
                    cls_name = module.__class__.__name__
                    # Target the actual gate/router modules
                    if 'MoEGate' in cls_name or ('Router' in cls_name and 'Mixtral' in cls_name) or ('Gate' in cls_name and 'Qwen' in cls_name):
                        hook = self._make_hook(i)
                        module.register_forward_hook(hook)
                        self.moe_layers.append(i)
                        found += 1
                        break
        self.moe_layers.sort()
        print(f"✓ Registered hooks on {found} MoE layers.")

    def _extract_from_step_buffer(self, position: int) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[int]]]:
        """Extracts expert routing and groups from the step buffer for a specific position index."""
        layer_experts = {}
        layer_groups = {}
        
        for L in self.moe_layers:
            if L not in self.step_buffer or self.step_buffer[L] is None:
                continue
            
            topk_idx, topk_weight = self.step_buffer[L]
            
            # Handle different tensor shapes flexibly
            # We expect [Batch*Time, K] or [Time, K] if Batch=1
            if len(topk_idx.shape) == 2:
                T = topk_idx.shape[0]
                # Clamp the position index to the actual sequence length T in the buffer
                p = min(position, T - 1)
                idxs = topk_idx[p]
                wts = topk_weight[p]
            # We might also see [Batch, Time, K]
            elif len(topk_idx.shape) == 3:
                T = topk_idx.shape[1]
                p = min(position, T - 1)
                # Assume batch size 1 during analysis
                idxs = topk_idx[0, p]
                wts = topk_weight[0, p]
            else:
                continue
            
            expert_list = []
            group_set = set()

            for e_id, w in zip(idxs.tolist(), wts.tolist()):
                expert_list.append((int(e_id), float(w)))
                # Calculate group ID
                if self.experts_per_group > 0:
                    g_id = int(e_id) // self.experts_per_group
                    group_set.add(g_id)

            layer_experts[L] = expert_list
            layer_groups[L] = list(group_set)
        
        return layer_experts, layer_groups

    def _update_statistics(self, layer_experts: Dict[int, List[Tuple[int, float]]]):
        """Updates the global statistics counters for experts and groups."""
        for layer_idx, experts_weights in layer_experts.items():
            for e_id, weight in experts_weights:
                self.expert_activation_counts[layer_idx][e_id] += 1
                self.expert_probability_mass[layer_idx][e_id] += weight
                
                # Update group stats
                if self.experts_per_group > 0:
                    g_id = e_id // self.experts_per_group
                    self.group_counts[layer_idx][g_id] += 1
                    self.group_mass[layer_idx][g_id] += weight


    def log_single_generation(self, prompt: str, prompt_idx: int = 0, max_new_tokens: int = 50, temperature: float = 0.0):
        """
        Generates completion using efficient KV caching and logs expert activations per step.
        """
        # Prompt formatting (Robust handling)
        try:
            messages = [{"role": "user", "content": prompt}]
            if "DeepSeek" in self.model_name:
                try:
                    formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=False)
                except TypeError:
                     formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_text = prompt
        
        # Tokenize and move inputs to the primary device
        inputs = self.tokenizer(formatted_text, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs["input_ids"].to(self.primary_device)
        attn_mask = inputs["attention_mask"].to(self.primary_device)

        B, L = input_ids.shape
        device = self.primary_device
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        token_logs = []
        generated_ids = []

        # Prepare attention mask for generation (avoids torch.cat in loop)
        full_attn_mask = torch.ones((B, L + max_new_tokens), dtype=attn_mask.dtype, device=device)
        full_attn_mask[:, :L] = attn_mask

        self.model.eval()
        with torch.inference_mode():
            # --- 1) Prefill Phase ---
            self.step_buffer.clear() # Clear buffer before forward pass
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values

            # Determine next token (greedy sampling for analysis)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = next_id.item()
            
            # Extract routing for the *last* token of the prompt (which produced next_id)
            # The buffer contains activations for all L tokens. We want index L-1.
            layer_experts, layer_groups = self._extract_from_step_buffer(L - 1)
            self._update_statistics(layer_experts)

            # Log the first generated token
            token_logs.append(TokenExpertLog(
                token_id=tok_id,
                token_text=self.tokenizer.decode([tok_id], skip_special_tokens=True),
                position=0,
                layer_experts=layer_experts,
                layer_groups=layer_groups
            ))
            generated_ids.append(tok_id)

            if tok_id == eos_id:
                pass
            else:
                # --- 2) Decoding Phase ---
                for step in range(1, max_new_tokens):
                    cur_len = L + step
                    
                    self.step_buffer.clear() # Clear buffer before forward pass
                    out = self.model(
                        input_ids=next_id, # Feed just the last token
                        attention_mask=full_attn_mask[:, :cur_len],
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values

                    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    tok_id = next_id.item()

                    # Extract routing for this single token.
                    # The buffer contains activation for 1 token. We want index 0.
                    layer_experts, layer_groups = self._extract_from_step_buffer(0)
                    self._update_statistics(layer_experts)

                    # Log the generated token
                    token_logs.append(TokenExpertLog(
                        token_id=tok_id,
                        token_text=self.tokenizer.decode([tok_id], skip_special_tokens=True),
                        position=step,
                        layer_experts=layer_experts,
                        layer_groups=layer_groups
                    ))
                    generated_ids.append(tok_id)

                    if tok_id == eos_id:
                        break
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return PromptCompletionLog(prompt=prompt, prompt_idx=prompt_idx, generated_text=generated_text, token_logs=token_logs, total_tokens=len(token_logs))


    def process_prompts(self, prompts: List[str], max_new_tokens: int = 50):
        print(f"\nProcessing {len(prompts)} prompts (using optimized KV Caching loop)...")
        logs = []
        for i, prompt in enumerate(prompts):
            print(f"  Progress: {i + 1}/{len(prompts)}")
            log = self.log_single_generation(prompt, i, max_new_tokens)
            logs.append(log)
            self.prompt_logs.append(log)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"✓ Completed processing.")
        return logs

# ==============================================================================
# MoE Analyzer (With Hot Set Identification and Group Analysis)
# ==============================================================================
class MoEAnalyzer:
    def __init__(self, logger: MoEExpertLogger):
        self.logger = logger

    def identify_hot_expert_set(self, coverage_threshold: float = 0.95, min_experts: Optional[int] = None) -> Dict[int, Set[int]]:
        """
        Identifies the minimal set of experts required to cover a percentage of the total probability mass per layer.
        """
        if min_experts is None:
            min_experts = self.logger.config.get('num_experts_per_tok', 8)

        hot_expert_set = {}
        print(f"\nIdentifying Hot Expert Set (Coverage: {coverage_threshold*100:.1f}%)")

        total_hot_experts = 0
        for layer_idx in self.logger.moe_layers:
            layer_mass = self.logger.expert_probability_mass.get(layer_idx, {})
            if not layer_mass:
                hot_expert_set[layer_idx] = set(range(min_experts))
                continue

            # Sort experts by probability mass contribution
            sorted_experts = sorted(layer_mass.items(), key=lambda item: item[1], reverse=True)
            total_mass = sum(mass for _, mass in sorted_experts)
            
            current_mass = 0
            hot_set = set()
            
            # Select experts until threshold is met OR minimum count is reached
            for expert_idx, mass in sorted_experts:
                if current_mass < total_mass * coverage_threshold or len(hot_set) < min_experts:
                    hot_set.add(expert_idx)
                    current_mass += mass
                else:
                    break
            
            hot_expert_set[layer_idx] = hot_set
            total_hot_experts += len(hot_set)

        print(f"Total Hot Experts identified globally: {total_hot_experts}")
        return hot_expert_set

    def export_hot_set(self, output_path: str, hot_set: Dict[int, Set[int]]):
        export_data = {
            'model_name': self.logger.model_name,
            'hot_expert_set': {k: list(v) for k, v in hot_set.items()}
        }
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Hot Expert Set exported to {output_path}")
    
    def analyze_group_utilization(self):
        # Added analysis based on the provided script's output format
        print(f"\n[Group Utilization Analysis]")
        all_groups = set()
        for layer_idx in self.logger.group_counts:
            for g_id in self.logger.group_counts[layer_idx]:
                all_groups.add(g_id)
        
        n_group = self.logger.config.get('n_group', 0)
        print(f"  Unique groups used (across all layers): {len(all_groups)} / {n_group}")
        print(f"  Groups IDs: {sorted(all_groups)}")


# ==============================================================================
# Workflow Function (Multi-GPU)
# ==============================================================================

def run_analysis_and_optimization(
    model_name: str,
    prompts: List[str],
    task_name: str = "domain_optimization_multigpu",
    max_new_tokens: int = 50,
    optimization_coverage: float = 0.98,
    output_dir: Optional[str] = None,
    device_map="auto", # Use "auto" or "balanced" to utilize all available GPUs
    dtype: torch.dtype = torch.bfloat16
):
    """
    Two-Stage Workflow (Multi-GPU):
    1. Analyze expert usage (using full disk offloading across GPUs).
    2. Re-initialize the model with Hybrid Optimization based on the analysis.
    """
    
    if init_empty_weights is None or load_checkpoint_and_dispatch is None:
        raise ImportError("This workflow requires the 'accelerate' library.")

    print(f"\n{'='*70}")
    print(f"MoE Domain Optimization Workflow (Multi-GPU): {task_name}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"{'='*70}")
    
    if output_dir is None:
        output_dir = f"./{model_name.replace('/', '_')}_{task_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load config and tokenizer (trust_remote_code=True is essential for DeepSeek)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # Stage 1: Analysis Phase (Full Disk Offload)
    # =====================================================
    print("\n--- Stage 1: Analyzing Domain Expert Usage (Full Disk Offload across GPUs) ---")
    
    # 1. Initialize empty structure (meta device)
    print("Initializing empty model structure (meta device)...")
    with init_empty_weights():
            # Ensure the model initialized from config also trusts remote code
            analysis_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()

    # 2. Initialize Patcher
    patcher = ModelPatcher(model_name, analysis_model, dtype)
    
    # 3. Load and dispatch dense skeleton across GPUs
    # This replaces experts with placeholders internally and uses accelerate dispatch
    patcher.load_and_dispatch_skeleton(device_map=device_map)
    
    # 4. Apply optimization with an empty hot set (forces 100% DiskOffloadExpert usage)
    # This replaces the placeholders with DiskOffloadExperts on the respective GPUs.
    patcher.apply_hybrid_optimization(hot_expert_set={})
    
    # 5. Initialize Logger and run analysis
    logger = MoEExpertLogger(model_name, analysis_model, tokenizer)
    
    # Run analysis (Distributed, but slow due to JIT disk loading)
    logger.process_prompts(prompts=prompts, max_new_tokens=max_new_tokens)
    
    analyzer = MoEAnalyzer(logger)
    
    # Analyze Group Utilization
    analyzer.analyze_group_utilization()

    # Identify Hot Set
    hot_expert_set = analyzer.identify_hot_expert_set(coverage_threshold=optimization_coverage)
    analyzer.export_hot_set(f"{output_dir}/hot_expert_set.json", hot_set=hot_expert_set)
    
    # Clean up the analysis model
    print("Cleaning up analysis resources...")
    del analysis_model, logger, analyzer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================
    # Stage 2: Optimization Phase (Hybrid Loading)
    # =====================================================
    print("\n--- Stage 2: Applying Hybrid Optimization (Pinned Hot Experts across GPUs) ---")
    
    # 1. Initialize empty structure again
    print("Initializing optimized model structure...")
    with init_empty_weights():
            optimized_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()

    # 2. Re-initialize Patcher (reuse analysis from previous patcher)
    optimized_patcher = ModelPatcher(model_name, optimized_model, dtype)
    optimized_patcher._expert_mapping = patcher._expert_mapping
    optimized_patcher._dense_mapping = patcher._dense_mapping
    
    # 3. Load and dispatch dense skeleton across GPUs
    optimized_patcher.load_and_dispatch_skeleton(device_map=device_map)
    
    # 4. Apply hybrid optimization using the identified hot set
    # This replaces placeholders: pins the hot set to the respective GPUs and uses DiskOffloadExperts for the cold set.
    optimized_patcher.apply_hybrid_optimization(hot_expert_set=hot_expert_set)

    print("\n✓ Workflow Complete. Optimized model is ready for use across GPUs with a fixed VRAM budget.")
    return optimized_model, tokenizer

# ==============================================================================
# Example Usage (Multi-GPU Hybrid Optimization Workflow)
# ==============================================================================
if __name__ == "__main__":
    
    # Define your domain-specific prompts
    domain_prompts = [
        "Analyze the complexity of the QuickSort algorithm.",
        "Explain how MergeSort works and why it is stable.",
        "What is the difference between Dijkstra's algorithm and A* search?",
        "Describe the time and space complexity of a Hash Table.",
        "Implement a Python function for Depth First Search (DFS) on a graph.",
        "How does a Bloom Filter work and where is it commonly used?",
    ]
    
    # NOTE: Ensure you have multiple GPUs available.
    if torch.cuda.device_count() < 1:
        print("Error: At least one CUDA-enabled GPU is required.")
    else:
        print(f"Detected {torch.cuda.device_count()} GPUs. Starting workflow.")

        # --- Configuration ---
        
        # To run the full DeepSeek V3.1 (671B):
        MODEL_NAME = "deepseek-ai/DeepSeek-V3.1" 
        
        # Recommended: Test with a smaller model first (e.g., DeepSeek V2 Lite 15.6B)
        # MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite" 
        # ---------------------

        # Run the analysis and optimization workflow
        optimized_model, tokenizer = run_analysis_and_optimization(
            model_name=MODEL_NAME,
            prompts=domain_prompts,
            task_name="algorithms_domain_multigpu_final",
            max_new_tokens=50, # Keep low during analysis phase as JIT loading is slow
            optimization_coverage=0.98,
            device_map="auto", # <<< Utilize all available GPUs
            dtype=torch.bfloat16
        )

        # Example inference with the optimized, dispatched model
        if optimized_model:
            print("\n--- Testing Optimized Model Inference (Multi-GPU) ---")
            
            # Determine input device robustly
            input_device = optimized_model.device
            try:
                if optimized_model.get_input_embeddings():
                     input_device = optimized_model.get_input_embeddings().weight.device
            except:
                pass

            inputs = tokenizer("What is the complexity of Bubble Sort?", return_tensors="pt").to(input_device)
            
            # Inference uses the distributed hybrid strategy
            with torch.no_grad():
                 outputs = optimized_model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
            
            print("\nResponse:")
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

            # Final cleanup
            del optimized_model
            gc.collect()
            torch.cuda.empty_cache()