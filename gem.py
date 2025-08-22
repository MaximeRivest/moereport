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
import types # Required for monkey-patching
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.activations import ACT2FN
from safetensors import safe_open
from functools import partial
from huggingface_hub import snapshot_download

# Requires 'accelerate' library
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers.cache_utils import Cache # Required for type hinting in the patch
except ImportError:
    print("Error: 'accelerate' or 'transformers' libraries not found or outdated. Please upgrade: pip install --upgrade accelerate transformers")
    init_empty_weights = None
    load_checkpoint_and_dispatch = None
    Cache = None

warnings.filterwarnings('ignore')

# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class TokenExpertLog:
    token_id: int
    token_text: str
    position: int
    prompt_idx: int
    layer_experts: Dict[int, List[Tuple[int, float]]]

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
        # FIX: Pass the directory path (self.repo_path), not a list of files.
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
                    expert_key = f'e{expert_idx}' if is_module_dict else expert_idx
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
                            new_experts[expert_key] = expert_mlp
                        else:
                            new_experts.append(expert_mlp)
                        pinned_count += 1
                    else:
                        # --- Cold Expert: Use DiskOffloadExpert Placeholder ---
                        disk_expert = DiskOffloadExpert(self.config, param_map)
                        # Ensure the placeholder itself (its buffer) is moved to the target device
                        disk_expert.to(target_device) 
                        
                        if is_module_dict:
                            new_experts[expert_key] = disk_expert
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
# Core MoE Expert Logger (Multi-GPU Compatible)
# ==============================================================================
class MoEExpertLogger:
    
    def __init__(self, model_name: str, model: nn.Module, tokenizer: AutoTokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer

        # Determine the primary input device (where inputs should be sent)
        # FIX: Robustly determine the device by checking the actual location of the input embeddings.
        try:
            if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
                self.primary_device = model.get_input_embeddings().weight.device
            else:
                # Fallback for models without standard embedding access or if embeddings are None
                self.primary_device = next(model.parameters()).device
        except Exception:
            print("Warning: Could not determine primary device from model parameters. Defaulting to model.device.")
            self.primary_device = model.device
        
        # Storage and Initialization
        self.prompt_logs: List[PromptCompletionLog] = []
        self.expert_activation_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.expert_probability_mass: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.hook_storage = defaultdict(list)
        self.moe_layers = []
        self._register_hooks()
        self.config = self._extract_model_config()
        print(f"✓ Logger initialized. Input device: {self.primary_device}")

    def _extract_model_config(self) -> Dict[str, Any]:
        # (Implementation remains the same as previous versions)
        cfg = self.model.config
        config = {}
        config['num_experts'] = (
            getattr(cfg, 'n_routed_experts', None) or getattr(cfg, 'num_experts', None)
        )
        config['top_k'] = getattr(cfg, 'num_experts_per_tok', 8)
        if config['num_experts'] is None:
             config['num_experts'] = 256 if len(self.moe_layers) > 0 else 0 
        config['total_experts'] = len(self.moe_layers) * (config['num_experts'] or 0)
        return config

    def _hook_fn(self, module, input, output, layer_idx):
        # (Implementation remains the same as previous versions)
        cls_name = module.__class__.__name__

        if 'MoEGate' in cls_name:
            if isinstance(output, tuple) and len(output) >= 2:
                self.hook_storage[layer_idx].append((output[0].detach().cpu(), output[1].detach().cpu()))
            return

        if ('Router' in cls_name and 'Mixtral' in cls_name) or ('Gate' in cls_name and 'Qwen' in cls_name):
            router_logits = output.detach().cpu()
            top_k = self.config.get('top_k')
            probs = torch.softmax(router_logits.to(torch.float32), dim=-1)
            topk_weight, topk_idx = torch.topk(probs, top_k, dim=-1)
            if getattr(self.model.config, 'norm_topk_prob', False):
                 topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
            self.hook_storage[layer_idx].append((topk_idx, topk_weight))

    def _register_hooks(self):
        # (Implementation remains the same as previous versions)
        print("Registering hooks...")
        found = 0
        base_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(base_model, 'layers'):
            for i, layer in enumerate(base_model.layers):
                for name, module in layer.named_modules():
                    cls_name = module.__class__.__name__
                    if 'MoEGate' in cls_name or ('Router' in cls_name and 'Mixtral' in cls_name) or ('Gate' in cls_name and 'Qwen' in cls_name):
                        hook = partial(self._hook_fn, layer_idx=i)
                        module.register_forward_hook(hook)
                        self.moe_layers.append(i)
                        found += 1
                        break
        self.moe_layers.sort()
        print(f"✓ Registered hooks on {found} MoE layers.")

    def log_single_generation(self, prompt: str, prompt_idx: int = 0, max_new_tokens: int = 50, temperature: float = 0.0):
        # Prompt formatting (Robust handling)
        try:
            messages = [{"role": "user", "content": prompt}]
            if "DeepSeek" in self.model_name:
                try:
                    # Use non-thinking mode for analysis
                    formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=False)
                except TypeError:
                     formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_text = prompt
        
        # Move inputs to the primary device (required for multi-GPU)
        inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.primary_device)
        self.hook_storage.clear()

        # Generation
        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
            )
        
        # Decoding and Processing
        output_ids = generation_output[0]
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[input_len:]
        # Move generated IDs to CPU for processing
        generated_ids_cpu = generated_ids.cpu()
        generated_text = self.tokenizer.decode(generated_ids_cpu, skip_special_tokens=True)

        token_logs = self._process_hook_data(generated_ids_cpu, prompt_idx)
        return PromptCompletionLog(prompt=prompt, prompt_idx=prompt_idx, generated_text=generated_text, token_logs=token_logs, total_tokens=len(token_logs))

    def _process_hook_data(self, generated_ids: torch.Tensor, prompt_idx: int) -> List[TokenExpertLog]:
        # (Implementation remains the same as previous versions)
        num_generated_tokens = len(generated_ids)
        token_data = defaultdict(dict)

        for layer_idx in self.moe_layers:
            captures = self.hook_storage.get(layer_idx, [])
            if not captures: continue

            valid_captures = [c for c in captures if len(c) == 2 and c[0].numel() > 0 and c[1].numel() > 0]
            if not valid_captures: continue

            try:
                all_indices = torch.cat([c[0] for c in valid_captures], dim=0)
                all_weights = torch.cat([c[1] for c in valid_captures], dim=0)
            except RuntimeError as e:
                print(f"Warning: Error concatenating hook data at layer {layer_idx}. Skipping layer. Error: {e}")
                continue

            if all_indices.shape[0] >= num_generated_tokens:
                gen_indices = all_indices[-num_generated_tokens:]
                gen_weights = all_weights[-num_generated_tokens:]
                for pos in range(num_generated_tokens):
                    experts_weights = [(idx.item(), weight.item()) for idx, weight in zip(gen_indices[pos], gen_weights[pos])]
                    token_data[pos][layer_idx] = experts_weights

        token_logs = []
        for pos in range(num_generated_tokens):
            token_id = generated_ids[pos].item()
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            layer_experts = token_data.get(pos, {})
            token_logs.append(TokenExpertLog(token_id=token_id, token_text=token_text, position=pos, prompt_idx=prompt_idx, layer_experts=layer_experts))
            
            for layer_idx, experts_weights in layer_experts.items():
                for expert_idx, weight in experts_weights:
                    self.expert_activation_counts[layer_idx][expert_idx] += 1
                    self.expert_probability_mass[layer_idx][expert_idx] += weight
        return token_logs

    def process_prompts(self, prompts: List[str], max_new_tokens: int = 50):
        print(f"\nProcessing {len(prompts)} prompts...")
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
# MoE Analyzer (With Hot Set Identification)
# ==============================================================================
class MoEAnalyzer:
    # (Implementation remains the same as previous versions)
    def __init__(self, logger: MoEExpertLogger):
        self.logger = logger

    def identify_hot_expert_set(self, coverage_threshold: float = 0.95, min_experts: Optional[int] = None) -> Dict[int, Set[int]]:
        if min_experts is None:
            min_experts = self.logger.config.get('top_k', 8)

        hot_expert_set = {}
        print(f"\nIdentifying Hot Expert Set (Coverage: {coverage_threshold*100:.1f}%)")

        total_hot_experts = 0
        for layer_idx in self.logger.moe_layers:
            layer_mass = self.logger.expert_probability_mass.get(layer_idx, {})
            if not layer_mass:
                hot_expert_set[layer_idx] = set(range(min_experts))
                continue

            sorted_experts = sorted(layer_mass.items(), key=lambda item: item[1], reverse=True)
            total_mass = sum(mass for _, mass in sorted_experts)
            
            current_mass = 0
            hot_set = set()
            
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

# ==============================================================================
# Workflow Function (Multi-GPU with Patching)
# ==============================================================================

def patch_model_for_cache_api(model):
    """
    Applies a monkey patch for compatibility with the latest transformers DynamicCache API.
    Fixes: AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
    """

    # Check if the model architecture might require patching (specifically DeepSeek models using remote code)
    if "deepseek" not in getattr(model.config, 'model_type', ''):
        return model
    
    if Cache is None:
        print("Warning: Cache utils not imported. Cannot apply DynamicCache patch.")
        return model

    print("Applying compatibility patch for DynamicCache API (prepare_inputs_for_generation)...")

    # Define the corrected method (needs 'self' as the first argument)
    # This definition is based on the standard implementation, correcting the access to past_length.
    def corrected_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                # --- FIX START ---
                # The original remote code used past_key_values.seen_tokens which caused the error.
                # We use the standard API get_seq_length() instead.
                past_length = past_key_values.get_seq_length()
                # --- FIX END ---
                max_cache_length = past_key_values.get_max_length()
            else:
                # Legacy tuple cache
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens (standard logic)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # Apply the patch: Replace the instance method with the corrected one
    model.prepare_inputs_for_generation = types.MethodType(corrected_prepare_inputs_for_generation, model)
    return model


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

    # =====================================================
    # Stage 1: Analysis Phase (Full Disk Offload)
    # =====================================================
    print("\n--- Stage 1: Analyzing Domain Expert Usage (Full Disk Offload across GPUs) ---")
    
    # 1. Initialize empty structure (meta device)
    print("Initializing empty model structure (meta device)...")
    with init_empty_weights():
            # Ensure the model initialized from config also trusts remote code
            analysis_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()

    # 1.1 Apply compatibility patch for transformers caching API
    analysis_model = patch_model_for_cache_api(analysis_model)

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

    # 1.1 Apply compatibility patch again
    optimized_model = patch_model_for_cache_api(optimized_model)

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
        # "Explain how MergeSort works and why it is stable.",
        # "What is the difference between Dijkstra's algorithm and A* search?",
        # "Describe the time and space complexity of a Hash Table.",
        # "Implement a Python function for Depth First Search (DFS) on a graph.",
        # "How does a Bloom Filter work and where is it commonly used?",
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
            task_name="algorithms_domain_multigpu",
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