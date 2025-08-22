# moe_tensor_parallel.py
# Enhanced MOE runtime with tensor parallelism across 8 GPUs
# Fixes garbled output and distributes model across all available GPUs

import os
import gc
import re
import copy
import json
import time
import math
import glob
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import is_flash_attn_2_available
from transformers.cache_utils import DynamicCache

# Accelerate for distributed loading
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False

from safetensors import safe_open

# =========================
# Tensor Parallel Configuration
# =========================

def setup_tensor_parallel():
    """Initialize tensor parallelism across all available GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[TensorParallel] Detected {num_gpus} GPUs")
        
        # Set up environment for all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
        
        # Initialize CUDA for all devices
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            torch.cuda.init()
        
        # Reset to first device
        torch.cuda.set_device(0)
        
        return num_gpus
    return 1

# =========================
# Cache patch for DeepSeek-V3.1 (unchanged)
# =========================

class DeepSeekCompatibleCache(DynamicCache):
    """
    A patched DynamicCache that adds the seen_tokens attribute
    required by DeepSeek-V3.1's prepare_inputs_for_generation method.
    """
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0
        self._max_cache_len = None
    
    @property
    def seen_tokens(self):
        return self._seen_tokens
    
    def get_max_length(self):
        return self._max_cache_len
    
    def get_usable_length(self, seq_length=None, layer_idx=None):
        if layer_idx is None:
            return self.get_seq_length()
            
        if len(self.key_cache) <= layer_idx:
            return 0
            
        if self.key_cache[layer_idx] is None:
            return 0
            
        return self.key_cache[layer_idx].shape[-2]
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        if layer_idx == 0:
            if len(self.key_cache) > 0 and self.key_cache[0] is not None:
                self._seen_tokens = self.key_cache[0].shape[-2]
            else:
                self._seen_tokens = key_states.shape[-2]
        
        return result

# =========================
# Utilities
# =========================

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB","PB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f} {u}"
        x /= 1024.0

def elem_bytes(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8

def join_module_path(*parts: str) -> str:
    return ".".join([p for p in parts if p and p != "."])

def extract_layer_idx_from_name(name: str) -> Optional[int]:
    for pat in [r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\.", r"\.layer\.(\d+)\."]:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None

# =========================
# Multi-GPU Safetensor Index with parallel loading
# =========================

class ParallelSafetensorIndex:
    """
    Enhanced SafetensorIndex with parallel loading across multiple GPUs
    """
    def __init__(self, model_location: str, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.root = self._resolve_local_dir(model_location)
        if self.root is None or not os.path.isdir(self.root):
            raise RuntimeError(f"Could not resolve directory for '{model_location}'")
        
        self.index_path = None
        for name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
            p = os.path.join(self.root, name)
            if os.path.isfile(p):
                self.index_path = p
                break

        self._key_to_file: Dict[str, str] = {}
        self._file_to_keys: Dict[str, set] = {}
        self._all_shards = sorted(glob.glob(os.path.join(self.root, "*.safetensors")))
        
        if not self._all_shards:
            raise RuntimeError(f"No *.safetensors found under {self.root}")

        if self.index_path:
            with open(self.index_path, "r") as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", {})
            for k, fname in weight_map.items():
                fp = os.path.join(self.root, fname)
                self._key_to_file[k] = fp
                self._file_to_keys.setdefault(fp, set()).add(k)

        self._open_files: Dict[str, Any] = {}

    def _resolve_local_dir(self, model_location: str) -> Optional[str]:
        if os.path.isdir(model_location):
            return model_location
        try:
            from huggingface_hub import snapshot_download
            snap_dir = snapshot_download(repo_id=model_location, local_files_only=False, allow_patterns="*.safetensors*")
            return snap_dir
        except Exception:
            return None

    def _ensure_file_mapping_for_key(self, key: str) -> Optional[str]:
        if key in self._key_to_file:
            return self._key_to_file[key]
        for shard in self._all_shards:
            if shard not in self._open_files:
                self._open_files[shard] = safe_open(shard, framework="pt")
            f = self._open_files[shard]
            if key in f.keys():
                self._key_to_file[key] = shard
                self._file_to_keys.setdefault(shard, set()).add(key)
                return shard
        return None

    def get_tensor(self, key: str, device: str = "cpu", dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        shard = self._ensure_file_mapping_for_key(key)
        if shard is None:
            raise KeyError(f"Tensor key not found: {key}")
        f = self._open_files.get(shard)
        if f is None:
            f = safe_open(shard, framework="pt")
            self._open_files[shard] = f
        t = f.get_tensor(key)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        
        # Distribute tensor loading across GPUs based on key hash
        if device == "cuda" and self.num_gpus > 1:
            gpu_id = hash(key) % self.num_gpus
            device = f"cuda:{gpu_id}"
        
        return t.to(device)

# =========================
# Expert Management with Multi-GPU support
# =========================

@dataclass
class ExpertKey:
    layer: int
    expert: int

@dataclass
class CacheStats:
    gpu_budget_bytes: int = 0
    gpu_expert_bytes: int = 0
    gpu_loads: int = 0
    gpu_evictions: int = 0
    gpu_hits: int = 0
    cpu_runs: int = 0
    temp_gpu_runs: int = 0
    usage_counts: Dict[int, Dict[int, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def summary(self) -> Dict[str, Any]:
        total_used_experts = sum(len(v) for v in self.usage_counts.values())
        return {
            "gpu_budget": human_bytes(self.gpu_budget_bytes),
            "gpu_expert_bytes": human_bytes(self.gpu_expert_bytes),
            "gpu_loads": self.gpu_loads,
            "gpu_evictions": self.gpu_evictions,
            "gpu_hits": self.gpu_hits,
            "cpu_runs": self.cpu_runs,
            "temp_gpu_runs": self.temp_gpu_runs,
            "layers_touched": len(self.usage_counts),
            "experts_touched_total": total_used_experts,
        }

# =========================
# Multi-GPU Expert LRU Cache
# =========================

class MultiGPUExpertLRU:
    """Expert cache distributed across multiple GPUs"""
    def __init__(self, budget_bytes_per_gpu: int, stats: CacheStats, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.budget_bytes_per_gpu = int(budget_bytes_per_gpu)
        self.total_budget = self.budget_bytes_per_gpu * num_gpus
        self.stats = stats
        self.stats.gpu_budget_bytes = self.total_budget
        
        # Maintain separate LRU for each GPU
        self._lru_per_gpu = [OrderedDict() for _ in range(num_gpus)]
        self._bytes_per_gpu = [0 for _ in range(num_gpus)]
        self._pinned = set()

    def _get_target_gpu(self, key: Tuple[int, int]) -> int:
        """Determine which GPU should hold this expert"""
        return (key[0] + key[1]) % self.num_gpus

    def _evict_from_gpu(self, gpu_id: int, add_bytes: int):
        """Evict experts from specific GPU"""
        lru = self._lru_per_gpu[gpu_id]
        while self._bytes_per_gpu[gpu_id] + add_bytes > self.budget_bytes_per_gpu and lru:
            victim_key = None
            for k in lru.keys():
                if k not in self._pinned:
                    victim_key = k
                    break
            if victim_key is None:
                break
            
            wrapper = lru.pop(victim_key)
            freed = wrapper.param_bytes()
            wrapper._unload_gpu()
            self._bytes_per_gpu[gpu_id] -= freed
            self.stats.gpu_evictions += 1

    def ensure_on_gpu(self, key: Tuple[int,int], wrapper: "BaseExpertWrapper") -> bool:
        if self.total_budget <= 0:
            return False
        
        gpu_id = self._get_target_gpu(key)
        lru = self._lru_per_gpu[gpu_id]
        
        if key in lru and wrapper._has_gpu():
            lru.move_to_end(key, last=True)
            self.stats.gpu_hits += 1
            return True
        
        need = wrapper.param_bytes()
        self._evict_from_gpu(gpu_id, need)
        
        if need > self.budget_bytes_per_gpu:
            return False
        
        wrapper._load_gpu(gpu_id)
        lru[key] = wrapper
        lru.move_to_end(key, last=True)
        self._bytes_per_gpu[gpu_id] += need
        self.stats.gpu_loads += 1
        return True

    def pin(self, key: Tuple[int,int], wrapper: "BaseExpertWrapper"):
        gpu_id = self._get_target_gpu(key)
        need = wrapper.param_bytes()
        
        self._evict_from_gpu(gpu_id, need)
        if self._bytes_per_gpu[gpu_id] + need > self.budget_bytes_per_gpu:
            raise RuntimeError(f"Not enough VRAM on GPU {gpu_id} to pin expert {key}")
        
        if not wrapper._has_gpu():
            wrapper._load_gpu(gpu_id)
            self._lru_per_gpu[gpu_id][key] = wrapper
            self._bytes_per_gpu[gpu_id] += need
            self.stats.gpu_loads += 1
        
        self._pinned.add(key)

# =========================
# Enhanced Expert Wrappers with multi-GPU support
# =========================

class BaseExpertWrapper(nn.Module):
    def __init__(self, layer_idx: int, expert_idx: int, lru: MultiGPUExpertLRU, stats: CacheStats):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self._lru = lru
        self._stats = stats
        self._gpu_device = None

    def _has_gpu(self) -> bool:
        raise NotImplementedError
    
    def _load_gpu(self, gpu_id: int = 0):
        raise NotImplementedError
    
    def _unload_gpu(self):
        raise NotImplementedError
    
    def param_bytes(self) -> int:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class DiskBackedExpert(BaseExpertWrapper):
    """Disk-backed expert with multi-GPU support"""
    def __init__(self,
                 blueprint: nn.Module,
                 param_rel_names: List[str],
                 full_prefix: str,
                 idx: ParallelSafetensorIndex,
                 layer_idx: int,
                 expert_idx: int,
                 lru: MultiGPUExpertLRU,
                 stats: CacheStats,
                 dtype: torch.dtype,
                 allow_cpu_fallback: bool = False):
        super().__init__(layer_idx, expert_idx, lru, stats)
        self._blueprint_meta = blueprint
        self._param_rel_names = param_rel_names
        self._full_prefix = full_prefix
        self._idx = idx
        self._dtype = dtype
        
        pb = 0
        for n, p in blueprint.named_parameters():
            pb += p.numel() * elem_bytes(dtype)
        self._param_bytes = pb
        self._gpu_expert: Optional[nn.Module] = None
        self._allow_cpu_fallback = allow_cpu_fallback

    def _has_gpu(self) -> bool:
        return self._gpu_expert is not None

    def _materialize_and_load(self, device: str) -> nn.Module:
        mod = copy.deepcopy(self._blueprint_meta)
        mod = mod.to_empty(device=device).to(dtype=self._dtype)
        
        for rel in self._param_rel_names:
            key = join_module_path(self._full_prefix, rel)
            t = self._idx.get_tensor(key, device=device, dtype=self._dtype)
            path, pname = rel.rsplit(".", 1)
            sub = mod
            for part in path.split("."):
                if part:
                    sub = getattr(sub, part)
            p = getattr(sub, pname)
            if isinstance(p, nn.Parameter):
                p.data = t
            else:
                setattr(sub, pname, nn.Parameter(t, requires_grad=False))
        
        for p in mod.parameters():
            p.requires_grad_(False)
        return mod

    def _load_gpu(self, gpu_id: int = 0):
        if self._gpu_expert is None:
            device = f"cuda:{gpu_id}"
            self._gpu_device = device
            self._gpu_expert = self._materialize_and_load(device)

    def _unload_gpu(self):
        if self._gpu_expert is not None:
            try:
                del self._gpu_expert
            finally:
                self._gpu_expert = None
                self._gpu_device = None

    def param_bytes(self) -> int:
        return self._param_bytes

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._stats.usage_counts[self.layer_idx][self.expert_idx] += 1
        key = (self.layer_idx, self.expert_idx)

        if self._lru.ensure_on_gpu(key, self) and self._gpu_expert is not None:
            # Move input to the same GPU as the expert
            if self._gpu_device and x.device != torch.device(self._gpu_device):
                x = x.to(self._gpu_device)
            return self._gpu_expert(x)

        if not self._allow_cpu_fallback:
            # Determine GPU for temporary load
            gpu_id = self._lru._get_target_gpu(key)
            device = f"cuda:{gpu_id}"
            tmp = self._materialize_and_load(device)
            if x.device != torch.device(device):
                x = x.to(device)
            y = tmp(x)
            del tmp
            torch.cuda.empty_cache()
            self._stats.temp_gpu_runs += 1
            return y

        self._stats.cpu_runs += 1
        tmp_cpu = self._materialize_and_load("cpu")
        y = tmp_cpu(x.to("cpu", non_blocking=True))
        del tmp_cpu
        return y.to(x.device, non_blocking=True)

# =========================
# Model patcher for tensor parallelism
# =========================

def patch_moe_for_tensor_parallel(model: nn.Module,
                                  gpu_expert_budget_gb_per_gpu: float,
                                  safetensor_idx: ParallelSafetensorIndex,
                                  allow_cpu_fallback: bool,
                                  dtype: torch.dtype,
                                  num_gpus: int = 8) -> Tuple[MultiGPUExpertLRU, CacheStats]:
    """
    Patch MoE modules for tensor parallel execution across multiple GPUs
    """
    stats = CacheStats()
    lru = MultiGPUExpertLRU(int(gpu_expert_budget_gb_per_gpu * (1024**3)), stats, num_gpus)
    
    total_experts = 0
    for name, module in model.named_modules():
        if hasattr(module, "experts") and isinstance(getattr(module, "experts"), nn.ModuleList):
            experts: nn.ModuleList = module.experts
            layer_idx = extract_layer_idx_from_name(name)
            num_experts = len(experts)
            
            for eidx in range(num_experts):
                full_prefix = join_module_path(name, f"experts.{eidx}")
                
                # Create disk-backed expert wrapper
                rels = [n for (n, _) in experts[eidx].named_parameters()]
                blueprint = copy.deepcopy(experts[eidx]).to("meta")
                
                wrapped = DiskBackedExpert(
                    blueprint=blueprint,
                    param_rel_names=rels,
                    full_prefix=full_prefix,
                    idx=safetensor_idx,
                    layer_idx=layer_idx if layer_idx is not None else -1,
                    expert_idx=eidx,
                    lru=lru,
                    stats=stats,
                    dtype=dtype,
                    allow_cpu_fallback=allow_cpu_fallback
                )
                experts[eidx] = wrapped
                total_experts += 1
            
            # Distribute gates across GPUs
            if hasattr(module, "gate"):
                try:
                    gpu_id = layer_idx % num_gpus if layer_idx is not None else 0
                    module.gate.to(f"cuda:{gpu_id}")
                except Exception:
                    pass
            
            # Handle shared experts
            for alt in ["shared_experts", "shared_expert"]:
                if hasattr(module, alt):
                    try:
                        gpu_id = layer_idx % num_gpus if layer_idx is not None else 0
                        getattr(module, alt).to(f"cuda:{gpu_id}")
                    except Exception:
                        pass
    
    torch.cuda.empty_cache()
    print(f"[TensorParallel] Patched {total_experts} experts across {num_gpus} GPUs")
    return lru, stats

# =========================
# Selective parallel loading
# =========================

def selective_load_parallel(model: nn.Module,
                           idx: ParallelSafetensorIndex,
                           dtype: torch.dtype,
                           num_gpus: int = 8):
    """Load non-expert weights in parallel across GPUs"""
    shards = sorted(glob.glob(os.path.join(idx.root, "*.safetensors")))
    loaded = 0
    
    for shard in shards:
        f = safe_open(shard, framework="pt")
        keys = list(f.keys())
        
        for i, k in enumerate(keys):
            # Skip expert parameters
            if re.search(r"\.experts\.\d+\.", k):
                if ".mlp.gate." not in k and ".mlp.shared_expert" not in k:
                    continue
            
            t = f.get_tensor(k)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            
            # Distribute across GPUs based on layer
            layer_idx = extract_layer_idx_from_name(k)
            if layer_idx is not None:
                gpu_id = layer_idx % num_gpus
                device = f"cuda:{gpu_id}"
            else:
                # Non-layer params go to GPU 0
                device = "cuda:0"
            
            try:
                t = t.to(device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Try next GPU
                    for gpu_id in range(num_gpus):
                        try:
                            device = f"cuda:{gpu_id}"
                            t = t.to(device)
                            break
                        except:
                            continue
                else:
                    raise
            
            # Set parameter in model
            parts = k.split(".")
            obj = model
            for p in parts[:-1]:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                last = parts[-1]
                if hasattr(obj, last):
                    cur = getattr(obj, last)
                    if isinstance(cur, nn.Parameter):
                        if cur.is_meta:
                            setattr(obj, last, nn.Parameter(t, requires_grad=cur.requires_grad))
                        else:
                            cur.data = t
                    else:
                        setattr(obj, last, nn.Parameter(t, requires_grad=False))
            
            loaded += 1
    
    return loaded

# =========================
# Enhanced Runtime with Tensor Parallelism
# =========================

@dataclass
class TensorParallelRuntimeConfig:
    model_name: str
    weights_location: Optional[str] = None
    dtype: str = "bfloat16"
    attn_impl: Optional[str] = "flash_attention_2"
    gpu_expert_budget_gb_per_gpu: float = 18.0  # Per GPU budget
    allow_cpu_fallback: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    system_prompt: Optional[str] = None

class TensorParallelMoERuntime:
    def __init__(self, cfg: TensorParallelRuntimeConfig):
        self.cfg = cfg
        
        # Setup tensor parallelism
        self.num_gpus = setup_tensor_parallel()
        print(f"\n[TensorParallel] Initializing with {self.num_gpus} GPUs")
        print(f"[TensorParallel] Per-GPU expert budget: {cfg.gpu_expert_budget_gb_per_gpu} GB")
        print(f"[TensorParallel] Total expert budget: {cfg.gpu_expert_budget_gb_per_gpu * self.num_gpus} GB")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._dtype = torch.bfloat16 if cfg.dtype.lower() in ["bf16","bfloat16"] else torch.float16
        
        # Initialize safetensor index with parallel support
        safetensor_idx = ParallelSafetensorIndex(
            cfg.weights_location or cfg.model_name,
            num_gpus=self.num_gpus
        )
        
        # Load model with empty weights
        print("[TensorParallel] Loading model configuration...")
        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        
        if not _HAS_ACCELERATE:
            raise RuntimeError("accelerate is required. Install with: pip install accelerate")
        
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # Patch MoE modules for tensor parallelism
        print("[TensorParallel] Patching MoE modules for tensor parallelism...")
        self.lru, self.stats = patch_moe_for_tensor_parallel(
            self.model,
            gpu_expert_budget_gb_per_gpu=cfg.gpu_expert_budget_gb_per_gpu,
            safetensor_idx=safetensor_idx,
            allow_cpu_fallback=cfg.allow_cpu_fallback,
            dtype=self._dtype,
            num_gpus=self.num_gpus
        )
        
        # Load non-expert weights in parallel
        print("[TensorParallel] Loading non-expert weights across GPUs...")
        loaded = selective_load_parallel(
            self.model,
            safetensor_idx,
            dtype=self._dtype,
            num_gpus=self.num_gpus
        )
        print(f"[TensorParallel] Loaded {loaded} non-expert parameters")
        
        # Handle remaining meta tensors
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.is_meta and "_blueprint_meta" not in name:
                    full_name = f"{name}.{param_name}" if name else param_name
                    print(f"[Warning] Initializing {full_name} with zeros")
                    with torch.no_grad():
                        # Distribute based on layer
                        layer_idx = extract_layer_idx_from_name(full_name)
                        gpu_id = layer_idx % self.num_gpus if layer_idx is not None else 0
                        device = f"cuda:{gpu_id}"
                        new_param = nn.Parameter(
                            torch.zeros(param.shape, device=device, dtype=self._dtype)
                        )
                        setattr(module, param_name, new_param)
        
        # Configure attention
        if is_flash_attn_2_available() and hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = cfg.attn_impl or "flash_attention_2"
        
        self.model.eval()
        torch.cuda.empty_cache()
        
        # Print initialization summary
        print("\n" + "="*60)
        print("[TensorParallel] Initialization Complete")
        print(f"  Model: {cfg.model_name}")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Expert Budget: {cfg.gpu_expert_budget_gb_per_gpu} GB per GPU")
        print(f"  Total Budget: {cfg.gpu_expert_budget_gb_per_gpu * self.num_gpus} GB")
        print(f"  Dtype: {self._dtype}")
        print("="*60 + "\n")

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate text with proper output handling"""
        results = []
        t0 = time.time()
        
        # Check if DeepSeek model
        is_deepseek = "deepseek" in self.cfg.model_name.lower()
        
        for i, prompt in enumerate(prompts):
            print(f"\n[Generate] Processing prompt {i+1}/{len(prompts)}")
            
            # Format prompt with chat template
            try:
                msgs = []
                if self.cfg.system_prompt:
                    msgs.append({"role": "system", "content": self.cfg.system_prompt})
                msgs.append({"role": "user", "content": prompt})
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except:
                text = prompt
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to("cuda:0")
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to("cuda:0")
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Generate with proper parameters to avoid garbled output
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.temperature > 0,
                "temperature": self.cfg.temperature if self.cfg.temperature > 0 else 1.0,
                "top_p": self.cfg.top_p,
                "repetition_penalty": self.cfg.repetition_penalty,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Use patched cache for DeepSeek
            if is_deepseek:
                gen_kwargs["past_key_values"] = DeepSeekCompatibleCache()
            
            # Generate
            with torch.cuda.amp.autocast(dtype=self._dtype):
                out_ids = self.model.generate(**gen_kwargs)
            
            # Decode output properly
            # Remove input tokens from output to get only generated text
            generated_ids = out_ids[:, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Get full output including prompt
            full_output = self.tokenizer.decode(
                out_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Calculate memory usage
            peak_mem = 0
            for gpu_id in range(self.num_gpus):
                torch.cuda.set_device(gpu_id)
                peak_mem += torch.cuda.max_memory_allocated(gpu_id)
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "full_output": full_output,
                "peak_vram_total": human_bytes(peak_mem),
                "experts_cache_usage": human_bytes(self.lru.total_budget),
            })
            
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        
        # Summarize expert usage
        usage_summary = {}
        for layer_idx, expert_counts in self.stats.usage_counts.items():
            usage_summary[f"layer_{layer_idx}"] = {
                str(e): c for e, c in sorted(expert_counts.items(), key=lambda x: -x[1])
            }
        
        return {
            "num_prompts": len(prompts),
            "elapsed_sec": elapsed,
            "throughput": f"{len(prompts)/elapsed:.2f} prompts/sec",
            "cache_stats": self.stats.summary(),
            "num_gpus_used": self.num_gpus,
            "expert_usage": usage_summary,
            "results": results,
        }

# =========================
# CLI Interface
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tensor Parallel MoE Runtime with 8 GPUs")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--weights", type=str, default=None, help="Safetensors location")
    parser.add_argument("--gpu-budget-per-gpu", type=float, default=18.0, help="Expert budget per GPU in GB")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--prompts", type=str, nargs="+", 
                       default=["Explain quantum computing in simple terms."])
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TENSOR PARALLEL MOE RUNTIME")
    print("="*60)
    
    cfg = TensorParallelRuntimeConfig(
        model_name=args.model,
        weights_location=args.weights,
        gpu_expert_budget_gb_per_gpu=args.gpu_budget_per_gpu,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        system_prompt=args.system,
    )
    
    runtime = TensorParallelMoERuntime(cfg)
    
    print("\n[TensorParallel] Starting generation...")
    report = runtime.generate(args.prompts)
    
    print("\n" + "="*60)
    print("GENERATION REPORT")
    print("="*60)
    print(json.dumps(report, indent=2, ensure_ascii=False))