# moe_on_demand_runtime.py
# On-demand MoE expert execution with:
#  - Disk-backed (safetensors) experts, optional CPU fallback
#  - GPU LRU cache for hot experts within a VRAM budget
#  - Selective non-expert weight loading to bring up huge MoE on single GPU
#  - Hot-set export + pinning for domain-specific serving
#
# Tested with: DeepSeek-V3/V3.1 (HF), Qwen MoE, Mixtral-style HF MoE modules.

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

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import is_flash_attn_2_available
from transformers.cache_utils import DynamicCache

# Accelerate is used to init empty weights for huge models like DeepSeek
try:
    from accelerate import init_empty_weights
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False

# Safetensors for on-demand tensor loading
from safetensors import safe_open

# =========================
# Cache patch for DeepSeek-V3.1
# =========================

class DeepSeekCompatibleCache(DynamicCache):
    """
    A patched DynamicCache that adds the seen_tokens attribute
    required by DeepSeek-V3.1's prepare_inputs_for_generation method.
    """
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0
        self._max_cache_len = None  # No max length by default
    
    @property
    def seen_tokens(self):
        return self._seen_tokens
    
    def get_max_length(self):
        """Return the maximum cache length (None means no limit)."""
        return self._max_cache_len
    
    def get_usable_length(self, seq_length=None, layer_idx=None):
        """Return the length of the cache that can be used."""
        # For DeepSeek-V3.1, this returns how much of the cache can be reused
        # This should return 0 on first generation (no cache yet) and the cached length on subsequent generations
        
        # During first forward pass, cache might not exist yet for this layer
        if layer_idx is None:
            # Use the general sequence length
            return self.get_seq_length()
            
        # Check if we have cache for this specific layer
        if len(self.key_cache) <= layer_idx:
            return 0
            
        # If the cache exists but is None for this layer (not yet populated)
        if self.key_cache[layer_idx] is None:
            return 0
            
        # Return the cached sequence length for this layer
        # This is the length that was cached from previous forward passes
        return self.key_cache[layer_idx].shape[-2]
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Update the cache and track seen tokens."""
        # Call parent update first to ensure cache is properly initialized
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # Update seen tokens tracking after cache is updated
        if layer_idx == 0:
            # The parent update should have populated the cache
            # Use the actual cached tensor to calculate seen_tokens
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
    # conservative default
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
# Safetensor shard index
# =========================

class SafetensorIndex:
    """
    Resolves which .safetensors shard a given parameter name belongs to,
    and loads only the requested tensors via memory-mapped I/O.

    Usage:
      idx = SafetensorIndex(model_dir_or_repo)
      t = idx.get_tensor("model.layers.10.mlp.experts.42.up_proj.weight", device="cuda", dtype=torch.bfloat16)
    """
    def __init__(self, model_location: str):
        """
        model_location: local dir containing safetensors (recommended).
                        If a repo id is given, this will still try to find a local snapshot dir.
        """
        self.root = self._resolve_local_dir(model_location)
        if self.root is None or not os.path.isdir(self.root):
            raise RuntimeError(
                f"[SafetensorIndex] Could not resolve a local directory for '{model_location}'. "
                "Pass a path that contains *.safetensors files (or use HF snapshot_download before)."
            )
        # Prefer using index.json if present
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
            raise RuntimeError(f"[SafetensorIndex] No *.safetensors found under {self.root}")

        if self.index_path:
            with open(self.index_path, "r") as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", {})
            for k, fname in weight_map.items():
                fp = os.path.join(self.root, fname)
                self._key_to_file[k] = fp
                self._file_to_keys.setdefault(fp, set()).add(k)
        else:
            # Build lazily: probe shards on demand
            pass

        # Cache open file handles to avoid re-parsing headers
        self._open_files: Dict[str, Any] = {}

    def _resolve_local_dir(self, model_location: str) -> Optional[str]:
        if os.path.isdir(model_location):
            return model_location
        # Try resolving a cached snapshot dir via huggingface_hub (if installed)
        try:
            from huggingface_hub import snapshot_download
            snap_dir = snapshot_download(repo_id=model_location, local_files_only=False, allow_patterns="*.safetensors*")
            return snap_dir
        except Exception:
            return None

    def _ensure_file_mapping_for_key(self, key: str) -> Optional[str]:
        if key in self._key_to_file:
            return self._key_to_file[key]
        # Otherwise, find which shard contains this key
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
            raise KeyError(f"[SafetensorIndex] Tensor key not found in shards: {key}")
        f = self._open_files.get(shard)
        if f is None:
            f = safe_open(shard, framework="pt")
            self._open_files[shard] = f
        t = f.get_tensor(key)  # CPU tensor
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t.to(device)

# =========================
# Telemetry
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
# GPU LRU with pinning
# =========================

class ExpertLRU:
    def __init__(self, budget_bytes: int, stats: CacheStats):
        self.budget_bytes = int(budget_bytes)
        self.stats = stats
        self.stats.gpu_budget_bytes = self.budget_bytes
        self._lru: "OrderedDict[Tuple[int,int], 'BaseExpertWrapper']" = OrderedDict()
        self._pinned: set = set()
        self._bytes = 0

    @property
    def used_bytes(self) -> int:
        return self._bytes

    def _evict_until_fits(self, add_bytes: int):
        while self._bytes + add_bytes > self.budget_bytes and self._lru:
            # pop LRU that is NOT pinned
            victim_key = None
            for k in self._lru.keys():
                if k not in self._pinned:
                    victim_key = k
                    break
            if victim_key is None:
                # Nothing evictable
                break
            _, wrapper = self._lru.popitem(last=False) if next(iter(self._lru.keys())) == victim_key else (victim_key, self._lru.pop(victim_key))
            freed = wrapper.param_bytes()
            wrapper._unload_gpu()
            self._bytes -= freed
            self.stats.gpu_evictions += 1

    def touch(self, key: Tuple[int,int]):
        if key in self._lru:
            self._lru.move_to_end(key, last=True)

    def ensure_on_gpu(self, key: Tuple[int,int], wrapper: "BaseExpertWrapper") -> bool:
        if self.budget_bytes <= 0:
            return False
        if key in self._lru and wrapper._has_gpu():
            self._lru.move_to_end(key, last=True)
            self.stats.gpu_hits += 1
            return True
        # load
        need = wrapper.param_bytes()
        self._evict_until_fits(need)
        if need > self.budget_bytes:
            return False
        wrapper._load_gpu()
        self._lru[key] = wrapper
        self._lru.move_to_end(key, last=True)
        self._bytes += need
        self.stats.gpu_loads += 1
        return True

    def pin(self, key: Tuple[int,int], wrapper: "BaseExpertWrapper"):
        need = wrapper.param_bytes()
        self._evict_until_fits(need)
        if self._bytes + need > self.budget_bytes:
            raise RuntimeError(f"[ExpertLRU] Not enough VRAM to pin expert {key}. Need {human_bytes(need)}, available {human_bytes(self.budget_bytes - self._bytes)}.")
        if not wrapper._has_gpu():
            wrapper._load_gpu()
            self._lru[key] = wrapper
            self._lru.move_to_end(key, last=True)
            self._bytes += need
            self.stats.gpu_loads += 1
        self._pinned.add(key)

# =========================
# Expert wrappers (base / CPU-backed / disk-backed)
# =========================

class BaseExpertWrapper(nn.Module):
    """Common interface for expert wrappers."""
    def __init__(self, layer_idx: int, expert_idx: int, lru: ExpertLRU, stats: CacheStats):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self._lru = lru
        self._stats = stats

    def _has_gpu(self) -> bool: ...
    def _load_gpu(self): ...
    def _unload_gpu(self): ...
    def param_bytes(self) -> int: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class CPUBackedExpert(BaseExpertWrapper):
    """
    Keeps a resident CPU copy of the expert; optional GPU cache (LRU) speeds up hot experts.
    """
    def __init__(self, expert_module: nn.Module, layer_idx: int, expert_idx: int,
                 lru: ExpertLRU, stats: CacheStats, prefer_gpu: bool):
        super().__init__(layer_idx, expert_idx, lru, stats)
        self._cpu_expert = expert_module.to("cpu")
        for p in self._cpu_expert.parameters(): p.requires_grad_(False)
        self._gpu_expert: Optional[nn.Module] = None
        self._dtype = next(self._cpu_expert.parameters()).dtype
        self._param_bytes = sum(p.numel() * p.element_size() for p in self._cpu_expert.parameters())
        self._prefer_gpu = prefer_gpu

    def _has_gpu(self) -> bool:
        return self._gpu_expert is not None

    def _load_gpu(self):
        if self._gpu_expert is None:
            self._gpu_expert = copy.deepcopy(self._cpu_expert).to("cuda", dtype=self._dtype)
            for p in self._gpu_expert.parameters(): p.requires_grad_(False)

    def _unload_gpu(self):
        if self._gpu_expert is not None:
            try: del self._gpu_expert
            finally: self._gpu_expert = None

    def param_bytes(self) -> int:
        return self._param_bytes

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._stats.usage_counts[self.layer_idx][self.expert_idx] += 1
        key = (self.layer_idx, self.expert_idx)
        run_gpu = self._prefer_gpu and self._lru.ensure_on_gpu(key, self)
        if run_gpu and self._gpu_expert is not None and x.is_cuda:
            return self._gpu_expert(x)
        self._stats.cpu_runs += 1
        y = self._cpu_expert(x.to("cpu", non_blocking=True))
        return y.to(x.device, non_blocking=True)

class DiskBackedExpert(BaseExpertWrapper):
    """
    No resident CPU copy. Blueprint is on 'meta' (no storage).
    Weights are loaded directly from .safetensors shards into GPU (cache) or CPU if fallback allowed.
    """
    def __init__(self,
                 blueprint: nn.Module,
                 param_rel_names: List[str],
                 full_prefix: str,  # e.g., "model.layers.10.mlp.experts.42"
                 idx: SafetensorIndex,
                 layer_idx: int,
                 expert_idx: int,
                 lru: ExpertLRU,
                 stats: CacheStats,
                 dtype: torch.dtype,
                 allow_cpu_fallback: bool = False):
        super().__init__(layer_idx, expert_idx, lru, stats)
        self._blueprint_meta = blueprint  # on meta
        self._param_rel_names = param_rel_names
        self._full_prefix = full_prefix
        self._idx = idx
        self._dtype = dtype
        # compute bytes using blueprint shapes (meta has shapes)
        pb = 0
        for n, p in blueprint.named_parameters():
            pb += p.numel() * elem_bytes(dtype)
        self._param_bytes = pb
        self._gpu_expert: Optional[nn.Module] = None
        self._allow_cpu_fallback = allow_cpu_fallback

    def _has_gpu(self) -> bool:
        return self._gpu_expert is not None

    def _materialize_and_load(self, device: str) -> nn.Module:
        # Clone structure from blueprint (meta) and load tensors from shards
        # First create empty module on target device
        mod = copy.deepcopy(self._blueprint_meta)
        # Move to target device using to_empty for meta tensors
        mod = mod.to_empty(device=device).to(dtype=self._dtype)
        
        for rel in self._param_rel_names:
            key = join_module_path(self._full_prefix, rel)
            t = self._idx.get_tensor(key, device=device, dtype=self._dtype)
            # assign into module
            # rel like "up_proj.weight" or "gate_proj.weight"
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
        # freeze
        for p in mod.parameters(): p.requires_grad_(False)
        return mod

    def _load_gpu(self):
        if self._gpu_expert is None:
            self._gpu_expert = self._materialize_and_load("cuda")

    def _unload_gpu(self):
        if self._gpu_expert is not None:
            try: del self._gpu_expert
            finally: self._gpu_expert = None

    def param_bytes(self) -> int:
        return self._param_bytes

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._stats.usage_counts[self.layer_idx][self.expert_idx] += 1
        key = (self.layer_idx, self.expert_idx)

        # Try to run using GPU cache
        if self._lru.ensure_on_gpu(key, self) and self._gpu_expert is not None and x.is_cuda:
            return self._gpu_expert(x)

        # If we reach here, either budget=0 or cache miss not admitted.
        # Avoid CPU if not allowed: do a temporary GPU load-run-evict (no LRU insert).
        if not self._allow_cpu_fallback:
            tmp = self._materialize_and_load("cuda")
            y = tmp(x)
            del tmp
            torch.cuda.empty_cache()
            self._stats.temp_gpu_runs += 1
            return y

        # CPU fallback path (uses CPU RAM only for this forward)
        self._stats.cpu_runs += 1
        tmp_cpu = self._materialize_and_load("cpu")
        y = tmp_cpu(x.to("cpu", non_blocking=True))
        del tmp_cpu
        return y.to(x.device, non_blocking=True)

# =========================
# Model patcher & selective loader
# =========================

@dataclass
class PatcherReport:
    moe_layers: List[Tuple[str,int,int]] = field(default_factory=list)  # (name, layer_idx, num_experts)
    total_experts_wrapped: int = 0
    gate_modules_to_cuda: int = 0
    shared_experts_to_cuda: int = 0
    per_expert_bytes_bf16: Optional[int] = None

def _list_param_rel_names(expert_module: nn.Module) -> List[str]:
    return [n for (n, _) in expert_module.named_parameters()]

def _make_meta_blueprint(expert_module: nn.Module) -> nn.Module:
    bp = copy.deepcopy(expert_module).to("meta")
    return bp

def patch_moe_modules_for_on_demand(model: nn.Module,
                                    gpu_expert_budget_gb: float,
                                    prefer_gpu_when_possible: bool,
                                    use_disk_backed: bool,
                                    safetensor_idx: Optional[SafetensorIndex],
                                    allow_cpu_fallback: bool,
                                    dtype: torch.dtype) -> Tuple[ExpertLRU, CacheStats, PatcherReport]:
    """
    Replace each expert with CPUBackedExpert (if use_disk_backed=False) or DiskBackedExpert otherwise.
    """
    stats = CacheStats()
    lru = ExpertLRU(int(gpu_expert_budget_gb * (1024**3)), stats)
    report = PatcherReport()

    # Discover MoE modules generically: modules with .experts: ModuleList
    for name, module in model.named_modules():
        if hasattr(module, "experts") and isinstance(getattr(module, "experts"), nn.ModuleList):
            experts: nn.ModuleList = module.experts
            layer_idx = extract_layer_idx_from_name(name)
            num_experts = len(experts)
            # Capture per-expert size (bf16) from the first example
            if report.per_expert_bytes_bf16 is None and num_experts > 0:
                first = experts[0]
                per_b = 0
                for _, p in first.named_parameters():
                    per_b += p.numel() * elem_bytes(torch.bfloat16)
                report.per_expert_bytes_bf16 = per_b

            for eidx in range(num_experts):
                full_prefix = join_module_path(name, f"experts.{eidx}")
                if use_disk_backed:
                    assert safetensor_idx is not None, "Disk-backed mode requires SafetensorIndex"
                    rels = _list_param_rel_names(experts[eidx])
                    blueprint = _make_meta_blueprint(experts[eidx])
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
                else:
                    wrapped = CPUBackedExpert(
                        expert_module=experts[eidx],
                        layer_idx=layer_idx if layer_idx is not None else -1,
                        expert_idx=eidx,
                        lru=lru,
                        stats=stats,
                        prefer_gpu=prefer_gpu_when_possible
                    )
                experts[eidx] = wrapped

            # Keep gate/shared experts on CUDA (tiny vs experts)
            if hasattr(module, "gate"):
                try:
                    module.gate.to("cuda")
                    report.gate_modules_to_cuda += 1
                except Exception:
                    pass
            for alt in ["shared_experts", "shared_expert"]:
                if hasattr(module, alt):
                    try:
                        getattr(module, alt).to("cuda")
                        report.shared_experts_to_cuda += 1
                    except Exception:
                        pass

            report.moe_layers.append((name, layer_idx if layer_idx is not None else -1, num_experts))
            report.total_experts_wrapped += num_experts

    # Free VRAM that belonged to replaced experts
    torch.cuda.empty_cache()
    return lru, stats, report

def set_param_by_fqn(model: nn.Module, fqn: str, tensor: torch.Tensor):
    parts = fqn.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p, None)
        if obj is None:
            return  # Skip if path doesn't exist
    last = parts[-1]
    if not hasattr(obj, last):
        return  # Skip if attribute doesn't exist
    cur = getattr(obj, last)
    if isinstance(cur, nn.Parameter):
        # For meta tensors, we need to replace them entirely
        if cur.is_meta:
            setattr(obj, last, nn.Parameter(tensor, requires_grad=cur.requires_grad))
        else:
            # Ensure tensor matches the parameter's dtype and device
            if cur.dtype != tensor.dtype:
                tensor = tensor.to(cur.dtype)
            if cur.device != tensor.device:
                tensor = tensor.to(cur.device)
            cur.data = tensor
    else:
        setattr(obj, last, nn.Parameter(tensor, requires_grad=False))

def selective_load_non_expert_weights(model: nn.Module,
                                      idx: SafetensorIndex,
                                      device: str,
                                      dtype: torch.dtype):
    """
    Load all non-expert weights (and gate/shared_experts) from safetensors into the (empty) model.
    Skips parameters that match ".experts.{N}.".
    """
    # Enumerate shards and all keys; skip expert param keys
    shards = sorted(glob.glob(os.path.join(idx.root, "*.safetensors")))
    loaded = 0
    for shard in shards:
        f = safe_open(shard, framework="pt")
        keys = list(f.keys())
        for k in keys:
            if re.search(r"\.experts\.\d+\.", k):
                # exceptions: allow gate/shared inside mlp
                if ".mlp.gate." in k or ".mlp.shared_expert" in k or ".mlp.shared_experts" in k:
                    pass
                else:
                    continue
            t = f.get_tensor(k)  # CPU
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            try:
                t = t.to(device)
            except RuntimeError as e:
                if "out of memory" in str(e) or "cudaGetDeviceCount" in str(e):
                    # This error often occurs when CUDA isn't properly initialized
                    # Try CPU fallback instead
                    print(f"Warning: CUDA initialization error for {k}, keeping on CPU")
                    t = t.to("cpu")
                    device = "cpu"
                else:
                    raise
            set_param_by_fqn(model, k, t)
            loaded += 1
    return loaded

# =========================
# Runtime
# =========================

@dataclass
class RuntimeConfig:
    model_name: str
    weights_location: Optional[str] = None     # local dir or repo id to find *.safetensors
    dtype: str = "bfloat16"
    attn_impl: Optional[str] = "flash_attention_2"
    gpu_expert_budget_gb: float = 12.0
    prefer_gpu_when_possible: bool = True
    allow_cpu_fallback_for_disk: bool = False  # usually False for very low-RAM scenarios
    max_new_tokens: int = 256
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    use_4bit_backbone: bool = False
    # Strategy to bring up huge models:
    #   "auto": use empty+selective load if DeepSeek-like, else standard load
    #   "empty_selective": always do empty+selective
    #   "standard": load everything then patch (OK for small MoE like Qwen/Mixtral)
    load_strategy: str = "auto"

class OnDemandMoERuntime:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        
        # Initialize CUDA early to avoid initialization errors later
        if torch.cuda.is_available():
            torch.cuda.init()
            # Don't set a specific device - let PyTorch handle device selection
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self._dtype = torch.bfloat16 if cfg.dtype.lower() in ["bf16","bfloat16"] else torch.float16

        # Detect DeepSeek-V3-ish models
        try:
            model_type_hint = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True).model_type
        except Exception:
            model_type_hint = ""
        is_deepseek = ("deepseek_v3" in (model_type_hint or "")) or ("DeepSeek" in cfg.model_name)

        # Decide strategy
        strat = cfg.load_strategy
        if strat == "auto":
            strat = "empty_selective" if is_deepseek else "standard"

        safetensor_idx: Optional[SafetensorIndex] = None
        if cfg.weights_location is not None:
            safetensor_idx = SafetensorIndex(cfg.weights_location)
        elif strat == "empty_selective":
            # We *need* to know where the shards are
            safetensor_idx = SafetensorIndex(cfg.model_name)

        if strat == "empty_selective":
            if not _HAS_ACCELERATE:
                raise RuntimeError("accelerate is required for empty_selective load. pip install accelerate")
            # Build empty model from config, then load only non-expert weights
            config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            # Patch MoE first, so expert params don't exist in the state dict anymore
            self.lru, self.stats, self.report = patch_moe_modules_for_on_demand(
                self.model,
                gpu_expert_budget_gb=cfg.gpu_expert_budget_gb,
                prefer_gpu_when_possible=cfg.prefer_gpu_when_possible,
                use_disk_backed=True,   # critical for DeepSeek on small GPU
                safetensor_idx=safetensor_idx,
                allow_cpu_fallback=cfg.allow_cpu_fallback_for_disk,
                dtype=self._dtype,
            )
            # Now load backbone/gate/shared to CUDA
            loaded = selective_load_non_expert_weights(self.model, safetensor_idx, device="cuda", dtype=self._dtype)
            
            # Check for any remaining meta tensors and replace them with zeros on CUDA
            # Skip _blueprint_meta parameters as they're expected to be on meta
            for name, module in self.model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    if param.is_meta and "_blueprint_meta" not in name:
                        full_name = f"{name}.{param_name}" if name else param_name
                        print(f"Warning: Parameter {full_name} still on meta device, initializing with zeros")
                        with torch.no_grad():
                            new_param = nn.Parameter(torch.zeros(param.shape, device='cuda', dtype=self._dtype or torch.float16))
                            setattr(module, param_name, new_param)
            
            # Optional: FA2
            if is_flash_attn_2_available() and hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = self.cfg.attn_impl or "flash_attention_2"
            
            # Set model to eval mode
            self.model.eval()

        elif strat == "standard":
            # Load full model (OK for small MoE checkpoints)
            quantization_config = None
            if cfg.use_4bit_backbone:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                    )
                except Exception as e:
                    print("BitsAndBytes not available; continuing without 4-bit:", e)
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=self._dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map={"": "cuda"},
                quantization_config=quantization_config,
            ).eval()
            if is_flash_attn_2_available() and hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = self.cfg.attn_impl or "flash_attention_2"

            # Patch experts; prefer disk-backed if we have an index available
            use_disk = safetensor_idx is not None
            self.lru, self.stats, self.report = patch_moe_modules_for_on_demand(
                self.model,
                gpu_expert_budget_gb=cfg.gpu_expert_budget_gb,
                prefer_gpu_when_possible=cfg.prefer_gpu_when_possible,
                use_disk_backed=use_disk,
                safetensor_idx=safetensor_idx,
                allow_cpu_fallback=cfg.allow_cpu_fallback_for_disk,
                dtype=self._dtype,
            )
        else:
            raise ValueError(f"Unknown load_strategy {cfg.load_strategy}")

        torch.cuda.empty_cache()

        print("\n[OnDemandMoE] Patch report:")
        print(json.dumps({
            "moe_layers": len(self.report.moe_layers),
            "experts_wrapped": self.report.total_experts_wrapped,
            "gate_to_cuda": self.report.gate_modules_to_cuda,
            "shared_to_cuda": self.report.shared_experts_to_cuda,
            "per_expert_bytes_bf16_est": human_bytes(self.report.per_expert_bytes_bf16 or 0),
            "gpu_expert_budget": human_bytes(self.lru.budget_bytes),
        }, indent=2))

    @torch.inference_mode()
    def _fmt_prompt(self, p: str) -> str:
        try:
            msgs = []
            if self.cfg.system_prompt:
                msgs.append({"role": "system", "content": self.cfg.system_prompt})
            msgs.append({"role": "user", "content": p})
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return p

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> Dict[str, Any]:
        results = []
        t0 = time.time()
        
        # Check if this is a DeepSeek model that needs our patched cache
        is_deepseek = "deepseek" in self.cfg.model_name.lower()
        
        for i, p in enumerate(prompts):
            text = self._fmt_prompt(p)
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
            
            # Use patched cache for DeepSeek models
            if is_deepseek:
                past_key_values = DeepSeekCompatibleCache()
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=(self.cfg.temperature > 0),
                    temperature=self.cfg.temperature,
                    use_cache=True,
                    past_key_values=past_key_values,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                )
            else:
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=(self.cfg.temperature > 0),
                    temperature=self.cfg.temperature,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                )
            out = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            peak = torch.cuda.max_memory_allocated()
            results.append({
                "prompt": p,
                "output": out,
                "peak_vram": human_bytes(peak),
                "experts_gpu_cache_bytes": human_bytes(self.lru.used_bytes),
            })
            gc.collect(); torch.cuda.empty_cache()
        elapsed = time.time() - t0
        usage_summary = {
            f"layer_{L}": {str(e): c for e, c in sorted(d.items(), key=lambda kv: -kv[1])}
            for L, d in self.stats.usage_counts.items()
        }
        return {
            "num_prompts": len(prompts),
            "elapsed_sec": elapsed,
            "throughput_prompts_per_sec": len(prompts)/max(elapsed, 1e-6),
            "cache_stats": self.stats.summary(),
            "gpu_cache_bytes": human_bytes(self.lru.used_bytes),
            "usage_by_layer": usage_summary,
            "outputs": results,
        }

    # ---------------------------
    # HOT-SET: export / estimate / pin
    # ---------------------------

    def export_hotset(self,
                      path: str,
                      coverage: float = 0.95,
                      min_per_layer: Optional[int] = None,
                      max_per_layer: Optional[int] = None) -> Dict[str, Any]:
        """
        Build a per-layer list of 'hot' experts that cover a given fraction of *usage counts*.
        Writes JSON with metadata + VRAM estimate.
        """
        hotset = {}
        per_expert_bytes = self.report.per_expert_bytes_bf16 or 0  # bf16 estimate (safe upper bound)
        total_bytes = 0
        for (layer_name, layer_idx, _) in self.report.moe_layers:
            counts = self.stats.usage_counts.get(layer_idx, {})
            if not counts:
                # no usage -> keep minimum (router top_k if available) or 1
                keep = min_per_layer or 1
                hot = list(range(keep))
                hotset[str(layer_idx)] = hot
                total_bytes += keep * per_expert_bytes
                continue
            items = sorted(counts.items(), key=lambda kv: -kv[1])
            total = sum(c for _, c in items) or 1
            cum = 0
            sel = []
            for e, c in items:
                sel.append(e); cum += c
                if cum / total >= coverage:
                    break
            if min_per_layer is not None and len(sel) < min_per_layer:
                # pad with next-most-used
                extra = [e for e, _ in items if e not in sel][:max(0, min_per_layer - len(sel))]
                sel.extend(extra)
            if max_per_layer is not None and len(sel) > max_per_layer:
                sel = sel[:max_per_layer]
            hotset[str(layer_idx)] = sorted(sel)
            total_bytes += len(sel) * per_expert_bytes

        meta = {
            "model_name": self.cfg.model_name,
            "dtype": str(self._dtype),
            "coverage": coverage,
            "min_per_layer": min_per_layer,
            "max_per_layer": max_per_layer,
            "per_expert_bytes_bf16_est": per_expert_bytes,
            "total_experts_selected": sum(len(v) for v in hotset.values()),
            "est_vram_bf16": total_bytes,
            "est_vram_bf16_human": human_bytes(total_bytes),
            "layers": sorted(hotset.keys(), key=lambda s: int(s)),
            "hotset": hotset,
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[OnDemandMoE] Hotset exported to {path} (≈{human_bytes(total_bytes)} VRAM)")
        return meta

    def estimate_hotset_bytes(self, hotset_json: str) -> Dict[str, Any]:
        data = json.load(open(hotset_json, "r"))
        per_expert_bytes = data.get("per_expert_bytes_bf16_est", self.report.per_expert_bytes_bf16 or 0)
        total = sum(len(v) for v in data["hotset"].values()) * per_expert_bytes
        return {"per_expert_bytes": per_expert_bytes, "total_bytes": total, "total_human": human_bytes(total)}

    def pin_hotset(self, hotset_json: str):
        """
        Load & pin the hot experts to GPU (won't be evicted). If budget is insufficient, raises an error.
        """
        data = json.load(open(hotset_json, "r"))
        hotset = data["hotset"]
        # Walk the model tree and fetch wrappers by (layer, expert)
        count = 0
        for (name, layer_idx, num_experts) in self.report.moe_layers:
            layer_key = str(layer_idx)
            if layer_key not in hotset: 
                continue
            keep_list = hotset[layer_key]
            experts = self._resolve_expert_wrappers(name)
            for eidx in keep_list:
                if eidx < 0 or eidx >= len(experts):
                    continue
                wrapper = experts[eidx]
                key = (layer_idx, eidx)
                self.lru.pin(key, wrapper)
                count += 1
        torch.cuda.synchronize()
        print(f"[OnDemandMoE] Pinned {count} experts from hotset to GPU (cache now {human_bytes(self.lru.used_bytes)})")

    def _resolve_expert_wrappers(self, moe_module_name: str) -> nn.ModuleList:
        # Resolve module by name and return its .experts list (wrappers)
        obj = self.model
        for p in moe_module_name.split("."):
            if p:
                obj = getattr(obj, p)
        return getattr(obj, "experts")

# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-demand MoE expert runtime w/ disk-backed experts and hotset export")
    parser.add_argument("--model", type=str, required=True, help="HF model name or local path")
    parser.add_argument("--weights", type=str, default=None, help="Local dir or repo id holding *.safetensors (required for DeepSeek on single GPU).")
    parser.add_argument("--gpu-expert-budget-gb", type=float, default=12.0)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow CPU fallback when disk-backed cache misses (uses CPU RAM).")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--use-4bit-backbone", action="store_true")
    parser.add_argument("--load-strategy", type=str, default="auto", choices=["auto","empty_selective","standard"])
    parser.add_argument("--prompts", type=str, nargs="+", default=["Explain the Fisher information."])
    parser.add_argument("--export-hotset", type=str, default=None, help="Path to write hotset JSON")
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--min-per-layer", type=int, default=None)
    parser.add_argument("--max-per-layer", type=int, default=None)
    parser.add_argument("--pin-hotset", type=str, default=None, help="Load/pin a previously exported hotset JSON")

    args = parser.parse_args()

    cfg = RuntimeConfig(
        model_name=args.model,
        weights_location=args.weights,
        gpu_expert_budget_gb=args.gpu_expert_budget_gb,
        prefer_gpu_when_possible=args.prefer_gpu,
        allow_cpu_fallback_for_disk=args.allow_cpu_fallback,
        max_new_tokens=args.max_new,
        temperature=args.temp,
        system_prompt=args.system,
        use_4bit_backbone=args.use_4bit_backbone,
        load_strategy=args.load_strategy,
    )

    runtime = OnDemandMoERuntime(cfg)
    report = runtime.generate(args.prompts)
    print("\n[OnDemandMoE] Generation report:")
    print(json.dumps(report, indent=2))

    if args.export_hotset:
        meta = runtime.export_hotset(
            path=args.export_hotset,
            coverage=args.coverage,
            min_per_layer=args.min_per_layer,
            max_per_layer=args.max_per_layer
        )
        print("\n[OnDemandMoE] Hotset meta:")
        print(json.dumps(meta, indent=2))

    if args.pin_hotset:
        runtime.pin_hotset(args.pin_hotset)
        # Optional: run again to observe cache hits and lower latency
