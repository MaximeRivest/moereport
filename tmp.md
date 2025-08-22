Is this code supporting gpt-oss to? Particularly could I use a 3090 nvidia rtx gpu and have a prompt set of 30 domain specific prompts and know what experts are activated at each layer for each token for each prompt in and exact completely accurate manner while loading (and keeping) in gpu only the experts needed as I go through the generation of the assistance token?


<code>
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

# Accelerate is used to init empty weights for huge models like DeepSeek
try:
    from accelerate import init_empty_weights
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False

# Safetensors for on-demand tensor loading
from safetensors import safe_open

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
        mod = copy.deepcopy(self._blueprint_meta).to(device, dtype=self._dtype)
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
        obj = getattr(obj, p)
    last = parts[-1]
    cur = getattr(obj, last)
    if isinstance(cur, nn.Parameter):
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
            t = t.to(device)
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
            # Optional: FA2
            if is_flash_attn_2_available() and hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = self.cfg.attn_impl or "flash_attention_2"

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
        for i, p in enumerate(prompts):
            text = self._fmt_prompt(p)
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
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
</code>

<gpt_oss_info>
---
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
tags:
- vllm
---

<p align="center">
  <img alt="gpt-oss-20b" src="https://raw.githubusercontent.com/openai/gpt-oss/main/docs/gpt-oss-20b.svg">
</p>

<p align="center">
  <a href="https://gpt-oss.com"><strong>Try gpt-oss</strong></a> ·
  <a href="https://cookbook.openai.com/topic/gpt-oss"><strong>Guides</strong></a> ·
  <a href="https://openai.com/index/gpt-oss-model-card"><strong>Model card</strong></a> ·
  <a href="https://openai.com/index/introducing-gpt-oss/"><strong>OpenAI blog</strong></a>
</p>

<br>

Welcome to the gpt-oss series, [OpenAI’s open-weight models](https://openai.com/open-models) designed for powerful reasoning, agentic tasks, and versatile developer use cases.

We’re releasing two flavors of these open models:
- `gpt-oss-120b` — for production, general purpose, high reasoning use cases that fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X) (117B parameters with 5.1B active parameters)
- `gpt-oss-20b` — for lower latency, and local or specialized use cases (21B parameters with 3.6B active parameters)

Both models were trained on our [harmony response format](https://github.com/openai/harmony) and should only be used with the harmony format as it will not work correctly otherwise.


> [!NOTE]
> This model card is dedicated to the smaller `gpt-oss-20b` model. Check out [`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) for the larger model.

# Highlights

* **Permissive Apache 2.0 license:** Build freely without copyleft restrictions or patent risk—ideal for experimentation, customization, and commercial deployment.  
* **Configurable reasoning effort:** Easily adjust the reasoning effort (low, medium, high) based on your specific use case and latency needs.  
* **Full chain-of-thought:** Gain complete access to the model’s reasoning process, facilitating easier debugging and increased trust in outputs. It’s not intended to be shown to end users.  
* **Fine-tunable:** Fully customize models to your specific use case through parameter fine-tuning.
* **Agentic capabilities:** Use the models’ native capabilities for function calling, [web browsing](https://github.com/openai/gpt-oss/tree/main?tab=readme-ov-file#browser), [Python code execution](https://github.com/openai/gpt-oss/tree/main?tab=readme-ov-file#python), and Structured Outputs.
* **MXFP4 quantization:** The models were post-trained with MXFP4 quantization of the MoE weights, making `gpt-oss-120b` run on a single 80GB GPU (like NVIDIA H100 or AMD MI300X) and the `gpt-oss-20b` model run within 16GB of memory. All evals were performed with the same MXFP4 quantization.

---

# Inference examples

## Transformers

You can use `gpt-oss-120b` and `gpt-oss-20b` with Transformers. If you use the Transformers chat template, it will automatically apply the [harmony response format](https://github.com/openai/harmony). If you use `model.generate` directly, you need to apply the harmony format manually using the chat template or use our [openai-harmony](https://github.com/openai/harmony) package.

To get started, install the necessary dependencies to setup your environment:

```
pip install -U transformers kernels torch 
```

Once, setup you can proceed to run the model by running the snippet below:

```py
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

Alternatively, you can run the model via [`Transformers Serve`](https://huggingface.co/docs/transformers/main/serving) to spin up a OpenAI-compatible webserver:

```
transformers serve
transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-20b
```

[Learn more about how to use gpt-oss with Transformers.](https://cookbook.openai.com/articles/gpt-oss/run-transformers)

## vLLM

vLLM recommends using [uv](https://docs.astral.sh/uv/) for Python dependency management. You can use vLLM to spin up an OpenAI-compatible webserver. The following command will automatically download the model and start the server.

```bash
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b
```

[Learn more about how to use gpt-oss with vLLM.](https://cookbook.openai.com/articles/gpt-oss/run-vllm)

## PyTorch / Triton

To learn about how to use this model with PyTorch and Triton, check out our [reference implementations in the gpt-oss repository](https://github.com/openai/gpt-oss?tab=readme-ov-file#reference-pytorch-implementation).

## Ollama

If you are trying to run gpt-oss on consumer hardware, you can use Ollama by running the following commands after [installing Ollama](https://ollama.com/download).

```bash
# gpt-oss-20b
ollama pull gpt-oss:20b
ollama run gpt-oss:20b
```

[Learn more about how to use gpt-oss with Ollama.](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)

#### LM Studio

If you are using [LM Studio](https://lmstudio.ai/) you can use the following commands to download.

```bash
# gpt-oss-20b
lms get openai/gpt-oss-20b
```

Check out our [awesome list](https://github.com/openai/gpt-oss/blob/main/awesome-gpt-oss.md) for a broader collection of gpt-oss resources and inference partners.

---

# Download the model

You can download the model weights from the [Hugging Face Hub](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) directly from Hugging Face CLI:

```shell
# gpt-oss-20b
huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
pip install gpt-oss
python -m gpt_oss.chat model/
```

# Reasoning levels

You can adjust the reasoning level that suits your task across three levels:

* **Low:** Fast responses for general dialogue.  
* **Medium:** Balanced speed and detail.  
* **High:** Deep and detailed analysis.

The reasoning level can be set in the system prompts, e.g., "Reasoning: high".

# Tool use

The gpt-oss models are excellent for:
* Web browsing (using built-in browsing tools)
* Function calling with defined schemas
* Agentic operations like browser tasks

# Fine-tuning

Both gpt-oss models can be fine-tuned for a variety of specialized use cases.

This smaller model `gpt-oss-20b` can be fine-tuned on consumer hardware, whereas the larger [`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) can be fine-tuned on a single H100 node.



{#-
  In addition to the normal inputs of `messages` and `tools`, this template also accepts the
  following kwargs:
  - "builtin_tools": A list, can contain "browser" and/or "python".
  - "model_identity": A string that optionally describes the model identity.
  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".
 #}

{#- Tool Definition Rendering ============================================== #}
{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}
    {%- if param_spec.type == "array" -%}
        {%- if param_spec['items'] -%}
            {%- if param_spec['items']['type'] == "string" -%}
                {{- "string[]" }}
            {%- elif param_spec['items']['type'] == "number" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "integer" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "boolean" -%}
                {{- "boolean[]" }}
            {%- else -%}
                {%- set inner_type = render_typescript_type(param_spec['items'], required_params) -%}
                {%- if inner_type == "object | object" or inner_type|length > 50 -%}
                    {{- "any[]" }}
                {%- else -%}
                    {{- inner_type + "[]" }}
                {%- endif -%}
            {%- endif -%}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- else -%}
            {{- "any[]" }}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}
        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}
        {%- if param_spec.type | length > 1 -%}
            {{- param_spec.type | join(" | ") }}
        {%- else -%}
            {{- param_spec.type[0] }}
        {%- endif -%}
    {%- elif param_spec.oneOf -%}
        {#- Handle oneOf schemas - check for complex unions and fallback to any #}
        {%- set has_object_variants = false -%}
        {%- for variant in param_spec.oneOf -%}
            {%- if variant.type == "object" -%}
                {%- set has_object_variants = true -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}
            {{- "any" }}
        {%- else -%}
            {%- for variant in param_spec.oneOf -%}
                {{- render_typescript_type(variant, required_params) -}}
                {%- if variant.description %}
                    {{- "// " + variant.description }}
                {%- endif -%}
                {%- if variant.default is defined %}
                    {{ "// default: " + variant.default|tojson }}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- " | " }}
                {% endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif param_spec.type == "string" -%}
        {%- if param_spec.enum -%}
            {{- '"' + param_spec.enum|join('" | "') + '"' -}}
        {%- else -%}
            {{- "string" }}
            {%- if param_spec.nullable %}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type == "number" -%}
        {{- "number" }}
    {%- elif param_spec.type == "integer" -%}
        {{- "number" }}
    {%- elif param_spec.type == "boolean" -%}
        {{- "boolean" }}

    {%- elif param_spec.type == "object" -%}
        {%- if param_spec.properties -%}
            {{- "{\n" }}
            {%- for prop_name, prop_spec in param_spec.properties.items() -%}
                {{- prop_name -}}
                {%- if prop_name not in (param_spec.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{ render_typescript_type(prop_spec, param_spec.required or []) }}
                {%- if not loop.last -%}
                    {{-", " }}
                {%- endif -%}
            {%- endfor -%}
            {{- "}" }}
        {%- else -%}
            {{- "object" }}
        {%- endif -%}
    {%- else -%}
        {{- "any" }}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_namespace(namespace_name, tools) -%}
    {{- "## " + namespace_name + "\n\n" }}
    {{- "namespace " + namespace_name + " {\n\n" }}
    {%- for tool in tools %}
        {%- set tool = tool.function %}
        {{- "// " + tool.description + "\n" }}
        {{- "type "+ tool.name + " = " }}
        {%- if tool.parameters and tool.parameters.properties %}
            {{- "(_: {\n" }}
            {%- for param_name, param_spec in tool.parameters.properties.items() %}
                {%- if param_spec.description %}
                    {{- "// " + param_spec.description + "\n" }}
                {%- endif %}
                {{- param_name }}
                {%- if param_name not in (tool.parameters.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}
                {%- if param_spec.default is defined -%}
                    {%- if param_spec.enum %}
                        {{- ", // default: " + param_spec.default }}
                    {%- elif param_spec.oneOf %}
                        {{- "// default: " + param_spec.default }}
                    {%- else %}
                        {{- ", // default: " + param_spec.default|tojson }}
                    {%- endif -%}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- ",\n" }}
                {%- else %}
                    {{- ",\n" }}
                {%- endif -%}
            {%- endfor %}
            {{- "}) => any;\n\n" }}
        {%- else -%}
            {{- "() => any;\n\n" }}
        {%- endif -%}
    {%- endfor %}
    {{- "} // namespace " + namespace_name }}
{%- endmacro -%}

{%- macro render_builtin_tools(browser_tool, python_tool) -%}
    {%- if browser_tool %}
        {{- "## browser\n\n" }}
        {{- "// Tool for browsing.\n" }}
        {{- "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\n" }}
        {{- "// Cite information from the tool using the following format:\n" }}
        {{- "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\n" }}
        {{- "// Do not quote more than 10 words directly from the tool output.\n" }}
        {{- "// sources=web (default: web)\n" }}
        {{- "namespace browser {\n\n" }}
        {{- "// Searches for information related to `query` and displays `topn` results.\n" }}
        {{- "type search = (_: {\n" }}
        {{- "query: string,\n" }}
        {{- "topn?: number, // default: 10\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\n" }}
        {{- "// Valid link ids are displayed with the formatting: `【{id}†.*】`.\n" }}
        {{- "// If `cursor` is not provided, the most recent page is implied.\n" }}
        {{- "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.\n" }}
        {{- "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\n" }}
        {{- "// Use this function without `id` to scroll to a new location of an opened page.\n" }}
        {{- "type open = (_: {\n" }}
        {{- "id?: number | string, // default: -1\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "loc?: number, // default: -1\n" }}
        {{- "num_lines?: number, // default: -1\n" }}
        {{- "view_source?: boolean, // default: false\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.\n" }}
        {{- "type find = (_: {\n" }}
        {{- "pattern: string,\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "}) => any;\n\n" }}
        {{- "} // namespace browser\n\n" }}
    {%- endif -%}

    {%- if python_tool %}
        {{- "## python\n\n" }}
        {{- "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\n" }}
        {{- "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\n\n" }}
    {%- endif -%}
{%- endmacro -%}

{#- System Message Construction ============================================ #}
{%- macro build_system_message() -%}
    {%- if model_identity is not defined %}
        {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}
    {%- endif %}
    {{- model_identity + "\n" }}
    {{- "Knowledge cutoff: 2024-06\n" }}
    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\n\n" }}
    {%- if reasoning_effort is not defined %}
        {%- set reasoning_effort = "medium" %}
    {%- endif %}
    {{- "Reasoning: " + reasoning_effort + "\n\n" }}
    {%- if builtin_tools %}
        {{- "# Tools\n\n" }}
        {%- set available_builtin_tools = namespace(browser=false, python=false) %}
        {%- for tool in builtin_tools %}
            {%- if tool == "browser" %}
                {%- set available_builtin_tools.browser = true %}
            {%- elif tool == "python" %}
                {%- set available_builtin_tools.python = true %}
            {%- endif %}
        {%- endfor %}
        {{- render_builtin_tools(available_builtin_tools.browser, available_builtin_tools.python) }}
    {%- endif -%}
    {{- "# Valid channels: analysis, commentary, final. Channel must be included for every message." }}
    {%- if tools -%}
        {{- "\nCalls to these tools must go to the commentary channel: 'functions'." }}
    {%- endif -%}
{%- endmacro -%}

{#- Main Template Logic ================================================= #}
{#- Set defaults #}

{#- Render system message #}
{{- "<|start|>system<|message|>" }}
{{- build_system_message() }}
{{- "<|end|>" }}

{#- Extract developer message #}
{%- if messages[0].role == "developer" or messages[0].role == "system" %}
    {%- set developer_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set developer_message = "" %}
    {%- set loop_messages = messages %}
{%- endif %}

{#- Render developer message #}
{%- if developer_message or tools %}
    {{- "<|start|>developer<|message|>" }}
    {%- if developer_message %}
        {{- "# Instructions\n\n" }}
        {{- developer_message }}
        {{- "\n\n" }}
    {%- endif %}
    {%- if tools -%}
        {{- "# Tools\n\n" }}
        {{- render_tool_namespace("functions", tools) }}
    {%- endif -%}
    {{- "<|end|>" }}
{%- endif %}

{#- Render messages #}
{%- set last_tool_call = namespace(name=none) %}
{%- for message in loop_messages -%}
    {#- At this point only assistant/user/tool messages should remain #}
    {%- if message.role == 'assistant' -%}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "thinking" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "tool_calls" in message %}
            {#- We need very careful handling here - we want to drop the tool call analysis message if the model #}
            {#- has output a later <|final|> message, but otherwise we want to retain it. This is the only case #}
            {#- when we render CoT/analysis messages in inference. #}
            {%- set future_final_message = namespace(found=false) %}
            {%- for future_message in loop_messages[loop.index:] %}
                {%- if future_message.role == 'assistant' and "tool_calls" not in future_message %}
                    {%- set future_final_message.found = true %}
                {%- endif %}
            {%- endfor %}
            {#- We assume max 1 tool call per message, and so we infer the tool call name #}
            {#- in "tool" messages from the most recent assistant tool call name #}
            {%- set tool_call = message.tool_calls[0] %}
            {%- if tool_call.function %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {%- if message.content and message.thinking %}
                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}
            {%- elif message.content and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}
            {%- elif message.thinking and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {{- "<|start|>assistant to=" }}
            {{- "functions." + tool_call.name + "<|channel|>commentary " }}
            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}
            {{- tool_call.arguments|tojson }}
            {{- "<|call|>" }}
            {%- set last_tool_call.name = tool_call.name %}
        {%- elif loop.last and not add_generation_prompt %}
            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}
            {#- This is a situation that should only occur in training, never in inference. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {#- <|return|> indicates the end of generation, but <|end|> does not #}
            {#- <|return|> should never be an input to the model, but we include it as the final token #}
            {#- when training, so the model learns to emit it. #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}
        {%- else %}
            {#- CoT is dropped during all previous turns, so we never render it for inference #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
            {%- set last_tool_call.name = none %}
        {%- endif %}
    {%- elif message.role == 'tool' -%}
        {%- if last_tool_call.name is none %}
            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
        {%- endif %}
        {{- "<|start|>functions." + last_tool_call.name }}
        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}
    {%- elif message.role == 'user' -%}
        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}
    {%- endif -%}
{%- endfor -%}

{#- Generation prompt #}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}



{
  "architectures": [
    "GptOssForCausalLM"
  ],
  "attention_bias": true,
  "attention_dropout": 0.0,
  "eos_token_id": 200002,
  "experts_per_token": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2880,
  "initial_context_length": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 2880,
  "layer_types": [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention"
  ],
  "max_position_embeddings": 131072,
  "model_type": "gpt_oss",
  "num_attention_heads": 64,
  "num_experts_per_tok": 4,
  "num_hidden_layers": 24,
  "num_key_value_heads": 8,
  "num_local_experts": 32,
  "output_router_logits": false,
  "pad_token_id": 199999,
  "quantization_config": {
    "modules_to_not_convert": [
      "model.layers.*.self_attn",
      "model.layers.*.mlp.router",
      "model.embed_tokens",
      "lm_head"
    ],
    "quant_method": "mxfp4"
  },
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "original_max_position_embeddings": 4096,
    "rope_type": "yarn",
    "truncate": false
  },
  "rope_theta": 150000,
  "router_aux_loss_coef": 0.9,
  "sliding_window": 128,
  "swiglu_limit": 7.0,
  "tie_word_embeddings": false,
  "transformers_version": "4.55.0.dev0",
  "use_cache": true,
  "vocab_size": 201088
}


{
  "bos_token_id": 199998,
  "do_sample": true,
  "eos_token_id": [
    200002,
    199999,
    200012
  ],
  "pad_token_id": 199999,
  "transformers_version": "4.55.0.dev0"
}



{"metadata": {"total_size": 13761264768}, "weight_map": {"model.layers.0.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.0.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.0.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.0.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.0.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.1.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.1.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.1.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.1.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.1.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.1.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.1.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.1.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.1.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.1.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.1.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.10.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.10.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.10.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.10.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.10.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.10.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.10.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.10.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.10.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.10.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.10.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.11.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.11.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.11.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.11.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.11.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.11.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.11.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.11.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.11.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.11.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.11.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.12.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.12.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.12.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.12.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.12.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.12.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.12.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.12.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.12.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.12.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.12.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.13.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.13.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.13.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.13.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.13.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.13.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.13.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.13.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.13.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.13.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.13.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.14.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.14.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.14.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.14.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.14.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.14.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.14.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.14.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.14.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.14.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.14.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.15.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.15.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.15.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.15.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.15.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.15.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.15.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.15.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.15.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.15.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.15.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.16.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.16.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.16.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.16.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.16.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.16.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.16.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.16.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.16.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.16.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.16.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.17.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.q_proj.weight": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.k_proj.weight": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.v_proj.weight": "model-00000-of-00002.safetensors", "model.layers.17.self_attn.sinks": "model-00000-of-00002.safetensors", "model.layers.17.mlp.router.bias": "model-00000-of-00002.safetensors", "model.layers.17.mlp.router.weight": "model-00000-of-00002.safetensors", "model.layers.17.mlp.experts.gate_up_proj_bias": "model-00000-of-00002.safetensors", "model.layers.17.mlp.experts.gate_up_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.17.mlp.experts.gate_up_proj_scales": "model-00000-of-00002.safetensors", "model.layers.17.mlp.experts.down_proj_bias": "model-00000-of-00002.safetensors", "model.layers.17.mlp.experts.down_proj_blocks": "model-00000-of-00002.safetensors", "model.layers.17.mlp.experts.down_proj_scales": "model-00000-of-00002.safetensors", "model.layers.17.post_attention_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.18.input_layernorm.weight": "model-00000-of-00002.safetensors", "model.layers.18.self_attn.o_proj.bias": "model-00000-of-00002.safetensors", "model.layers.18.self_attn.o_proj.weight": "model-00000-of-00002.safetensors", "model.layers.18.self_attn.q_proj.bias": "model-00000-of-00002.safetensors", "model.layers.18.self_attn.k_proj.bias": "model-00000-of-00002.safetensors", "model.layers.18.self_attn.v_proj.bias": "model-00000-of-00002.safetensors", "model.layers.18.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.18.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.18.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.18.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.18.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.18.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.18.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.18.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.18.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.18.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.18.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.18.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.18.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.19.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.19.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.19.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.19.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.19.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.19.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.19.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.19.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.19.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.19.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.19.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.2.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.2.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.2.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.2.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.2.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.2.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.2.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.2.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.2.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.2.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.2.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.20.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.20.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.20.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.20.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.20.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.20.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.20.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.20.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.20.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.20.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.20.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.21.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.21.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.21.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.21.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.21.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.21.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.21.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.21.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.21.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.21.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.21.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.22.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.22.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.22.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.22.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.22.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.22.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.22.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.22.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.22.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.22.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.22.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.23.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.23.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.23.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.23.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.23.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.23.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.23.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.23.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.23.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.23.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.23.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.3.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.3.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.3.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.3.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.3.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.3.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.3.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.3.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.3.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.3.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.3.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.4.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.4.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.4.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.4.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.4.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.4.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.4.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.4.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.4.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.4.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.4.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.5.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.5.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.5.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.5.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.5.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.5.mlp.experts.gate_up_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.5.mlp.experts.gate_up_proj_scales": "model-00001-of-00002.safetensors", "model.layers.5.mlp.experts.down_proj_bias": "model-00001-of-00002.safetensors", "model.layers.5.mlp.experts.down_proj_blocks": "model-00001-of-00002.safetensors", "model.layers.5.mlp.experts.down_proj_scales": "model-00001-of-00002.safetensors", "model.layers.5.post_attention_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.6.input_layernorm.weight": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.o_proj.bias": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.o_proj.weight": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.q_proj.bias": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.k_proj.bias": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.v_proj.bias": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.k_proj.weight": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.v_proj.weight": "model-00001-of-00002.safetensors", "model.layers.6.self_attn.sinks": "model-00001-of-00002.safetensors", "model.layers.6.mlp.router.bias": "model-00001-of-00002.safetensors", "model.layers.6.mlp.router.weight": "model-00001-of-00002.safetensors", "model.layers.6.mlp.experts.gate_up_proj_bias": "model-00001-of-00002.safetensors", "model.layers.6.mlp.experts.gate_up_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.6.mlp.experts.gate_up_proj_scales": "model-00002-of-00002.safetensors", "model.layers.6.mlp.experts.down_proj_bias": "model-00002-of-00002.safetensors", "model.layers.6.mlp.experts.down_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.6.mlp.experts.down_proj_scales": "model-00002-of-00002.safetensors", "model.layers.6.post_attention_layernorm.weight": "model-00002-of-00002.safetensors", "model.layers.7.input_layernorm.weight": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.o_proj.bias": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.o_proj.weight": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.q_proj.bias": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.k_proj.bias": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.v_proj.bias": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.q_proj.weight": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.k_proj.weight": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.v_proj.weight": "model-00002-of-00002.safetensors", "model.layers.7.self_attn.sinks": "model-00002-of-00002.safetensors", "model.layers.7.mlp.router.bias": "model-00002-of-00002.safetensors", "model.layers.7.mlp.router.weight": "model-00002-of-00002.safetensors", "model.layers.7.mlp.experts.gate_up_proj_bias": "model-00002-of-00002.safetensors", "model.layers.7.mlp.experts.gate_up_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.7.mlp.experts.gate_up_proj_scales": "model-00002-of-00002.safetensors", "model.layers.7.mlp.experts.down_proj_bias": "model-00002-of-00002.safetensors", "model.layers.7.mlp.experts.down_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.7.mlp.experts.down_proj_scales": "model-00002-of-00002.safetensors", "model.layers.7.post_attention_layernorm.weight": "model-00002-of-00002.safetensors", "model.layers.8.input_layernorm.weight": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.o_proj.bias": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.o_proj.weight": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.q_proj.bias": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.k_proj.bias": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.v_proj.bias": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.q_proj.weight": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.k_proj.weight": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.v_proj.weight": "model-00002-of-00002.safetensors", "model.layers.8.self_attn.sinks": "model-00002-of-00002.safetensors", "model.layers.8.mlp.router.bias": "model-00002-of-00002.safetensors", "model.layers.8.mlp.router.weight": "model-00002-of-00002.safetensors", "model.layers.8.mlp.experts.gate_up_proj_bias": "model-00002-of-00002.safetensors", "model.layers.8.mlp.experts.gate_up_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.8.mlp.experts.gate_up_proj_scales": "model-00002-of-00002.safetensors", "model.layers.8.mlp.experts.down_proj_bias": "model-00002-of-00002.safetensors", "model.layers.8.mlp.experts.down_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.8.mlp.experts.down_proj_scales": "model-00002-of-00002.safetensors", "model.layers.8.post_attention_layernorm.weight": "model-00002-of-00002.safetensors", "model.layers.9.input_layernorm.weight": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.o_proj.bias": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.o_proj.weight": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.q_proj.bias": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.k_proj.bias": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.v_proj.bias": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.q_proj.weight": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.k_proj.weight": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.v_proj.weight": "model-00002-of-00002.safetensors", "model.layers.9.self_attn.sinks": "model-00002-of-00002.safetensors", "model.layers.9.mlp.router.bias": "model-00002-of-00002.safetensors", "model.layers.9.mlp.router.weight": "model-00002-of-00002.safetensors", "model.layers.9.mlp.experts.gate_up_proj_bias": "model-00002-of-00002.safetensors", "model.layers.9.mlp.experts.gate_up_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.9.mlp.experts.gate_up_proj_scales": "model-00002-of-00002.safetensors", "model.layers.9.mlp.experts.down_proj_bias": "model-00002-of-00002.safetensors", "model.layers.9.mlp.experts.down_proj_blocks": "model-00002-of-00002.safetensors", "model.layers.9.mlp.experts.down_proj_scales": "model-00002-of-00002.safetensors", "model.layers.9.post_attention_layernorm.weight": "model-00002-of-00002.safetensors", "model.embed_tokens.weight": "model-00002-of-00002.safetensors", "model.norm.weight": "model-00002-of-00002.safetensors", "lm_head.weight": "model-00002-of-00002.safetensors"}}



{
  "bos_token": "<|startoftext|>",
  "eos_token": "<|return|>",
  "pad_token": "<|endoftext|>"
}



{
  "added_tokens_decoder": {
    "199998": {
      "content": "<|startoftext|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "199999": {
      "content": "<|endoftext|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200000": {
      "content": "<|reserved_200000|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200001": {
      "content": "<|reserved_200001|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200002": {
      "content": "<|return|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200003": {
      "content": "<|constrain|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200004": {
      "content": "<|reserved_200004|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200005": {
      "content": "<|channel|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200006": {
      "content": "<|start|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200007": {
      "content": "<|end|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200008": {
      "content": "<|message|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200009": {
      "content": "<|reserved_200009|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200010": {
      "content": "<|reserved_200010|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200011": {
      "content": "<|reserved_200011|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200012": {
      "content": "<|call|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200013": {
      "content": "<|reserved_200013|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200014": {
      "content": "<|reserved_200014|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200015": {
      "content": "<|reserved_200015|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200016": {
      "content": "<|reserved_200016|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200017": {
      "content": "<|reserved_200017|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "200018": {
      "content": "<|endofprompt|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "bos_token": "<|startoftext|>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "<|return|>",
  "extra_special_tokens": {},
  "model_input_names": [
    "input_ids",
    "attention_mask"
  ],
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "<|endoftext|>",
  "tokenizer_class": "PreTrainedTokenizerFast"
}



</gpt_oss_info>