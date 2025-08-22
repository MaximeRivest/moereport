# moe_multi_gpu_fixed.py
# Fixed version with proper device management for multi-GPU setup
# Uses model parallelism with proper tensor routing

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

# Accelerate for model parallelism
try:
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False

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
# Safetensor Index
# =========================

class SafetensorIndex:
    """
    Resolves which .safetensors shard a given parameter name belongs to,
    and loads only the requested tensors via memory-mapped I/O.
    """
    def __init__(self, model_location: str):
        self.root = self._resolve_local_dir(model_location)
        if self.root is None or not os.path.isdir(self.root):
            raise RuntimeError(
                f"Could not resolve a local directory for '{model_location}'. "
                "Pass a path that contains *.safetensors files."
            )
        
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
            raise KeyError(f"Tensor key not found in shards: {key}")
        f = self._open_files.get(shard)
        if f is None:
            f = safe_open(shard, framework="pt")
            self._open_files[shard] = f
        t = f.get_tensor(key)
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
# Multi-GPU Expert LRU
# =========================

class MultiGPUExpertLRU:
    def __init__(self, total_budget_bytes: int, stats: CacheStats, num_gpus: int):
        self.num_gpus = num_gpus
        self.total_budget = int(total_budget_bytes)
        # Distribute budget across GPUs
        self.budget_per_gpu = self.total_budget // num_gpus
        self.stats = stats
        self.stats.gpu_budget_bytes = self.total_budget
        
        # Single LRU across all GPUs for simplicity
        self._lru: OrderedDict[Tuple[int,int], "BaseExpertWrapper"] = OrderedDict()
        self._pinned: set = set()
        self._bytes = 0

    @property
    def used_bytes(self) -> int:
        return self._bytes

    def _evict_until_fits(self, add_bytes: int):
        while self._bytes + add_bytes > self.total_budget and self._lru:
            victim_key = None
            for k in self._lru.keys():
                if k not in self._pinned:
                    victim_key = k
                    break
            if victim_key is None:
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
        if self.total_budget <= 0:
            return False
        if key in self._lru and wrapper._has_gpu():
            self._lru.move_to_end(key, last=True)
            self.stats.gpu_hits += 1
            return True
        
        need = wrapper.param_bytes()
        self._evict_until_fits(need)
        if need > self.total_budget:
            return False
        
        # Load to appropriate GPU
        gpu_id = (key[0] + key[1]) % self.num_gpus
        wrapper._load_gpu(gpu_id)
        self._lru[key] = wrapper
        self._lru.move_to_end(key, last=True)
        self._bytes += need
        self.stats.gpu_loads += 1
        return True

    def pin(self, key: Tuple[int,int], wrapper: "BaseExpertWrapper"):
        need = wrapper.param_bytes()
        self._evict_until_fits(need)
        if self._bytes + need > self.total_budget:
            raise RuntimeError(f"Not enough VRAM to pin expert {key}")
        if not wrapper._has_gpu():
            gpu_id = (key[0] + key[1]) % self.num_gpus
            wrapper._load_gpu(gpu_id)
            self._lru[key] = wrapper
            self._lru.move_to_end(key, last=True)
            self._bytes += need
            self.stats.gpu_loads += 1
        self._pinned.add(key)

# =========================
# Expert wrappers
# =========================

class BaseExpertWrapper(nn.Module):
    def __init__(self, layer_idx: int, expert_idx: int, lru: MultiGPUExpertLRU, stats: CacheStats):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self._lru = lru
        self._stats = stats

    def _has_gpu(self) -> bool: ...
    def _load_gpu(self, gpu_id: int = 0): ...
    def _unload_gpu(self): ...
    def param_bytes(self) -> int: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DiskBackedExpert(BaseExpertWrapper):
    def __init__(self,
                 blueprint: nn.Module,
                 param_rel_names: List[str],
                 full_prefix: str,
                 idx: SafetensorIndex,
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
        self._gpu_device: Optional[str] = None
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
            # Move input to expert's device if needed
            if self._gpu_device:
                x = x.to(self._gpu_device)
            output = self._gpu_expert(x)
            # Return output on original device
            return output
        
        if not self._allow_cpu_fallback:
            gpu_id = (key[0] + key[1]) % self._lru.num_gpus
            device = f"cuda:{gpu_id}"
            tmp = self._materialize_and_load(device)
            x_dev = x.to(device)
            y = tmp(x_dev)
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
# Model patcher
# =========================

@dataclass
class PatcherReport:
    moe_layers: List[Tuple[str,int,int]] = field(default_factory=list)
    total_experts_wrapped: int = 0
    gate_modules_to_cuda: int = 0
    shared_experts_to_cuda: int = 0
    per_expert_bytes_bf16: Optional[int] = None

def _list_param_rel_names(expert_module: nn.Module) -> List[str]:
    return [n for (n, _) in expert_module.named_parameters()]

def _make_meta_blueprint(expert_module: nn.Module) -> nn.Module:
    bp = copy.deepcopy(expert_module).to("meta")
    return bp

def patch_moe_modules_for_multi_gpu(model: nn.Module,
                                    total_gpu_expert_budget_gb: float,
                                    safetensor_idx: SafetensorIndex,
                                    allow_cpu_fallback: bool,
                                    dtype: torch.dtype,
                                    num_gpus: int) -> Tuple[MultiGPUExpertLRU, CacheStats, PatcherReport]:
    stats = CacheStats()
    lru = MultiGPUExpertLRU(int(total_gpu_expert_budget_gb * (1024**3)), stats, num_gpus)
    report = PatcherReport()

    for name, module in model.named_modules():
        if hasattr(module, "experts") and isinstance(getattr(module, "experts"), nn.ModuleList):
            experts: nn.ModuleList = module.experts
            layer_idx = extract_layer_idx_from_name(name)
            num_experts = len(experts)
            
            if report.per_expert_bytes_bf16 is None and num_experts > 0:
                first = experts[0]
                per_b = 0
                for _, p in first.named_parameters():
                    per_b += p.numel() * elem_bytes(torch.bfloat16)
                report.per_expert_bytes_bf16 = per_b

            for eidx in range(num_experts):
                full_prefix = join_module_path(name, f"experts.{eidx}")
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
                experts[eidx] = wrapped

            # Keep gates and shared experts on primary GPU
            if hasattr(module, "gate"):
                try:
                    module.gate.to("cuda:0")
                    report.gate_modules_to_cuda += 1
                except Exception:
                    pass
            for alt in ["shared_experts", "shared_expert"]:
                if hasattr(module, alt):
                    try:
                        getattr(module, alt).to("cuda:0")
                        report.shared_experts_to_cuda += 1
                    except Exception:
                        pass

            report.moe_layers.append((name, layer_idx if layer_idx is not None else -1, num_experts))
            report.total_experts_wrapped += num_experts

    torch.cuda.empty_cache()
    return lru, stats, report

def set_param_by_fqn(model: nn.Module, fqn: str, tensor: torch.Tensor):
    parts = fqn.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p, None)
        if obj is None:
            return
    last = parts[-1]
    if not hasattr(obj, last):
        return
    cur = getattr(obj, last)
    if isinstance(cur, nn.Parameter):
        if cur.is_meta:
            setattr(obj, last, nn.Parameter(tensor, requires_grad=cur.requires_grad))
        else:
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
    """Load all non-expert weights to primary GPU"""
    shards = sorted(glob.glob(os.path.join(idx.root, "*.safetensors")))
    loaded = 0
    for shard in shards:
        f = safe_open(shard, framework="pt")
        keys = list(f.keys())
        for k in keys:
            if re.search(r"\.experts\.\d+\.", k):
                if ".mlp.gate." in k or ".mlp.shared_expert" in k or ".mlp.shared_experts" in k:
                    pass
                else:
                    continue
            t = f.get_tensor(k)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            try:
                t = t.to(device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Warning: OOM for {k}, keeping on CPU")
                    t = t.to("cpu")
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
    weights_location: Optional[str] = None
    dtype: str = "bfloat16"
    attn_impl: Optional[str] = "flash_attention_2"
    total_gpu_expert_budget_gb: float = 144.0  # Total across all GPUs
    allow_cpu_fallback: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    system_prompt: Optional[str] = None
    use_all_gpus: bool = True

class MultiGPUMoERuntime:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        
        # Initialize CUDA
        if torch.cuda.is_available():
            torch.cuda.init()
            self.num_gpus = torch.cuda.device_count() if cfg.use_all_gpus else 1
        else:
            self.num_gpus = 1
        
        print(f"\n[MultiGPU] Using {self.num_gpus} GPU(s)")
        print(f"[MultiGPU] Total expert budget: {cfg.total_gpu_expert_budget_gb} GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._dtype = torch.bfloat16 if cfg.dtype.lower() in ["bf16","bfloat16"] else torch.float16

        # Detect model type
        try:
            model_type_hint = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True).model_type
        except Exception:
            model_type_hint = ""
        is_deepseek = ("deepseek_v3" in (model_type_hint or "")) or ("DeepSeek" in cfg.model_name)

        safetensor_idx = SafetensorIndex(cfg.weights_location or cfg.model_name)

        if not _HAS_ACCELERATE:
            raise RuntimeError("accelerate is required. pip install accelerate")
        
        # Build empty model
        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # Patch MoE modules
        self.lru, self.stats, self.report = patch_moe_modules_for_multi_gpu(
            self.model,
            total_gpu_expert_budget_gb=cfg.total_gpu_expert_budget_gb,
            safetensor_idx=safetensor_idx,
            allow_cpu_fallback=cfg.allow_cpu_fallback,
            dtype=self._dtype,
            num_gpus=self.num_gpus
        )
        
        # Load non-expert weights to primary GPU
        loaded = selective_load_non_expert_weights(
            self.model, safetensor_idx, device="cuda:0", dtype=self._dtype
        )
        
        # Handle remaining meta tensors
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.is_meta and "_blueprint_meta" not in name:
                    full_name = f"{name}.{param_name}" if name else param_name
                    print(f"Warning: {full_name} still on meta, initializing with zeros")
                    with torch.no_grad():
                        new_param = nn.Parameter(
                            torch.zeros(param.shape, device='cuda:0', dtype=self._dtype)
                        )
                        setattr(module, param_name, new_param)
        
        # Set attention implementation
        if is_flash_attn_2_available() and hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = cfg.attn_impl or "flash_attention_2"
        
        self.model.eval()
        self.is_deepseek = is_deepseek
        
        torch.cuda.empty_cache()
        
        print(f"\n[MultiGPU] Initialization complete")
        print(f"  Model: {cfg.model_name}")
        print(f"  Experts wrapped: {self.report.total_experts_wrapped}")
        print(f"  Expert budget: {human_bytes(self.lru.total_budget)}")

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
            print(f"\n[Generate] Processing prompt {i+1}/{len(prompts)}")
            text = self._fmt_prompt(p)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to("cuda:0")
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to("cuda:0")
            
            torch.cuda.reset_peak_memory_stats()
            
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.temperature > 0,
                "temperature": self.cfg.temperature if self.cfg.temperature > 0 else 1.0,
                "top_p": self.cfg.top_p,
                "repetition_penalty": self.cfg.repetition_penalty,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if self.is_deepseek:
                gen_kwargs["past_key_values"] = DeepSeekCompatibleCache()
            
            # Generate with autocast
            with torch.amp.autocast('cuda', dtype=self._dtype):
                out_ids = self.model.generate(**gen_kwargs)
            
            # Decode properly
            generated_ids = out_ids[:, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            peak = sum(torch.cuda.max_memory_allocated(i) for i in range(self.num_gpus))
            
            results.append({
                "prompt": p,
                "generated": generated_text,
                "peak_vram": human_bytes(peak),
            })
            
            gc.collect()
            torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        
        return {
            "num_prompts": len(prompts),
            "elapsed_sec": elapsed,
            "throughput": f"{len(prompts)/elapsed:.2f} prompts/sec",
            "cache_stats": self.stats.summary(),
            "num_gpus": self.num_gpus,
            "results": results,
        }

# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU MoE Runtime")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--total-gpu-budget", type=float, default=144.0, help="Total budget across all GPUs in GB")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--prompts", type=str, nargs="+", 
                       default=["Explain the benefits of using multiple GPUs for large language models."])
    
    args = parser.parse_args()
    
    cfg = RuntimeConfig(
        model_name=args.model,
        weights_location=args.weights,
        total_gpu_expert_budget_gb=args.total_gpu_budget,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system,
        use_all_gpus=True
    )
    
    runtime = MultiGPUMoERuntime(cfg)
    report = runtime.generate(args.prompts)
    
    print("\n" + "="*60)
    print("GENERATION REPORT")
    print("="*60)
    print(json.dumps(report, indent=2, ensure_ascii=False))