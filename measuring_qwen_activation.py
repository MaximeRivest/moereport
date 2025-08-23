#!/usr/bin/env python3
# analyze_experts_qwen3_coder_fp8.py
# Works with Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 (Qwen3-MoE).
# Captures MoE gating from `mlp.gate` (router logits) and computes top-k experts per token.

import os
import json
import time
import re
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, Cache  # keep both

# -------------------- Qwen/Transformers compatibility shims --------------------
# Some repos call get_usable_length(new_seq_length, layer_idx). On newer TF, public API is get_seq_length(layer_idx).
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        if hasattr(self, "get_seq_length"):
            try:
                return int(self.get_seq_length(layer_idx))
            except TypeError:
                return int(self.get_seq_length())
        return int(getattr(self, "seen_tokens", 0))
    DynamicCache.get_usable_length = _get_usable_length
    try:
        if not hasattr(Cache, "get_usable_length"):
            Cache.get_usable_length = _get_usable_length
    except Exception:
        pass
# ------------------------------------------------------------------------------

# Regex helpers to extract layer index from module names
_LAYER_PATTERNS = [
    r"\.layers\.(\d+)\.",
    r"\.blocks\.(\d+)\.",
    r"\.h\.(\d+)\.",
    r"\.layer\.(\d+)\.",
]
def _extract_layer_idx(name: str) -> int:
    for pat in _LAYER_PATTERNS:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return -1

def setup_gpus() -> int:
    """Detect GPUs and print inventory."""
    if not torch.cuda.is_available():
        print("[Setup] CUDA not available; using CPU (this FP8 model will not run on CPU).")
        return 0
    n = torch.cuda.device_count()
    print(f"[Setup] Found {n} CUDA device(s)")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / (1024**3):.1f} GiB")
    torch.cuda.set_device(0)
    return n

def make_device_map(config, num_gpus: int) -> Dict[str, int]:
    """
    Evenly shard transformer layers across available GPUs.
    Keeps embeddings on GPU0 and final norm/lm_head on the last GPU.
    """
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_layers <= 0 or num_gpus <= 1:
        return {"": 0}  # single device

    layers_per = num_layers // num_gpus
    remainder = num_layers % num_gpus

    device_map = {"model.embed_tokens": 0, "model.norm": num_gpus - 1, "lm_head": num_gpus - 1}
    cur = 0
    for gid in range(num_gpus):
        cnt = layers_per + (1 if gid < remainder else 0)
        for _ in range(cnt):
            if cur < num_layers:
                device_map[f"model.layers.{cur}"] = gid
                cur += 1
    return device_map

# --------------------------- Hooking for Qwen3-MoE -----------------------------
def register_qwen_gate_hooks(model, config):
    """
    Hook `*.mlp.gate` modules to capture router logits (Tensor) per step.
    Returns: handles, step_buffer, moe_layers(list[int]), routing_config(dict)
    """
    step_buffer: Dict[int, Any] = {}
    handles = []
    moe_layers = set()

    routing_config = {
        # from config.json of Qwen3-MoE
        "num_experts": int(getattr(config, "num_experts", 0)),
        "num_experts_per_tok": int(getattr(config, "num_experts_per_tok", 0)),
        "norm_topk_prob": bool(getattr(config, "norm_topk_prob", True)),
        "decoder_sparse_step": int(getattr(config, "decoder_sparse_step", 1)),
        "mlp_only_layers": list(getattr(config, "mlp_only_layers", []) or []),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
    }

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            try:
                # In Qwen3-MoE, gate is (Replicated)Linear: output is logits, shape [BT, num_experts] or [B, T, E]
                step_buffer[layer_idx] = output.detach().float().cpu()
            except Exception as e:
                print(f"[Hook] Layer {layer_idx} error: {e}")
                step_buffer[layer_idx] = None
        return hook

    for name, module in model.named_modules():
        if name.endswith(".mlp.gate"):
            L = _extract_layer_idx(name)
            if L >= 0:
                moe_layers.add(L)
                handles.append(module.register_forward_hook(make_hook(L)))
                print(f"  Hooked gate at layer {L}: {name}")

    return handles, step_buffer, sorted(moe_layers), routing_config

def extract_experts_from_qwen_buffer(step_buffer, position: int, routing_config, moe_layers: List[int]):
    """
    From captured router logits (or tuple), compute per-layer top-k expert ids and weights for a given absolute position.
    Returns: (layer_experts, issues)
      - layer_experts: dict[layer_idx] -> List[(expert_id, weight)]
      - issues: list[str]
    """
    out: Dict[int, List[Tuple[int, float]]] = {}
    issues: List[str] = []

    k = routing_config["num_experts_per_tok"]
    if k <= 0:
        return out, ["num_experts_per_tok not set (>0)"]

    for L in moe_layers:
        buf = step_buffer.get(L, None)
        if buf is None:
            continue

        # Normalize buffer to a single logits vector for the position of interest
        if torch.is_tensor(buf):
            if buf.dim() == 2:  # [T, E] or [BT, E]
                T = buf.shape[0]
                p = min(max(position, 0), T - 1)
                logits = buf[p]  # [E]
            elif buf.dim() == 3:  # [B, T, E]
                T = buf.shape[1]
                p = min(max(position, 0), T - 1)
                logits = buf[0, p]  # [E]  (we assume B==1 in this script)
            else:
                continue

            # Convert logits -> probabilities, pick top-k, then (optionally) renormalize
            probs = torch.softmax(logits, dim=-1)
            if k > probs.numel():
                issues.append(f"Layer {L}: k={k} > num_experts={probs.numel()}, clamping")
                k_eff = min(k, probs.numel())
            else:
                k_eff = k
            topk = torch.topk(probs, k=k_eff)
            idxs = topk.indices.cpu().tolist()
            wts = topk.values.cpu()
            if routing_config["norm_topk_prob"]:
                s = float(wts.sum().item())
                if s > 0:
                    wts = (wts / s)
            wts = wts.tolist()
            out[L] = [(int(e), float(w)) for e, w in zip(idxs, wts)]

        elif isinstance(buf, tuple) and len(buf) == 2:
            # If future repos return (topk_idx, topk_weight), keep compatibility
            topk_idx, topk_weight = buf
            if topk_idx.dim() == 2:
                T = topk_idx.shape[0]
                p = min(position, T - 1)
                idxs = topk_idx[p].tolist()
                wts = topk_weight[p].tolist()
            elif topk_idx.dim() == 3:
                T = topk_idx.shape[1]
                p = min(position, T - 1)
                idxs = topk_idx[0, p].tolist()
                wts = topk_weight[0, p].tolist()
            else:
                continue
            out[L] = [(int(e), float(w)) for e, w in zip(idxs, wts)]
        else:
            continue

    return out, issues

def verify_qwen_routing(layer_experts: Dict[int, List[Tuple[int, float]]], routing_config) -> List[str]:
    """Basic sanity checks for Qwen3-MoE routing (no groups)."""
    issues = []
    k = routing_config["num_experts_per_tok"]
    norm_topk = routing_config["norm_topk_prob"]
    for L, experts in layer_experts.items():
        if len(experts) != k:
            issues.append(f"Layer {L}: expected {k} experts, got {len(experts)}")
        if norm_topk:
            s = sum(w for _, w in experts)
            if not (0.99 <= s <= 1.01):
                issues.append(f"Layer {L}: weights not normalized (sum={s:.3f})")
    return issues
# ------------------------------------------------------------------------------

def main():
    # Optional: set for FP8 + multi-GPU stability noted in Qwen model card
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    prompts = [
        "Convert this list [[1,2,3], [4,5,6]] into a pandas DataFrame with columns 'A', 'B', 'C'",
        #"Convert this list [[8,9,10], [6,5,4]] into a pandas DataFrame with columns 'A', 'B', 'C'",
        #"Convert this list [[1,2,3], [4,5,6]] into a julia DataFrame with columns 'A', 'B', 'C'",
        #"Convert this list [[1,2,3], [4,5,6]] into a R DataFrame with columns 'A', 'B', 'C'",
    ]

    print("=" * 60)
    print("QWEN3-Coder FP8 — EXPERT ACTIVATION ANALYSIS")
    print("=" * 60)

    num_gpus = setup_gpus()
    model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

    print("\n[Loading] Tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[Loading] Config…")
    config = AutoConfig.from_pretrained(model_name)

    # Build a balanced device_map if multiple GPUs are available
    device_map = make_device_map(config, num_gpus)

    # Heuristic max_memory map: 90% of each device memory (string form accepted by HF)
    max_memory = None
    if num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            total = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            # keep 10% headroom
            use_gb = max(1, int(total * 0.9))
            max_memory[i] = f"{use_gb}GiB"

    # Show device map summary
    n_layers = getattr(config, "num_hidden_layers", 0)
    print(f"\n[Device Map] {n_layers} layers → {num_gpus} GPU(s)")
    if num_gpus > 1:
        for gid in range(num_gpus):
            lay = [k for k, v in device_map.items() if v == gid and "layers" in k]
            print(f"  GPU {gid}: {len(lay)} layers")

    print(f"\n[Loading] Model ({model_name})…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",              # allow fp8-aware loading
        device_map=device_map,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    # Memory snapshot
    if num_gpus > 0:
        print("\n[Memory Distribution]")
        total = 0.0
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            gb = torch.cuda.memory_allocated(i) / (1024**3)
            print(f"  GPU {i}: {gb:.1f} GiB")
            total += gb
        print(f"  Total: {total:.1f} GiB across {num_gpus} GPU(s)")

    # Register hooks
    print("\n[Registering MoE gate hooks]")
    handles, step_buffer, moe_layers, routing_config = register_qwen_gate_hooks(model, config)

    # Expected MoE layers from config: layers not in mlp_only_layers and spaced by decoder_sparse_step
    expected = []
    step = routing_config["decoder_sparse_step"]
    mlp_only = set(routing_config["mlp_only_layers"])
    for i in range(routing_config["num_hidden_layers"]):
        if i in mlp_only:
            continue
        if step <= 1 or (i % step) == 0:
            expected.append(i)

    print("\n[Routing Configuration]")
    print(f"  num_experts          : {routing_config['num_experts']}")
    print(f"  experts_per_token    : {routing_config['num_experts_per_tok']}")
    print(f"  norm_topk_prob       : {routing_config['norm_topk_prob']}")
    print(f"  decoder_sparse_step  : {routing_config['decoder_sparse_step']}")
    print(f"  mlp_only_layers (#)  : {len(routing_config['mlp_only_layers'])}")
    print(f"  MoE layers found     : {len(moe_layers)} (indices like {moe_layers[:5]}…)")

    print("\n[MoE Layer Verification]")
    print(f"  Expected MoE layers  : {len(expected)}")
    print(f"  Actual MoE layers    : {len(moe_layers)}")
    if len(expected) != len(moe_layers):
        print("  WARNING: Mismatch in MoE layer count (naming differences or config overrides?)")

    # ------------------------ Run prompts and collect logs ----------------------
    print(f"\n[Processing] {len(prompts)} prompt(s)…")
    detailed_logs: List[Dict[str, Any]] = []

    expert_counts = defaultdict(lambda: defaultdict(int))
    expert_mass   = defaultdict(lambda: defaultdict(float))

    BATCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    for idx, prompt in enumerate(prompts, start=1):
        print(f"{idx:2d}. {prompt[:60]}…")

        # Use chat template (Qwen provides one)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        batch = tokenizer(
            formatted,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=32768,  # keep reasonable for experimentation
        )
        input_ids = batch["input_ids"].to(BATCH_DEVICE)
        attn_mask = batch["attention_mask"].to(BATCH_DEVICE)

        token_logs = []
        generated_tokens = []
        sanity_issues = []

        max_new = 200
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

        B, L = input_ids.shape
        base_pos = L - 1

        model.eval()
        with torch.inference_mode():
            # 1) Prefill
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values

            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = int(next_id.item())
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)

            # Capture routing for last prompt token (absolute position base_pos)
            L_experts, issues = extract_experts_from_qwen_buffer(
                step_buffer, base_pos, routing_config, moe_layers
            )
            sanity_issues.extend(verify_qwen_routing(L_experts, routing_config))
            sanity_issues.extend(issues)

            # Accumulate per-layer stats
            for L_, pairs in L_experts.items():
                for e_id, w in pairs:
                    expert_counts[L_][e_id] += 1
                    expert_mass[L_][e_id] += w

            token_logs.append({
                "position": 0,
                "token_id": tok_id,
                "token_text": tok_text,
                "layer_experts": L_experts,
            })
            generated_tokens.append(tok_text)

            if eos_id is not None and tok_id == eos_id:
                pass
            else:
                # 2) Decode with KV cache (1 token at a time)
                # Pre-allocate attention mask up to L + max_new for speed
                full_attn = torch.ones((B, L + max_new), dtype=attn_mask.dtype, device=BATCH_DEVICE)
                full_attn[:, :L] = attn_mask

                for step_i in range(1, max_new):
                    cur_len = L + step_i
                    out = model(
                        input_ids=next_id,
                        attention_mask=full_attn[:, :cur_len],
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values

                    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    tok_id = int(next_id.item())
                    tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)

                    # The gate hooks now contain the logits for the single new token we just fed;
                    # use absolute position to keep semantics (clamped inside extractor).
                    L_experts, issues = extract_experts_from_qwen_buffer(
                        step_buffer, base_pos + step_i, routing_config, moe_layers
                    )
                    sanity_issues.extend(verify_qwen_routing(L_experts, routing_config))
                    sanity_issues.extend(issues)

                    for L_, pairs in L_experts.items():
                        for e_id, w in pairs:
                            expert_counts[L_][e_id] += 1
                            expert_mass[L_][e_id] += w

                    token_logs.append({
                        "position": step_i,
                        "token_id": tok_id,
                        "token_text": tok_text,
                        "layer_experts": L_experts,
                    })
                    generated_tokens.append(tok_text)

                    if eos_id is not None and tok_id == eos_id:
                        break

        detailed_logs.append({
            "prompt_idx": idx - 1,
            "prompt": prompt,
            "generated_text": "".join(generated_tokens),
            "moe_layers": moe_layers,
            "routing_config": routing_config,
            "token_logs": token_logs,
            "sanity_issues": sanity_issues[:10] if sanity_issues else [],
        })

    # Remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # --------------------------- Aggregate statistics ---------------------------
    total_activations = 0
    unique_experts = set()
    layer_stats = {}

    for L_idx, layer_counts in expert_counts.items():
        layer_masses = expert_mass[L_idx]
        layer_stats[L_idx] = {
            "unique_experts": len(layer_counts),
            "total_activations": sum(layer_counts.values()),
            "total_mass": float(sum(layer_masses.values())),
            "top_experts": sorted(
                [(e, layer_counts[e], layer_masses[e]) for e in layer_counts],
                key=lambda x: x[2],
                reverse=True
            )[:5],
        }
        for e in layer_counts:
            total_activations += layer_counts[e]
            unique_experts.add((L_idx, e))

    total_experts = len(moe_layers) * routing_config["num_experts"]

    print("\n[Results]")
    print(f"  MoE layers: {len(moe_layers)}")
    print(f"  Total experts: {total_experts:,}")
    print(f"  Unique experts activated: {len(unique_experts):,}")
    if total_experts > 0:
        print(f"  Expert activation ratio: {len(unique_experts)/total_experts*100:.2f}%")
    print(f"  Total activations: {total_activations:,}")

    print("\n[Most Active Layers by Mass]")
    for L_idx, stats in sorted(layer_stats.items(), key=lambda x: x[1]["total_mass"], reverse=True)[:5]:
        print(f"  Layer {L_idx}: "
              f"{stats['unique_experts']} unique, "
              f"{stats['total_activations']} activations, "
              f"mass={stats['total_mass']:.2f}")

    # Save JSON
    summary = {
        "model": model_name,
        "num_gpus": num_gpus,
        "num_prompts": len(prompts),
        "routing_config": routing_config,
        "moe_layers": moe_layers,
        "total_experts": total_experts,
        "unique_experts_activated": len(unique_experts),
        "expert_activation_ratio": (len(unique_experts)/total_experts*100) if total_experts > 0 else 0,
        "total_activations": total_activations,
        "counts_per_layer": {int(L): {int(e): c for e, c in d.items()} for L, d in expert_counts.items()},
        "mass_per_layer": {int(L): {int(e): float(m) for e, m in d.items()} for L, d in expert_mass.items()},
    }
    with open("expert_activation_qwen3_coder_fp8.json", "w") as f:
        json.dump({"summary": summary, "details": detailed_logs}, f, indent=2)
    print("\n[Saved] expert_activation_qwen3_coder_fp8.json")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE (Qwen3-MoE)")
    print("=" * 60)

if __name__ == "__main__":
    main()
