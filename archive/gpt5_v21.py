# moe_log_and_prune.py

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class TokenRouting:
    gen_pos: int                 # 0-based index within generated tokens
    token_id: int
    piece: str
    per_layer_topk: Dict[int, List[int]]                  # {layer_idx: [expert_ids]}
    per_layer_probs: Optional[Dict[int, List[float]]] = None  # Optional top-k probs per layer


@dataclass
class PromptRouting:
    prompt: str
    generated_text: str
    tokens: List[TokenRouting] = field(default_factory=list)


@dataclass
class RoutingLog:
    model_name: str
    num_experts: int             # experts per MoE layer
    top_k: int                   # experts per token (router top-k)
    moe_layers: List[int]        # indices (within num_hidden_layers) that are MoE
    num_hidden_layers: int
    prompts: List[PromptRouting] = field(default_factory=list)


# ---------------------------
# 1) Core single-step logger
# ---------------------------

def log_experts_for_step(
    router_logits: List[Optional[torch.Tensor]],
    *,
    last_pos: int,
    moe_layers: List[int],
    top_k: int,
    with_probs: bool = True,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, List[float]]]]:
    """
    Log the routed experts for a SINGLE generation step (the token predicted from `last_pos`).

    Args:
        router_logits: list (len=num_hidden_layers) of tensors or None.
                       Each tensor is either [batch, seq_len, num_experts] or [seq_len, num_experts].
        last_pos:      int, index of the last input position used to predict the next token.
        moe_layers:    indices of layers that are MoE (others will be ignored).
        top_k:         how many experts are routed per token (from model.config).
        with_probs:    also return top-k probabilities per layer (softmax over experts).

    Returns:
        (per_layer_topk, per_layer_probs) where:
            per_layer_topk: {layer_idx: [expert_ids]}
            per_layer_probs: {layer_idx: [probabilities]} or None
    """
    per_layer_topk: Dict[int, List[int]] = {}
    per_layer_probs: Optional[Dict[int, List[float]]] = {} if with_probs else None

    for L, logits in enumerate(router_logits):
        if L not in moe_layers or logits is None:
            continue

        # Normalize shape -> [num_experts]
        if logits.dim() == 3:
            # [batch, seq_len, num_experts]
            v = logits[0, min(last_pos, logits.shape[1] - 1)]
        elif logits.dim() == 2:
            # [seq_len, num_experts]
            v = logits[min(last_pos, logits.shape[0] - 1)]
        else:
            # Unexpected
            continue

        probs = torch.softmax(v.to(torch.float32), dim=-1)
        top = torch.topk(probs, k=top_k, dim=-1)
        per_layer_topk[L] = top.indices.cpu().tolist()
        if with_probs and per_layer_probs is not None:
            per_layer_probs[L] = top.values.cpu().tolist()

    return per_layer_topk, per_layer_probs


# ---------------------------
# Helpers: model probing
# ---------------------------

def _detect_moe_layers(model) -> List[int]:
    """Run a cheap probe to see which transformer layers expose router logits."""
    probe = {"input_ids": torch.tensor([[model.config.eos_token_id or 1]], device=model.device)}
    with torch.no_grad():
        out = model(**probe, output_router_logits=True, return_dict=True)
    return [i for i, t in enumerate(out.router_logits) if t is not None]


def _collect_eos_ids(tokenizer) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    # Chat models often have chat-specific end tokens
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.append(tid)
        except Exception:
            pass
    return sorted(set(ids))


# ---------------------------
# 2) Runner over prompts
# ---------------------------

def capture_routing_for_prompts(
    model_name: str,
    prompts: List[str],
    *,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    temperature: float = 0.0,       # greedy by default
    top_p: Optional[float] = None,   # if you want sampling
    with_probs: bool = True,         # record top-k probabilities per layer
) -> RoutingLog:
    """
    Generate for each prompt while capturing MoE routing per token, per layer.

    Returns:
        RoutingLog (see dataclass)
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"[load] {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype
    )
    print("[ok] model loaded.")

    top_k = getattr(model.config, "num_experts_per_tok", 8)
    num_hidden = model.config.num_hidden_layers
    num_experts = getattr(model.config, "num_routed_experts",
                          getattr(model.config, "num_local_experts", 128))
    moe_layers = _detect_moe_layers(model)

    def apply_chat(msg: str) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": msg})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    eos_ids = set(_collect_eos_ids(tok))

    log = RoutingLog(
        model_name=model_name,
        num_experts=num_experts,
        top_k=top_k,
        moe_layers=moe_layers,
        num_hidden_layers=num_hidden,
    )

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] capture …")
        formatted = apply_chat(prompt)
        inputs = tok(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        seq = inputs.input_ids
        attn = inputs.attention_mask

        pr = PromptRouting(prompt=prompt, generated_text="", tokens=[])

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=seq,
                    attention_mask=attn,
                    output_router_logits=True,
                    return_dict=True,
                )

                # Next-token distribution
                logits = out.logits[:, -1, :]
                if temperature and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    if top_p is not None and 0.0 < top_p < 1.0:
                        # nucleus sampling
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cdf = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = (cdf > top_p).float().argmax(dim=-1)
                        # mask tail per batch=1
                        k = int(cutoff.item()) + 1
                        probs = torch.zeros_like(probs).scatter_(
                            1, sorted_idx[:, :k], sorted_probs[:, :k]
                        )
                        probs.div_(probs.sum(dim=-1, keepdim=True))
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                next_id_item = int(next_id[0, 0].item())
                piece = tok.decode([next_id_item], skip_special_tokens=False)

                # Log MoE routing for THIS step
                last_pos = seq.shape[1] - 1
                per_layer_topk, per_layer_probs = log_experts_for_step(
                    out.router_logits,
                    last_pos=last_pos,
                    moe_layers=moe_layers,
                    top_k=top_k,
                    with_probs=with_probs,
                )

                pr.tokens.append(TokenRouting(
                    gen_pos=step,
                    token_id=next_id_item,
                    piece=tok.decode([next_id_item], skip_special_tokens=True),
                    per_layer_topk=per_layer_topk,
                    per_layer_probs=per_layer_probs,
                ))

                # Update sequence
                seq = torch.cat([seq, next_id], dim=1)
                attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
                pr.generated_text += tok.decode([next_id_item], skip_special_tokens=True)

                if next_id_item in eos_ids:
                    break

        log.prompts.append(pr)

    return log


# ---------------------------
# 3) Summary analytics
# ---------------------------

@dataclass
class Summary:
    # Aggregates
    counts_per_layer_expert: Dict[int, Dict[int, int]]
    mass_per_layer_expert: Optional[Dict[int, Dict[int, float]]]  # if probs present
    # Simple derived metrics
    per_position_unique_experts: Dict[int, int]
    utilization_percent: float
    unique_pairs: Set[Tuple[int, int]]  # (layer, expert)
    top_per_layer: Dict[int, List[Tuple[int, int]]]  # [(expert, count)]


def summarize_routing(log: RoutingLog) -> Summary:
    counts: Dict[int, Dict[int, int]] = {}              # layer -> expert -> count
    mass: Optional[Dict[int, Dict[int, float]]] = {}    # if probs were recorded
    per_pos_unique: Dict[int, Set[Tuple[int, int]]] = {}  # pos -> {(layer, expert)}

    # Initialize
    for L in log.moe_layers:
        counts[L] = {}
        if mass is not None:
            mass[L] = {}

    # Traverse
    for pr in log.prompts:
        for t in pr.tokens:
            pos = t.gen_pos
            if pos not in per_pos_unique:
                per_pos_unique[pos] = set()
            for L, experts in t.per_layer_topk.items():
                for e_idx, e in enumerate(experts):
                    counts[L][e] = counts[L].get(e, 0) + 1
                    per_pos_unique[pos].add((L, e))
                    if mass is not None and t.per_layer_probs and L in t.per_layer_probs:
                        prob = float(t.per_layer_probs[L][e_idx])
                        mass[L][e] = mass[L].get(e, 0.0) + prob

    unique_pairs = set()
    for L, es in counts.items():
        for e in es.keys():
            unique_pairs.add((L, e))

    total_possible = len(log.moe_layers) * log.num_experts
    utilization_percent = (len(unique_pairs) / total_possible * 100.0) if total_possible else 0.0
    per_position_unique_experts = {pos: len(s) for pos, s in per_pos_unique.items()}

    top_per_layer: Dict[int, List[Tuple[int, int]]] = {}
    for L, es in counts.items():
        top_per_layer[L] = sorted(es.items(), key=lambda x: x[1], reverse=True)

    return Summary(
        counts_per_layer_expert=counts,
        mass_per_layer_expert=mass if mass != {} else None,
        per_position_unique_experts=per_position_unique_experts,
        utilization_percent=utilization_percent,
        unique_pairs=unique_pairs,
        top_per_layer=top_per_layer,
    )


# ---------------------------
# 4) Unused experts per layer
# ---------------------------

def find_never_routed_experts(
    log: RoutingLog,
    summary: Optional[Summary] = None
) -> Dict[int, Set[int]]:
    """
    Returns {layer_idx: {expert_ids never routed}}.
    """
    if summary is None:
        summary = summarize_routing(log)

    unused: Dict[int, Set[int]] = {}
    for L in log.moe_layers:
        used = set(summary.counts_per_layer_expert.get(L, {}).keys())
        all_e = set(range(log.num_experts))
        unused[L] = all_e - used
    return unused


# ---------------------------
# 5) Build keep plans & prune exporter
# ---------------------------

def build_keep_map_from_summary(
    summary: Summary,
    *,
    num_experts: int,
    moe_layers: Iterable[int],
    coverage: float = 1.00,
    min_keep: Optional[int] = None,
    use_mass_if_available: bool = True,
) -> Dict[int, List[int]]:
    """
    For each MoE layer, choose a subset of experts to KEEP that covers the requested
    probability mass (if available) or routed-count mass.

    Returns:
        {layer_idx: [expert_ids_to_keep]}
    """
    keep_map: Dict[int, List[int]] = {}

    for L in moe_layers:
        if use_mass_if_available and summary.mass_per_layer_expert is not None:
            vec = summary.mass_per_layer_expert.get(L, {})
        else:
            vec = summary.counts_per_layer_expert.get(L, {})

        # Build dense vector of length num_experts
        dense = torch.zeros(num_experts, dtype=torch.float64)
        for e, v in vec.items():
            dense[e] = float(v)

        total = float(dense.sum().item())
        if total <= 0.0:
            # nothing routed for this layer; keep at least one (or min_keep/top_k)
            k = min_keep or 1
            keep_map[L] = list(range(k))
            continue

        order = torch.argsort(dense, descending=True)
        cumsum = torch.cumsum(dense[order], dim=0)
        target = coverage * total
        idx = int(torch.searchsorted(cumsum, torch.tensor(target)).item())
        keep_n = idx + 1
        if min_keep is not None:
            keep_n = max(keep_n, min_keep)
        keep_n = min(keep_n, num_experts)

        keep_map[L] = order[:keep_n].cpu().tolist()

    return keep_map


def _uniformize_keep_map(keep_map: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], int]:
    """
    Ensure all layers keep the same count. We trim each layer to the MIN count.
    """
    if not keep_map:
        return keep_map, 0
    counts = [len(v) for v in keep_map.values()]
    m = min(counts)
    uniform: Dict[int, List[int]] = {L: v[:m] for L, v in keep_map.items()}
    return uniform, m


# ---- pruning helpers (generic, Mixtral/Qwen-style packed expert dims) ----

_PATTERNS_LAYER_IDX = [
    r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\.",
    r"\.model\.(\d+)\.", r"\.transformer\.(\d+)\."
]

def _infer_layer_idx_from_name(name: str) -> Optional[int]:
    for pat in _PATTERNS_LAYER_IDX:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    # fallback: first plain number token
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _slice_packed_param(
    tensor: torch.Tensor,
    keep: List[int],
    num_experts: int,
    *,
    name: str,
) -> torch.Tensor:
    """
    Try to slice expert-packed parameters. Handles common patterns:

    - 3D with expert axis at dim 0: [E, *]
    - 2D packed along rows: [E * d, in]  -> reshape [E, d, in], slice, reshape back
    - 2D packed along cols: [out, E * d] -> reshape [out, E, d], slice on dim=1, reshape back
    - 1D biases packed: [E * d]          -> reshape [E, d], slice, reshape back
    - Router with explicit E dimension [E, in] or bias [E]

    If the shape doesn’t look like any of these, it’s returned unchanged.
    """
    E = num_experts
    dims = tensor.dim()

    # 3D with experts along dim 0
    if dims >= 3 and tensor.shape[0] == E:
        return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))

    # 2D
    if dims == 2:
        rows, cols = tensor.shape

        # Router-like: [E, in] (slice rows)
        if rows == E:
            return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))

        # Row-packed: [E * d, in]  -> slice rows
        if rows % E == 0 and rows // E > 1:
            d = rows // E
            t = tensor.view(E, d, cols)
            t = t.index_select(0, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(len(keep) * d, cols)

        # Col-packed: [out, E * d] -> slice columns
        if cols % E == 0 and cols // E > 1:
            d = cols // E
            t = tensor.view(rows, E, d)
            t = t.index_select(1, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(rows, len(keep) * d)

        # Router-like: [out, E] (slice cols)
        if cols == E:
            return tensor.index_select(1, torch.as_tensor(keep, device=tensor.device))

    # 1D bias packed: [E * d] or [E]
    if dims == 1:
        n = tensor.shape[0]
        if n == E:
            return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))
        if n % E == 0 and n // E > 1:
            d = n // E
            t = tensor.view(E, d)
            t = t.index_select(0, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(len(keep) * d)

    # Not recognized – leave unchanged
    return tensor


def export_pruned_checkpoint(
    model_name: str,
    keep_map: Dict[int, List[int]],
    *,
    output_dir: str,
    dtype: str = "bfloat16",
    device_map: str = "cpu",
    strict_uniform: bool = True,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Export a *hard-pruned* checkpoint containing only the kept experts.

    Constraints:
      - All MoE layers must keep the same number of experts (HF configs assume uniform E).
      - This function recognizes common packed expert layouts (Qwen/Mixtral). For other
        layouts you may need to extend `_slice_packed_param`.

    Returns:
      A small report dict describing parameter changes. If dry_run=True, no files are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"[load] {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    print("[ok] model loaded.")

    # Validate uniformity
    keep_map_u, keep_count = _uniformize_keep_map(keep_map)
    if strict_uniform:
        lens = {len(v) for v in keep_map_u.values()}
        if len(lens) != 1:
            raise ValueError(
                "Non-uniform keep counts per layer. Use a uniform keep plan "
                "or set strict_uniform=False (not recommended)."
            )

    # Probe to know which layers are MoE and num_experts
    moe_layers = _detect_moe_layers(model)
    num_experts = getattr(model.config, "num_routed_experts",
                          getattr(model.config, "num_local_experts", 128))
    print(f"[prune] MoE layers: {moe_layers}")
    print(f"[prune] experts per layer: {num_experts}, keep: {keep_count}")

    sd = model.state_dict()
    new_sd = {}
    changes: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for name, tensor in sd.items():
        L = _infer_layer_idx_from_name(name)
        if L is None or L not in keep_map_u or L not in moe_layers:
            # Not a MoE layer param (or not identifiable) -> keep as-is
            new_sd[name] = tensor
            continue

        kept = keep_map_u[L]
        new_tensor = _slice_packed_param(tensor, kept, num_experts, name=name)
        if new_tensor.shape != tensor.shape:
            changes.append((name, tuple(tensor.shape), tuple(new_tensor.shape)))
        new_sd[name] = new_tensor

    # Update config – set experts per layer to keep_count (uniform)
    cfg = model.config.to_dict()
    for key in ("num_routed_experts", "num_local_experts"):
        if key in cfg:
            cfg[key] = keep_count

    report = {
        "model_name": model_name,
        "moe_layers": moe_layers,
        "original_num_experts": num_experts,
        "kept_per_layer": keep_count,
        "param_changes": changes,
        "output_dir": output_dir,
        "dry_run": dry_run,
    }

    if dry_run:
        print("\n[dry-run] The following parameters would be sliced:")
        for n, old, new in changes:
            print(f"  - {n}: {old} -> {new}")
        print("\n[dry-run] No files written.")
        return report

    # Write files
    torch.save(new_sd, os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Save tokenizer artifacts (for convenience)
    try:
        tok.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"[ok] wrote pruned checkpoint to: {output_dir}")
    return report


# ---------------------------
# Convenience I/O
# ---------------------------

def save_routing_log(log: RoutingLog, path: str) -> None:
    def _tok(tok: TokenRouting) -> Dict[str, Any]:
        return {
            "gen_pos": tok.gen_pos,
            "token_id": tok.token_id,
            "piece": tok.piece,
            "per_layer_topk": tok.per_layer_topk,
            "per_layer_probs": tok.per_layer_probs,
        }

    payload = {
        "model_name": log.model_name,
        "num_experts": log.num_experts,
        "top_k": log.top_k,
        "moe_layers": log.moe_layers,
        "num_hidden_layers": log.num_hidden_layers,
        "prompts": [
            {
                "prompt": p.prompt,
                "generated_text": p.generated_text,
                "tokens": [_tok(t) for t in p.tokens],
            }
            for p in log.prompts
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_routing_log(path: str) -> RoutingLog:
    with open(path, "r") as f:
        data = json.load(f)

    log = RoutingLog(
        model_name=data["model_name"],
        num_experts=data["num_experts"],
        top_k=data["top_k"],
        moe_layers=data["moe_layers"],
        num_hidden_layers=data["num_hidden_layers"],
    )
    for p in data["prompts"]:
        pr = PromptRouting(prompt=p["prompt"], generated_text=p["generated_text"])
        for t in p["tokens"]:
            pr.tokens.append(TokenRouting(
                gen_pos=t["gen_pos"],
                token_id=t["token_id"],
                piece=t.get("piece", ""),
                per_layer_topk={int(k): v for k, v in t["per_layer_topk"].items()},
                per_layer_probs={int(k): v for k, v in (t.get("per_layer_probs") or {}).items()} if t.get("per_layer_probs") else None,
            ))
        log.prompts.append(pr)
    return log


# ---------------------------
# Example (optional)
# ---------------------------

if __name__ == "__main__":
    prompts = [
        'Review: "This product is amazing!" Sentiment:',
        'Translate to French: "Hello, how are you?"',
    ]

    # 1) Capture routing
    log = capture_routing_for_prompts(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts,
        max_new_tokens=16,
        temperature=0.0,
        with_probs=True,
    )
    save_routing_log(log, "routing_log.json")

    # 2) Summarize
    summ = summarize_routing(log)
    print(f"\nUtilization: {summ.utilization_percent:.2f}% "
          f"({len(summ.unique_pairs)} unique (layer, expert) pairs)")

    # 3) Unused experts
    never = find_never_routed_experts(log, summ)
    for L in sorted(never):
        print(f"Layer {L}: {len(never[L])} never routed experts")

    # 4) Build a 99% mass keep plan (using probs if available) and export
    keep = build_keep_map_from_summary(
        summ,
        num_experts=log.num_experts,
        moe_layers=log.moe_layers,
        coverage=0.99,
        min_keep=log.top_k,            # never keep fewer than top_k
        use_mass_if_available=True,
    )
    # Export (dry run first)
    export_pruned_checkpoint(
        log.model_name,
        keep,
        output_dir="./pruned-qwen-a3b-99",
        strict_uniform=True,
        dry_run=True,                   # set to False after you inspect changes
    )
