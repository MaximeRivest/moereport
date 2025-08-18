"""
moe_reporter.py — General MoE routing reporter + pruning plan generator.

Works with HF MoE models that expose `output_router_logits=True`, e.g. Qwen3 MoE.

What it does
------------
- Loads a model + tokenizer
- Runs step-by-step generation to capture router logits exactly at the step that
  predicts each next token
- Aggregates expert usage per layer using *router softmax probabilities*
  (probability mass per expert) and also top-k hit counts
- Reports:
    * First content token stats (skipping whitespace/quotes/newline)
    * Full response stats (all generated tokens)
    * Optional "label token" stats if a regex is provided
- Builds coverage-based pruning plans per layer (95%, 99% by default),
  ensuring at least top_k experts remain
- Estimates VRAM savings from those plans
- Exports a JSON plan you can inspect or apply in custom code

Notes
-----
- “Pruning plan” = list of experts to keep per MoE layer.
- To *enforce* the plan at runtime, you need to modify the router or weights.
  A best-effort soft-mask helper for Qwen is included but disabled by default.

Author: You
"""

import os
import re
import json
import math
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")


# ---------------------------
# Utility: token categorization
# ---------------------------
PUNCT_QUOTES = set(["'", '"', "“", "”", "‘", "’", "«", "»", "`"])

def is_structural_piece(piece: str) -> bool:
    """Return True if this decoded piece is only whitespace/newlines/quotes/punctuation."""
    if piece is None:
        return True
    stripped = piece.strip()
    if stripped == "":
        return True
    # Pure quotes or tiny punctuation
    if all(ch in PUNCT_QUOTES for ch in stripped):
        return True
    # Sometimes a token is just a colon/comma/dot etc.
    if len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped):
        return True
    return False


# ---------------------------
# Aggregator
# ---------------------------
class LayerAggregator:
    """Accumulates probability mass and top-k hit counts per expert for ONE scope."""
    def __init__(self, moe_layers: List[int], num_experts: int):
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.mass = {L: torch.zeros(num_experts, dtype=torch.float64) for L in moe_layers}
        self.topk_hits = {L: torch.zeros(num_experts, dtype=torch.int64) for L in moe_layers}
        self.tokens_seen = 0

    def update_from_router_logits_lastpos(
        self,
        router_logits: List[Optional[torch.Tensor]],
        last_pos: int,
        top_k: int
    ):
        """
        router_logits: list (len=num_hidden_layers) of tensors or None.
                       Each tensor shape is [batch, seq_len, num_experts] or [seq_len, num_experts].
        We read row at [*, last_pos, :] and softmax to probs.
        """
        self.tokens_seen += 1
        for L, layer_logits in enumerate(router_logits):
            if layer_logits is None or L not in self.moe_layers:
                continue

            # Unify shape
            if layer_logits.dim() == 2:
                v = layer_logits[last_pos]  # [num_experts]
            else:
                v = layer_logits[0, last_pos]  # [num_experts]

            # probs over all experts
            probs = torch.softmax(v.to(torch.float32), dim=-1)  # GPU->CPU optional; we keep on CPU later
            self.mass[L] += probs.to(torch.float64).cpu()

            # top-k hits
            top_idx = torch.topk(probs, k=top_k, dim=-1).indices.cpu()
            self.topk_hits[L].index_add_(0, top_idx, torch.ones_like(top_idx, dtype=torch.int64))


class MultiScopeAggregator:
    """Holds aggregators for multiple scopes (first_content, full, label_token)."""
    def __init__(self, moe_layers: List[int], num_experts: int):
        self.first_content = LayerAggregator(moe_layers, num_experts)
        self.full = LayerAggregator(moe_layers, num_experts)
        self.label_token = LayerAggregator(moe_layers, num_experts)  # used only if label detected


# ---------------------------
# Core runner
# ---------------------------
def apply_chat_template(tokenizer, user_text: str, system_text: Optional[str] = None) -> str:
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def generate_with_router_logging(
    model,
    tokenizer,
    prompt: str,
    *,
    system: Optional[str] = None,
    max_new_tokens: int = 64,
    mask_eos_for_steps: int = 0,   # diagnostics: force at least N content steps
    label_regex: Optional[str] = None,
    skip_structural_for_first_content: bool = True,
    aggregator: Optional[MultiScopeAggregator] = None,
    top_k: int = 8,
    eos_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Step-by-step generation for a single prompt, logging router probs at each step.

    Returns a dict with:
      - 'text': generated text (decoded)
      - 'tokens': list of token ids
      - 'pieces': list of decoded pieces per token
      - 'first_content_step': int or None
      - 'label_step': int or None
    """
    # Format
    formatted = apply_chat_template(tokenizer, prompt, system)

    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)
    seq = inputs.input_ids
    attn = inputs.attention_mask

    # Bookkeeping
    pieces = []
    token_ids = []
    decoded_so_far = ""
    first_content_step = None
    label_step = None
    label_pat = re.compile(label_regex, re.IGNORECASE) if label_regex else None

    # EOS ids
    eos_ids = set()
    if eos_token_ids:
        eos_ids.update(eos_token_ids)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # Qwen chat-specific end token (e.g., <|im_end|>)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id >= 0:
            eos_ids.add(im_end_id)
    except Exception:
        pass

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=seq,
                attention_mask=attn,
                output_router_logits=True,
                return_dict=True,
            )
            # Router logits at the last input position govern the NEXT token
            last_pos = seq.shape[1] - 1

            # Language logits for next token
            logits = out.logits[:, -1, :]

            # Optionally mask EOS early to force content for first N steps
            if step < mask_eos_for_steps and len(eos_ids) > 0:
                logits[:, list(eos_ids)] = -float("inf")

            next_id = torch.argmax(logits, dim=-1)  # greedy for diagnostics
            next_id_item = int(next_id[0].item())

            piece = tokenizer.decode([next_id_item], skip_special_tokens=False)
            pieces.append(piece)
            token_ids.append(next_id_item)

            # Update "full" aggregator first (we log routing for each generated token)
            if aggregator is not None:
                aggregator.full.update_from_router_logits_lastpos(
                    out.router_logits, last_pos, top_k=top_k
                )

            # Detect first content token (skip structural)
            if first_content_step is None:
                if skip_structural_for_first_content and is_structural_piece(piece):
                    pass  # keep going
                else:
                    first_content_step = step
                    if aggregator is not None:
                        aggregator.first_content.update_from_router_logits_lastpos(
                            out.router_logits, last_pos, top_k=top_k
                        )

            # Append token to sequence to continue
            seq = torch.cat([seq, next_id.unsqueeze(0)], dim=1)
            attn = torch.cat(
                [
                    attn,
                    torch.ones((1, 1), device=attn.device, dtype=attn.dtype),
                ],
                dim=1,
            )

            decoded_so_far += tokenizer.decode([next_id_item], skip_special_tokens=True)

            # Label detection (optional)
            if label_pat and label_step is None:
                if label_pat.search(decoded_so_far):
                    label_step = step
                    if aggregator is not None:
                        aggregator.label_token.update_from_router_logits_lastpos(
                            out.router_logits, last_pos, top_k=top_k
                        )

            # Stop on EOS tokens
            if next_id_item in eos_ids:
                break

    return {
        "text": decoded_so_far,
        "tokens": token_ids,
        "pieces": pieces,
        "first_content_step": first_content_step,
        "label_step": label_step,
    }


# ---------------------------
# Reporting & pruning
# ---------------------------
def compute_layer_plan_from_mass(
    mass_vec: torch.Tensor,
    coverage: float,
    top_k: int
) -> Tuple[List[int], float]:
    """
    Given a 1D tensor of per-expert mass for a layer, return:
      - keep_indices (list of experts to keep)
      - achieved_coverage (float in [0,1])
    We ensure at least 'top_k' experts are kept.
    """
    total = float(mass_vec.sum().item()) if mass_vec.numel() > 0 else 0.0
    if total <= 0.0:
        return [], 0.0
    order = torch.argsort(mass_vec, descending=True)
    cumsum = torch.cumsum(mass_vec[order], dim=0)
    target = coverage * total

    # minimal count to reach target; ensure >= top_k
    idx = int(torch.searchsorted(cumsum, torch.tensor(target)).item())
    keep_n = max(top_k, idx + 1)  # +1 because searchsorted returns index of first >= target
    keep_indices = order[:keep_n].cpu().tolist()
    achieved = float(cumsum[min(keep_n - 1, cumsum.numel() - 1)].item() / total)
    return keep_indices, achieved


def build_pruning_plans(
    agg: LayerAggregator,
    coverage_list: List[float],
    top_k: int
) -> Dict[str, Any]:
    """
    Return per-layer plans for each coverage in coverage_list using probability mass.
    """
    plans = {}
    for cov in coverage_list:
        layer_keep = {}
        for L in agg.moe_layers:
            keep, got = compute_layer_plan_from_mass(agg.mass[L], cov, top_k)
            layer_keep[L] = {
                "keep": keep,
                "achieved": got,
                "keep_count": len(keep),
                "experts_total": agg.num_experts,
            }
        plans[f"coverage_{int(cov*100)}"] = layer_keep
    return plans


def estimate_vram_savings(
    plan: Dict[int, Dict[str, Any]],
    num_experts: int
) -> float:
    """
    Rough VRAM savings estimate: average keep_count / total experts over MoE layers.
    """
    keep_counts = [info["keep_count"] for _, info in sorted(plan.items())]
    if not keep_counts:
        return 0.0
    avg_keep = sum(keep_counts) / len(keep_counts)
    ratio = avg_keep / num_experts  # fraction of experts kept
    return max(0.0, 1.0 - ratio)


def summarize_agg(
    name: str,
    agg: LayerAggregator,
    top_n: int = 10
) -> str:
    lines = []
    lines.append(f"\n=== {name} ===")
    lines.append(f"Tokens considered: {agg.tokens_seen}")
    for L in sorted(agg.moe_layers):
        mass = agg.mass[L]
        total = float(mass.sum().item())
        if total <= 0:
            continue
        order = torch.argsort(mass, descending=True).cpu().tolist()
        lines.append(f"\nLayer {L}: experts used = {(mass > 0).sum().item()} / {agg.num_experts}")
        lines.append(f"Top-{top_n} experts by probability mass:")
        for i, e in enumerate(order[:top_n], 1):
            share = float(mass[e].item() / total * 100.0)
            lines.append(f"  {i:2d}. Expert {e:3d}: {share:6.2f}% mass")
    return "\n".join(lines)


# ---------------------------
# Optional: soft-prune (Qwen-ish) — OFF by default
# ---------------------------
def attempt_runtime_soft_prune_qwen(model, plan_layer_keep: Dict[int, List[int]], num_experts: int) -> None:
    """
    Best-effort attempt to apply a *soft* gating mask for Qwen-like MoE routers by
    setting very negative biases for pruned experts. This is model- and version-
    dependent and should be used cautiously. Test on a copy of the model.

    plan_layer_keep: {layer_index: [experts_to_keep]}
    """
    NEG = -1e9  # large negative bias
    applied = 0
    for name, module in model.named_modules():
        # Heuristic: find linear layers that look like "router" with bias dim = num_experts
        if hasattr(module, "bias") and hasattr(module, "weight"):
            b = module.bias
            w = module.weight
            if b is None or w is None:
                continue
            # candidates: bias length equals num_experts
            if b.dim() == 1 and b.numel() == num_experts:
                lname = name.lower()
                if any(tag in lname for tag in ["router", "gate", "moe"]):
                    # Try to parse a layer index from the module name (best effort)
                    layer_idx = None
                    for tok in re.findall(r"\d+", lname):
                        try:
                            ti = int(tok)
                            # choose the first number that looks like a layer index
                            layer_idx = ti
                            break
                        except:
                            pass
                    if layer_idx is None or layer_idx not in plan_layer_keep:
                        continue
                    keep = set(plan_layer_keep[layer_idx])
                    mask = torch.ones_like(b, dtype=torch.bool)
                    # experts NOT kept are True in mask -> set bias to NEG
                    for e in range(num_experts):
                        if e in keep:
                            mask[e] = False
                    with torch.no_grad():
                        b[mask] = NEG
                    applied += 1
    if applied == 0:
        print("[soft-prune] No suitable router biases were found to mask. Your model layout may differ.")
    else:
        print(f"[soft-prune] Applied gating masks to {applied} router-like modules.")


# ---------------------------
# Main orchestration
# ---------------------------
def run_moe_report(
    model_name: str,
    prompts: List[str],
    *,
    system: Optional[str] = None,
    dtype: str = "bfloat16",   # 'bfloat16' or 'float16'
    device_map: str = "auto",
    max_new_tokens: int = 32,
    mask_eos_for_steps: int = 0,
    label_regex: Optional[str] = None,   # e.g., r"\b(yes|no|positive|negative)\b"
    coverage_list: List[float] = [0.95, 0.99],
    top_n_print: int = 1,
    try_soft_prune_qwen: bool = False
) -> Dict[str, Any]:

    print(f"Loading model: {model_name} ...")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    print("Model loaded.\n")

    # MoE config
    top_k = getattr(model.config, "num_experts_per_tok", 8)
    num_hidden_layers = model.config.num_hidden_layers
    # A robust way to guess num_experts: often `num_routed_experts` or `num_local_experts`
    num_experts = getattr(model.config, "num_routed_experts", None)
    if num_experts is None:
        num_experts = getattr(model.config, "num_local_experts", 128)
    print("Model Configuration:")
    print(f"  - Hidden layers: {num_hidden_layers}")
    print(f"  - Experts activated per token (top_k): {top_k}")
    print(f"  - Experts per MoE layer (num_experts): {num_experts}")

    # Probe forward to find which layers are MoE (router_logits not None)
    probe_inputs = tokenizer("probe", return_tensors="pt").to(model.device)
    with torch.no_grad():
        probe_out = model(
            **probe_inputs, output_router_logits=True, return_dict=True
        )
    moe_layers = [L for L, x in enumerate(probe_out.router_logits) if x is not None]
    print(f"  - MoE layers detected: {len(moe_layers)} at indices {moe_layers}\n")

    # Aggregators
    aggs = MultiScopeAggregator(moe_layers, num_experts)

    # EOS token ids to recognize chat end
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id >= 0:
            eos_ids.append(im_end_id)
    except Exception:
        pass

    # Process prompts
    results = []
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Running…")
        out = generate_with_router_logging(
            model, tokenizer, p,
            system=system,
            max_new_tokens=max_new_tokens,
            mask_eos_for_steps=mask_eos_for_steps,
            label_regex=label_regex,
            skip_structural_for_first_content=True,
            aggregator=aggs,
            top_k=top_k,
            eos_token_ids=eos_ids
        )
        results.append(out)

    # Build pruning plans for each scope
    report: Dict[str, Any] = {"model_name": model_name, "num_experts": num_experts, "top_k": top_k}
    report["moe_layers"] = moe_layers
    report["prompts"] = len(prompts)
    report["max_new_tokens"] = max_new_tokens
    report["mask_eos_for_steps"] = mask_eos_for_steps
    report["label_regex"] = label_regex

    # Summaries
    print(summarize_agg("FIRST CONTENT TOKEN", aggs.first_content, top_n=top_n_print))
    print(summarize_agg("FULL RESPONSE", aggs.full, top_n=top_n_print))
    if any(v > 0 for v in [aggs.label_token.tokens_seen]):
        print(summarize_agg("LABEL TOKEN", aggs.label_token, top_n=top_n_print))

    # Plans (per scope)
    report["plans"] = {}
    for scope_name, agg in [("first_content", aggs.first_content),
                            ("full", aggs.full),
                            ("label_token", aggs.label_token)]:
        coverage_plans = build_pruning_plans(agg, coverage_list, top_k)
        report["plans"][scope_name] = coverage_plans

    # VRAM savings estimates (per coverage, using FULL scope as the default)
    vram_estimates = {}
    full_plans = report["plans"]["full"]
    for cov_key, per_layer in full_plans.items():
        savings = estimate_vram_savings(per_layer, num_experts)
        vram_estimates[cov_key] = {
            "estimated_vram_savings": savings,  # fraction, e.g., 0.76 => 76%
            "per_layer_keep_counts": {int(L): info["keep_count"] for L, info in per_layer.items()}
        }
    report["vram_estimates"] = vram_estimates

    # Optional: Attempt runtime soft prune (Qwen-ish) – OFF by default
    if try_soft_prune_qwen:
        # Example: pick the 95% full-response plan
        chosen = full_plans.get("coverage_95", {})
        plan_layer_keep = {int(L): info["keep"] for L, info in chosen.items()}
        print("[soft-prune] Attempting to apply soft gating masks (experimental)…")
        attempt_runtime_soft_prune_qwen(model, plan_layer_keep, num_experts)

    # Serialize pruning plan (JSON) for downstream use
    plan_path = "moe_pruning_plan.json"
    payload = {
        "model_name": model_name,
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "plans": report["plans"],
        "vram_estimates": report["vram_estimates"],
    }
    with open(plan_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nPruning plans saved to: {plan_path}")

    # Brief human-readable wrap-up
    print("\n=== VRAM SAVINGS (based on FULL scope) ===")
    for cov_key, info in vram_estimates.items():
        pct = info["estimated_vram_savings"] * 100.0
        print(f"  {cov_key}: ~{pct:.1f}% potential savings "
              f"(keep counts per MoE layer: {info['per_layer_keep_counts']})")

    return report, results


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example 1: single-token style (Yes/No or Positive/Negative)

    # Sports equipment review prompts for sentiment classification
    # (The full lists are included below)
    positive_reviews = [
        """Review: "These running shoes are incredible! Perfect cushioning and support."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Best tennis racket I've ever owned. Great power and control."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "The yoga mat has excellent grip and thickness. Worth every penny!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Basketball has perfect bounce and grip. Professional quality!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "These dumbbells are solid and well-balanced. Excellent purchase."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Golf clubs exceeded expectations. Amazing distance and accuracy."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "The bike helmet is lightweight and comfortable. Very safe design."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Swimming goggles don't leak at all. Crystal clear vision underwater!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Hiking boots are extremely durable and waterproof. Love them!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Jump rope is the perfect length and weight. Smooth rotation."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Resistance bands are strong and versatile. Great for home workouts."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Soccer ball maintains perfect shape. Professional feel and quality."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "The weightlifting belt provides excellent support. Very sturdy construction."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Skateboard has smooth wheels and stable deck. Rides like a dream!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Boxing gloves fit perfectly and protect well. Outstanding quality."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "The foam roller works wonders for recovery. Firm but comfortable."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Climbing harness is super secure and adjustable. Feel very safe."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Baseball glove has perfect pocket formation. Catches everything!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Exercise bike is quiet and smooth. Display is very accurate."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Gym bag has tons of compartments. Durable material and zippers."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Pull-up bar is rock solid when installed. Holds weight perfectly."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Paddle board is stable and easy to maneuver. Great for beginners!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Ski goggles have amazing anti-fog coating. Clear vision all day."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "The rowing machine operates smoothly and quietly. Excellent cardio!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Volleyball has perfect weight and texture. Tournament quality for sure."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Treadmill is sturdy and has great features. Motor runs perfectly."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Cycling shorts have excellent padding. No discomfort on long rides."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "The punching bag is well-filled and durable. Takes heavy hits well."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Fitness tracker is accurate and comfortable. Battery lasts forever!"
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Hockey stick has perfect flex and weight. Shots are more powerful."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    ]
    negative_reviews = [
        """Review: "Running shoes fell apart after two weeks. Terrible quality."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Tennis racket strings broke on first use. Complete waste of money."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Yoga mat is too thin and slippery. Keeps sliding during practice."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Basketball loses air constantly. Won't maintain proper pressure."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Dumbbells have uneven weight distribution. Paint chips off easily."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Golf clubs bend too easily. Grips came loose after few games."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Bike helmet cracked on minor impact. Not safe at all."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Swimming goggles leak constantly. Straps broke within a week."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Hiking boots gave me blisters immediately. Not waterproof as claimed."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Jump rope handle broke on day one. Cord is too short."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Resistance bands snapped during light use. Dangerous and poorly made."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Soccer ball went flat after one game. Stitching came undone."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Weightlifting belt buckle broke immediately. Material is very cheap."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Skateboard wheels are uneven. Bearings seized up quickly."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Boxing gloves padding compressed flat. No protection anymore."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Foam roller is too soft to be effective. Started falling apart."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Climbing harness has weak stitching. Would not trust my life to it."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Baseball glove is stiff and won't break in. Laces keep breaking."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Exercise bike makes horrible noises. Display stopped working."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Gym bag zipper broke first day. Material ripped at seams."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Pull-up bar bent under normal weight. Mounting hardware failed."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Paddle board has multiple leaks. Impossible to stay inflated."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Ski goggles fog up instantly. Strap adjustment broke off."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Rowing machine resistance is inconsistent. Seat is uncomfortable."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Volleyball is lopsided and won't fly straight. Material feels cheap."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Treadmill belt slips constantly. Motor overheats after 10 minutes."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Cycling shorts padding is too thin. Seams cause painful chafing."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Punching bag leaked sand everywhere. Chain attachment broke."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Fitness tracker gives inaccurate readings. Won't sync with phone."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
        """Review: "Hockey stick blade separated from shaft. Grip tape peeled off."
    Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    ]
    prompts = positive_reviews + negative_reviews

    run_moe_report(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts=prompts,
        system=None,
        dtype="bfloat16",
        device_map="auto",
        max_new_tokens=32,         # increase for longer outputs
        mask_eos_for_steps=0,      # set to 1–2 for classification to prevent immediate EOS
        label_regex=r"\b(yes|no|positive|negative)\b",  # optional: detects label token
        coverage_list=[0.95, 0.99, 1.0],
        top_n_print=10,
        try_soft_prune_qwen=False  # leave False unless you know your router layout
    )
