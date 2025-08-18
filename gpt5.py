#!/usr/bin/env python3
"""
Mixture-of-Experts routing report with per-layer tallies and stability checks.

What it prints:
  - Overall utilization (unique experts hit across all layers/prompts)
  - Layer-wise density bars
  - Per-layer tallies & stability across prompts:
      sets (distinct expert sets across prompts), constant(True/False),
      intersection/union sizes, mean Jaccard similarity,
      "top" experts by across-prompt presence (E<idx>:count/N)
  - Optional POS vs NEG deltas (only if both groups provided)
  - Global stability summaries (mean/median Jaccard across layers)

Defaults:
  - Model: Qwen/Qwen3-30B-A3B-Instruct-2507
  - Position: "all" (aggregate experts across all token positions in each prompt)

Example:
  python moe_report.py
  python moe_report.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --position all
"""

import argparse
import warnings
from collections import defaultdict, Counter
from itertools import combinations
import math
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

# ---------- CLI ----------
def build_args():
    p = argparse.ArgumentParser(description="MoE per-layer expert tally & stability")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                   help="HF model id or local path")
    p.add_argument("--position", type=str, default="all",
                   choices=["last", "first", "middle", "all"],
                   help="Which token position to base routing on (or aggregate across all)")
    p.add_argument("--max_prompts", type=int, default=24,
                   help="Max prompts to use per test suite (to keep runs quick)")
    p.add_argument("--print_unused", action="store_true",
                   help="Print detailed 'unused experts' lists (can be very long)")
    p.add_argument("--verbose_lists", action="store_true",
                   help="Print full union/intersection lists per layer")
    p.add_argument("--per_prompt", action="store_true",
                   help="Print per-prompt top-k sets for a few examples")
    p.add_argument("--per_prompt_examples", type=int, default=3,
                   help="How many prompts to show if --per_prompt is used")
    p.add_argument("--topn_tally", type=int, default=16,
                   help="How many most frequent experts to show in the 'top:' snippet")
    return p.parse_args()

# ---------- Test prompt suites ----------
def make_high_overlap_prompts(n=24):
    # Similar format and instruction => routing tends to be stable
    base = [
        'Review: "These running shoes are incredible! Perfect cushioning and support."',
        'Review: "Best tennis racket I have owned. Great power and control."',
        'Review: "Yoga mat has excellent grip and thickness. Worth every penny!"',
        'Review: "Basketball has perfect bounce and grip. Professional quality!"',
        'Review: "These dumbbells are solid and well-balanced. Excellent purchase."',
        'Review: "Golf clubs exceeded expectations. Distance and accuracy are amazing."',
        'Review: "Bike helmet is lightweight and comfortable. Very safe design."',
        'Review: "Swimming goggles do not leak. Crystal clear vision underwater!"',
        'Review: "Hiking boots are durable and waterproof. Love them."',
        'Review: "Jump rope has perfect weight. Smooth rotation."',
        'Review: "Resistance bands are strong and versatile. Great for home workouts."',
        'Review: "Soccer ball keeps perfect shape. Professional feel."',
        'Review: "Weightlifting belt provides excellent support. Very sturdy."',
        'Review: "Skateboard has smooth wheels and a stable deck."',
        'Review: "Boxing gloves fit perfectly and protect well. Outstanding quality."',
        'Review: "Foam roller works wonders for recovery. Firm but comfortable."',
        'Review: "Climbing harness is secure and adjustable. I feel safe."',
        'Review: "Exercise bike is quiet and smooth. Display is accurate."',
        'Review: "Gym bag has many compartments. Durable zippers."',
        'Review: "Pull-up bar is rock solid when installed. Holds weight perfectly."',
        'Review: "Paddle board is stable and easy to maneuver."',
        'Review: "Ski goggles have amazing anti-fog coating."',
        'Review: "Rowing machine operates smoothly and quietly."',
        'Review: "Volleyball has perfect weight and texture. Tournament quality."',
        'Review: "Treadmill is sturdy and has great features."',
    ]
    # Instruction kept uniform to push routing toward overlap
    tail = " Is this review positive or negative? Answer only with Yes for positive or No for negative."
    prompts = [(s + tail) for s in base]
    return prompts[:n]

def make_diverse_prompts(n=24):
    # Mixed domains / languages / styles / tasks => routing should vary
    raw = [
        "Summarize the key ideas behind general relativity in two sentences.",
        "Translate to Japanese: 'The library opens at 9am on weekdays.'",
        "Given a SQL table 'sales(order_id, item, price, ts)', write a query for total revenue per day.",
        "Explain the difference between breadth-first search and Dijkstra's algorithm.",
        "Generate a Python function that returns the nth Fibonacci number iteratively.",
        "French to English: 'Le modèle surajuste les données d'entraînement.'",
        "Write a haiku about winter rain.",
        "Prove or disprove: the sum of two rational numbers is rational.",
        "Refactor this snippet for clarity: for i in range(len(a)): b.append(a[i]*2)",
        "Create a 3-day itinerary for first-time visitors to Kyoto.",
        "What are the key trade-offs of event-driven microservices?",
        "Translate to Spanish: 'We should back up the database before upgrading.'",
        "Design a regex to validate an IPv4 address.",
        "Draft a polite email requesting a deadline extension for a project.",
        "Explain Bayesian vs. frequentist inference in one paragraph.",
        "Generate a minimal HTML page with a centered title and a paragraph.",
        "What is the time complexity of quicksort on average?",
        "Write a unit test for a function that calculates factorial.",
        "Translate to German: 'This feature is behind a flag and may change.'",
        "Explain how transformers use self-attention.",
        "Give a short recipe for garlic-lemon roasted chicken.",
        "Rewrite the following to active voice: 'The issue was identified by the team.'",
        "Create a JSON schema for a 'user' object with id, name, and email.",
        "List five commonly confused English word pairs and clarify each.",
        "Draft a 1-sentence elevator pitch for a task management app.",
        "Translate to Arabic: 'The contract will be reviewed by our legal team.'",
        "Explain the purpose of Dockerfiles in containerization.",
    ]
    return raw[:n]

# ---------- Core MoE tracking ----------
def track_single_token_generation(model, tokenizer, prompts, position="all", top_k=8):
    """
    For each prompt, collect experts chosen by the router.

    position:
      - "last"/"first"/"middle": use top-k experts at that token position.
      - "all": union of top-k experts across all token positions (per layer) in the prompt.

    Returns:
      expert_counts: dict[layer][expert] -> occurrence count (across tokens/prompts)
      total_selections: dict[layer] -> number of selections added
      predictions: list[str] next-token predictions (debug)
      per_prompt_topk: list[dict[layer] -> list[int]]  (per-prompt sets, list entries are deduped for stability calc)
    """
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    predictions = []
    per_prompt_topk = []

    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"    Processed {idx + 1}/{len(prompts)} prompts...")

        # Chat template for instruct models
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_router_logits=True,
                return_dict=True,
            )

        # Debug: next-token prediction (not used for routing metrics)
        try:
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            predictions.append(tokenizer.decode([next_token_id]).strip())
        except Exception:
            predictions.append("")

        # Router logits handling
        per_layer_union = {}
        if not hasattr(outputs, "router_logits") or not outputs.router_logits:
            # Some models may not expose router logits
            continue

        for layer_idx, layer_logits in enumerate(outputs.router_logits):
            if layer_logits is None:
                continue

            # layer_logits: (batch, seq, experts) or (batch, experts)
            if layer_logits.dim() == 3:
                # (B, T, E)
                token_matrix = layer_logits[0]  # (T, E)
                T, E = token_matrix.shape

                # select position(s)
                idxs = []
                if position == "last":
                    idxs = [T - 1]
                elif position == "first":
                    idxs = [0]
                elif position == "middle":
                    idxs = [max(0, T // 2)]
                elif position == "all":
                    idxs = list(range(T))
                else:
                    idxs = [T - 1]

                union_set = set()
                for t in idxs:
                    probs = torch.softmax(token_matrix[t], dim=-1)
                    k = min(top_k, probs.shape[-1])
                    top_idx = torch.topk(probs, k).indices.tolist()
                    union_set.update(top_idx)
                    # counts per occurrence (token-level)
                    for e in top_idx:
                        expert_counts[layer_idx][e] += 1
                    total_selections[layer_idx] += k

                per_layer_union[layer_idx] = sorted(list(union_set))

            elif layer_logits.dim() == 2:
                # (B, E) – no token dimension, treat as single position
                probs = torch.softmax(layer_logits[0], dim=-1)
                k = min(top_k, probs.shape[-1])
                top_idx = torch.topk(probs, k).indices.tolist()
                per_layer_union[layer_idx] = sorted(top_idx)
                for e in top_idx:
                    expert_counts[layer_idx][e] += 1
                total_selections[layer_idx] += k

        per_prompt_topk.append(per_layer_union)

    return expert_counts, total_selections, predictions, per_prompt_topk

# ---------- Stats & Printing ----------
def compute_layerwise_stats(per_prompt_topk, num_layers):
    """
    Given per-prompt dicts {layer -> [experts]}, compute per-layer:
      - unique_set_count, is_constant, intersection, union, mean_jaccard, freq
    """
    by_layer_sets = defaultdict(list)  # layer -> list[set]
    for prompt_dict in per_prompt_topk:
        for L, exp_list in prompt_dict.items():
            by_layer_sets[L].append(set(exp_list))

    stats = {}
    for L in range(num_layers):
        sets_list = by_layer_sets.get(L, [])
        if not sets_list:
            stats[L] = {
                "n_prompts": 0, "unique_set_count": 0, "is_constant": False,
                "intersection": [], "union": [], "freq": {}, "mean_jaccard": 0.0
            }
            continue

        unique_sets = set(map(frozenset, sets_list))
        is_constant = (len(unique_sets) == 1)

        inter = set(sets_list[0])
        uni = set(sets_list[0])
        for s in sets_list[1:]:
            inter &= s
            uni |= s

        # presence frequency per prompt
        freq = Counter()
        for s in sets_list:
            for e in s:
                freq[e] += 1

        # mean pairwise Jaccard
        if len(sets_list) >= 2:
            jaccs = []
            for a, b in combinations(sets_list, 2):
                denom = len(a | b)
                jaccs.append((len(a & b) / denom) if denom > 0 else 1.0)
            mean_jacc = sum(jaccs) / len(jaccs)
        else:
            mean_jacc = 1.0

        stats[L] = {
            "n_prompts": len(sets_list),
            "unique_set_count": len(unique_sets),
            "is_constant": is_constant,
            "intersection": sorted(inter),
            "union": sorted(uni),
            "freq": dict(freq),
            "mean_jaccard": mean_jacc,
        }
    return stats

def print_tally_line(layer_idx, stat, label, topn=16, verbose_lists=False):
    freq_sorted = sorted(stat["freq"].items(), key=lambda x: (-x[1], x[0]))[:topn]
    top_str = ", ".join([f"E{e}:{c}/{stat['n_prompts']}" for e, c in freq_sorted]) if stat["n_prompts"] else "-"
    core_sz = len(stat["intersection"])
    union_sz = len(stat["union"])
    print(f"{label}L{layer_idx:02d} | sets:{stat['unique_set_count']:>2} | "
          f"constant:{str(stat['is_constant']):<5} | meanJ:{stat['mean_jaccard']:.2f} | "
          f"core:{core_sz:>2} | union:{union_sz:>2} | top: {top_str}")
    if verbose_lists and stat["n_prompts"]:
        print(f"    intersection={stat['intersection']}")
        print(f"    union       ={stat['union']}")

def summarize_stability(all_stats):
    # Global layer-averaged stability
    jaccs = [st["mean_jaccard"] for st in all_stats.values() if st["n_prompts"] > 1]
    if not jaccs:
        return {"mean": 0.0, "median": 0.0, "layers": 0}
    jaccs_sorted = sorted(jaccs)
    m = sum(jaccs) / len(jaccs)
    mid = len(jaccs_sorted) // 2
    if len(jaccs_sorted) % 2 == 1:
        med = jaccs_sorted[mid]
    else:
        med = 0.5 * (jaccs_sorted[mid-1] + jaccs_sorted[mid])
    return {"mean": m, "median": med, "layers": len(jaccs)}

def analyze_task_utilization(
    model_name,
    num_layers,
    num_experts_per_layer,
    top_k,
    total_experts_in_model,
    pos_counts, neg_counts,
    pos_predictions, neg_predictions,
    pos_per_prompt, neg_per_prompt,
    *,
    print_unused=False,
    verbose_lists=False,
    per_prompt=False,
    per_prompt_examples=3,
    topn_tally=16
):
    print("\n" + "=" * 70)
    print("TASK-SPECIFIC EXPERT UTILIZATION ANALYSIS")
    print("=" * 70)

    # Totals (unique experts hit)
    utilized_experts = set()
    for layer_idx, layer_counts in pos_counts.items():
        for expert_idx in layer_counts.keys():
            utilized_experts.add((layer_idx, expert_idx))
    for layer_idx, layer_counts in neg_counts.items():
        for expert_idx in layer_counts.keys():
            utilized_experts.add((layer_idx, expert_idx))

    num_utilized_experts = len(utilized_experts)
    utilization_percentage = (num_utilized_experts / total_experts_in_model) * 100
    num_unused_experts = total_experts_in_model - num_utilized_experts
    prunable_percentage = 100 - utilization_percentage

    print(f"\nModel: {model_name}")
    print(f"Layers: {num_layers} | Experts/layer: {num_experts_per_layer} | Top-K: {top_k}")
    print(f"Total experts in model: {total_experts_in_model:,}")
    print(f"Total experts utilized: {num_utilized_experts:,} ({utilization_percentage:.2f}%)")
    print(f"Total experts UNUSED : {num_unused_experts:,} ({prunable_percentage:.2f}%)")

    # Layer-wise density
    print(f"\nLayer-wise Utilization Density (unique experts used / {num_experts_per_layer}):")
    print("-" * 70)
    layer_utilization = defaultdict(int)
    for L, E in utilized_experts:
        layer_utilization[L] += 1

    def get_bar(count, total=num_experts_per_layer, width=50):
        if total == 0: return '░' * width
        filled = int(width * count / total)
        return '█' * filled + '░' * (width - filled)

    for L in range(num_layers):
        count = layer_utilization.get(L, 0)
        pct = (count / num_experts_per_layer * 100) if num_experts_per_layer else 0.0
        print(f"L{L:02d}: {count:4d}/{num_experts_per_layer} ({pct:5.1f}%) | {get_bar(count)}")

    # Tally & stability (ALL, POS-only, NEG-only)
    print("\n" + "=" * 40)
    print("PER-LAYER EXPERT TALLY & STABILITY ACROSS PROMPTS")
    print("=" * 40)

    all_per_prompt = pos_per_prompt + neg_per_prompt
    all_stats = compute_layerwise_stats(all_per_prompt, num_layers)
    pos_stats = compute_layerwise_stats(pos_per_prompt, num_layers)
    neg_stats = compute_layerwise_stats(neg_per_prompt, num_layers)

    print("\n--- ALL PROMPTS ---")
    for L in range(num_layers):
        print_tally_line(L, all_stats[L], label="", topn=topn_tally, verbose_lists=verbose_lists)

    if pos_per_prompt:
        print("\n--- POSITIVE ONLY ---")
        for L in range(num_layers):
            print_tally_line(L, pos_stats[L], label="POS ", topn=topn_tally, verbose_lists=verbose_lists)

    if neg_per_prompt:
        print("\n--- NEGATIVE ONLY ---")
        for L in range(num_layers):
            print_tally_line(L, neg_stats[L], label="NEG ", topn=topn_tally, verbose_lists=verbose_lists)

    # POS vs NEG deltas only if both present
    if pos_per_prompt and neg_per_prompt:
        print("\n" + "=" * 40)
        print("POS vs NEG DELTAS (per layer)")
        print("=" * 40)
        for L in range(num_layers):
            pos_union = set(pos_stats[L]["union"])
            neg_union = set(neg_stats[L]["union"])
            common = sorted(pos_union & neg_union)
            pos_only = sorted(pos_union - neg_union)
            neg_only = sorted(neg_union - pos_union)
            print(f"L{L:02d} | common:{len(common):>2} | pos_only:{len(pos_only):>2} | neg_only:{len(neg_only):>2}")
            if verbose_lists:
                print(f"    common   = {common}")
                print(f"    pos_only = {pos_only}")
                print(f"    neg_only = {neg_only}")

    # Optional per-prompt peek
    if per_prompt and pos_per_prompt:
        print("\n--- PER-PROMPT TOP-K (first few POS prompts) ---")
        for i, d in enumerate(pos_per_prompt[:per_prompt_examples]):
            print(f"  Prompt {i}:")
            for L in range(num_layers):
                experts = d.get(L, None)
                if experts is not None:
                    print(f"    L{L:02d}: {experts}")

    if per_prompt and neg_per_prompt:
        print("\n--- PER-PROMPT TOP-K (first few NEG prompts) ---")
        for i, d in enumerate(neg_per_prompt[:per_prompt_examples]):
            print(f"  Prompt {i}:")
            for L in range(num_layers):
                experts = d.get(L, None)
                if experts is not None:
                    print(f"    L{L:02d}: {experts}")

    # Global stability summary
    all_summary = summarize_stability(all_stats)
    print("\n" + "=" * 40)
    print("GLOBAL STABILITY SUMMARY (ALL PROMPTS)")
    print("=" * 40)
    print(f"Layers considered: {all_summary['layers']}")
    print(f"Mean Jaccard across layers : {all_summary['mean']:.3f}")
    print(f"Median Jaccard across layers: {all_summary['median']:.3f}")

    # Optional: massive "unused experts" lists
    if print_unused:
        print(f"\n" + "=" * 40)
        print("UNUSED EXPERTS IDENTIFICATION (Detailed List)")
        print("=" * 40)
        unused_experts_list = []
        used = set(utilized_experts)
        for L in range(num_layers):
            unused_in_layer = []
            for e in range(num_experts_per_layer):
                if (L, e) not in used:
                    unused_in_layer.append(e)
            if unused_in_layer:
                print(f"\nLayer {L:02d}: {len(unused_in_layer)} unused experts.")
                print("  Indices:")
                for i in range(0, len(unused_in_layer), 10):
                    print(f"    {unused_in_layer[i:i+10]}")
                unused_experts_list.extend([(L, idx) for idx in unused_in_layer])
        if not unused_experts_list:
            print("\n✅ All experts were utilized at least once.")
        else:
            print(f"\nTotal unused experts identified: {len(unused_experts_list)}")

    return all_summary

# ---------- Main ----------
def main():
    args = build_args()

    print(f"Loading model: {args.model} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # fallback to float16 if needed
            trust_remote_code=True,
        )
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Config fetch
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_experts_per_layer = getattr(cfg, "num_routed_experts", None)
    top_k = getattr(cfg, "num_experts_per_tok", None)

    if num_layers is None or num_experts_per_layer is None or top_k is None:
        # Sensible defaults for Qwen3 MoE variants; adjust if your model differs
        num_layers = num_layers or 48
        num_experts_per_layer = num_experts_per_layer or 128
        top_k = top_k or 8

    total_experts_in_model = num_layers * num_experts_per_layer

    print("\nModel Configuration:")
    print(f"  - Hidden layers             : {num_layers}")
    print(f"  - Experts per layer         : {num_experts_per_layer}")
    print(f"  - Experts activated per tok : {top_k}")
    print(f"  - TOTAL unique experts      : {total_experts_in_model:,}")
    print()

    # ---- TEST 1: High-overlap prompts ----
    print("=" * 70)
    print("TEST 1: HIGH-OVERLAP PROMPTS (expect HIGH stability)")
    print("=" * 70)
    hi_prompts = make_high_overlap_prompts(args.max_prompts)
    print(f"  Using {len(hi_prompts)} prompts | position='{args.position}'")
    pos_counts, pos_totals, pos_preds, pos_per_prompt = track_single_token_generation(
        model, tokenizer, hi_prompts, position=args.position, top_k=top_k
    )
    # Empty NEG to keep the interface; we don't need a second group for this test
    neg_counts, neg_totals, neg_preds, neg_per_prompt = {}, {}, [], []

    hi_summary = analyze_task_utilization(
        args.model,
        num_layers,
        num_experts_per_layer,
        top_k,
        total_experts_in_model,
        pos_counts, neg_counts,
        pos_preds, neg_preds,
        pos_per_prompt, neg_per_prompt,
        print_unused=args.print_unused,
        verbose_lists=args.verbose_lists,
        per_prompt=args.per_prompt,
        per_prompt_examples=args.per_prompt_examples,
        topn_tally=args.topn_tally,
    )

    # ---- TEST 2: Diverse prompts ----
    print("\n\n" + "=" * 70)
    print("TEST 2: DIVERSE PROMPTS (expect LOWER stability)")
    print("=" * 70)
    dv_prompts = make_diverse_prompts(args.max_prompts)
    print(f"  Using {len(dv_prompts)} prompts | position='{args.position}'")
    pos_counts2, pos_totals2, pos_preds2, pos_per_prompt2 = track_single_token_generation(
        model, tokenizer, dv_prompts, position=args.position, top_k=top_k
    )
    neg_counts2, neg_totals2, neg_preds2, neg_per_prompt2 = {}, {}, [], []

    dv_summary = analyze_task_utilization(
        args.model,
        num_layers,
        num_experts_per_layer,
        top_k,
        total_experts_in_model,
        pos_counts2, neg_counts2,
        pos_preds2, neg_preds2,
        pos_per_prompt2, neg_per_prompt2,
        print_unused=args.print_unused,
        verbose_lists=args.verbose_lists,
        per_prompt=args.per_prompt,
        per_prompt_examples=args.per_prompt_examples,
        topn_tally=args.topn_tally,
    )

    # ---- Quick comparison ----
    print("\n" + "=" * 70)
    print("COMPARISON: High-overlap vs Diverse (global mean Jaccard)")
    print("=" * 70)
    print(f"High-overlap mean Jaccard: {hi_summary['mean']:.3f} (median {hi_summary['median']:.3f}, layers {hi_summary['layers']})")
    print(f"Diverse     mean Jaccard: {dv_summary['mean']:.3f} (median {dv_summary['median']:.3f}, layers {dv_summary['layers']})")
    if hi_summary["mean"] > dv_summary["mean"]:
        print("\n✅ As expected: high-overlap > diverse (stability).")
    else:
        print("\n⚠️ Unexpected: high-overlap ≤ diverse. Check --position or try more prompts.")

if __name__ == "__main__":
    main()
