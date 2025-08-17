import json, torch
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# --- Load model (easy & VRAM-friendly) ---
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # Ampere (RTX 3090) handles fp16 well; bf16 works too on recent stacks.
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant,
    device_map="auto",             # spread across your 2 GPUs automatically
    torch_dtype=torch.float16,
)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Ask HF to return router logits from MoE layers:
# (Transformers Qwen3-MoE supports this)
model.config.output_router_logits = True  # :contentReference[oaicite:4]{index=4}

# Useful metadata from the config:
NUM_LAYERS   = int(model.config.num_hidden_layers)
NUM_EXPERTS  = int(model.config.num_experts)             # 128 on this model
TOPK         = int(model.config.num_experts_per_tok)     # 8 on this model

# --- Aggregators (CPU tensors to keep GPU memory clean) ---
counts      = torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.long)      # hit counts per expert
weight_sums = torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.float32)   # sum of mixture weights
steps_seen  = 0

def _last_token_router_probs(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits -> probs and select the *last token* of the current forward.
    Handles shapes (B, S, E), (S, E), or (B*S, E).
    Returns 1D tensor [E].
    """
    if t.dim() == 3:          # (batch, seq, experts)
        v = t[0, -1, :]
    elif t.dim() == 2:        # (seq, experts) or (B*S, experts)
        v = t[-1, :]
    else:
        v = t.view(-1, t.size(-1))[-1, :]
    return torch.softmax(v.float(), dim=-1)

def _accumulate(router_logits: List[Optional[torch.Tensor]]):
    """
    router_logits: list of length NUM_LAYERS. Some entries can be None if
    a layer is configured as dense MLP-only (see HF issue). :contentReference[oaicite:5]{index=5}
    """
    global steps_seen
    with torch.no_grad():
        for li, rl in enumerate(router_logits):
            if rl is None:
                continue
            probs = _last_token_router_probs(rl)
            topw, topi = torch.topk(probs, k=TOPK, dim=-1)
            # Move to CPU and accumulate
            topi_cpu = topi.cpu()
            topw_cpu = topw.cpu()
            counts[li].index_add_(0, topi_cpu, torch.ones_like(topi_cpu, dtype=torch.long))
            weight_sums[li].index_add_(0, topi_cpu, topw_cpu)
        steps_seen += 1

@torch.inference_mode()
def generate_and_trace(prompt: str,
                       max_new_tokens=128,
                       temperature=0.7,
                       top_p=0.95,
                       stop_id: Optional[int]=None) -> str:
    """
    Token-by-token decoding so we can log router decisions each step.
    We call model(...) directly (not .generate) because .generate() does
    not consistently surface router logits across MoE models. :contentReference[oaicite:6]{index=6}
    """
    device = next(model.parameters()).device  # first shard device
    x = tok(prompt, return_tensors="pt").to(device)
    past = None
    ids  = x["input_ids"]

    for _ in range(max_new_tokens):
        out = model(
            input_ids=ids if past is None else ids[:, -1:],
            use_cache=True,
            past_key_values=past,
            output_router_logits=True,   # ensure routers are returned this step
            return_dict=True,
        )
        past = out.past_key_values
        _accumulate(out.router_logits)   # log which experts fired this step

        logits = out.logits[:, -1, :]
        if temperature is not None and temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        if 0 < (top_p or 1.0) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            trimmed = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs * keep)
            probs = trimmed / trimmed.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=-1)

        if stop_id is None:
            stop_id = tok.eos_token_id
        if int(next_id[0]) == stop_id:
            break

    return tok.decode(ids[0], skip_special_tokens=True)

def summarize_report():
    """Produce a compact JSON-able summary."""
    used_mask = counts > 0
    layer_coverage = (used_mask.sum(dim=1).float() / NUM_EXPERTS).tolist()
    never_hit = {int(li): torch.nonzero(~used_mask[li]).squeeze(-1).tolist()
                 for li in range(NUM_LAYERS)}

    # Very simple "overuse" heuristic: > 2× expected uniform hit rate.
    expected_hits = (steps_seen * TOPK) / NUM_EXPERTS
    overused = {}
    for li in range(NUM_LAYERS):
        over_idx = torch.nonzero(counts[li].float() > (2.0 * expected_hits)).squeeze(-1).tolist()
        if over_idx:
            overused[int(li)] = over_idx

    return {
        "model_id": MODEL_ID,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "topk_per_token": TOPK,
        "steps_seen": int(steps_seen),
        "coverage_by_layer": layer_coverage,
        "global_coverage_mean": float(torch.tensor(layer_coverage).mean()),
        "never_hit_by_layer": never_hit,
        "overused_experts_by_layer": overused,
    }

if __name__ == "__main__":
    # --- Put your domain prompts here (examples) ---
    prompts = [
        "You are an expert tax accountant. Explain Section 179 deductions for a small US LLC with examples.",
        "Summarize a 10-K risk section focusing on supply-chain disruptions and FX exposure.",
        "Translate the following medical abstract from English to German and keep terminology precise: ...",
    ]
    for p in prompts:
        print("\n--- PROMPT ---\n", p)
        out = generate_and_trace(p, max_new_tokens=200)
        print("\n--- OUTPUT ---\n", out)

    rep = summarize_report()
    with open("moe_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    torch.save({"counts": counts, "weight_sums": weight_sums}, "moe_raw.pt")
    print("\nSaved moe_report.json and moe_raw.pt")
    print(json.dumps(rep, indent=2))
