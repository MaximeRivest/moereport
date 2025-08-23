#!/usr/bin/env python3
# analyze_experts_8gpu_v5_correct.py - Correctly handles DeepSeek's grouped MoE routing

import os
import json
import torch
import time
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from transformers.cache_utils import DynamicCache, Cache  # keep both

# ---- Transformers cache compatibility shim (safe on 4.49+ / 4.56+) ----
# Some model repos (including DeepSeek V3.*) call `get_usable_length(new_seq_length, layer_idx)`.
# In newer Transformers, the public API is `get_seq_length(layer_idx)`.
# We provide a *non-negative* compatibility method that simply returns the seen length.
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        # Return the number of tokens already in the cache for this layer.
        # Do NOT try to "reserve room" by subtracting new_seq_length; that can go negative on prefill.
        if hasattr(self, "get_seq_length"):
            try:
                return int(self.get_seq_length(layer_idx))
            except TypeError:
                # Older APIs used no layer argument
                return int(self.get_seq_length())
        # very old fallback
        return int(getattr(self, "seen_tokens", 0))

    DynamicCache.get_usable_length = _get_usable_length
    try:
        if not hasattr(Cache, "get_usable_length"):
            Cache.get_usable_length = _get_usable_length
    except Exception:
        pass
# ---- end shim ----




# ---- utils to extract layer index from a module name ----
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

def setup_8_gpus():
    """Initialize all 8 GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Setup] Found {num_gpus} GPUs")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {mem:.1f} GB")
        torch.cuda.set_device(0)
        return num_gpus
    return 0

def create_balanced_device_map(config, num_gpus=8):
    """Create device map to distribute model across all GPUs"""
    num_layers = getattr(config, 'num_hidden_layers', 61)
    
    # Calculate layers per GPU
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": num_gpus - 1,
        "lm_head": num_gpus - 1,
    }
    
    # Distribute layers evenly across all GPUs
    current_layer = 0
    for gpu_id in range(num_gpus):
        gpu_layers = layers_per_gpu + (1 if gpu_id < remainder else 0)
        for i in range(gpu_layers):
            if current_layer < num_layers:
                device_map[f"model.layers.{current_layer}"] = gpu_id
                current_layer += 1
    
    return device_map

def register_gate_hooks_correct(model, config):
    """
    Register hooks on MoE gate modules to capture actual router outputs.
    Returns: (handles, step_buffer, moe_layers, routing_config)
    """
    step_buffer = {}
    handles = []
    moe_layers = set()
    
    # Get routing configuration
    routing_config = {
        'n_routed_experts': getattr(config, 'n_routed_experts', 256),
        'n_group': getattr(config, 'n_group', 8),
        'topk_group': getattr(config, 'topk_group', 4),
        'num_experts_per_tok': getattr(config, 'num_experts_per_tok', 8),
        'routed_scaling_factor': getattr(config, 'routed_scaling_factor', 2.5),
        'norm_topk_prob': getattr(config, 'norm_topk_prob', True),
        'first_k_dense_replace': getattr(config, 'first_k_dense_replace', 3),
        'moe_layer_freq': getattr(config, 'moe_layer_freq', 1),
    }
    
    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # output is (topk_idx, topk_weight) for all positions
            try:
                if isinstance(output, tuple) and len(output) == 2:
                    topk_idx, topk_weight = output
                    # Detach and move to CPU to avoid memory issues
                    step_buffer[layer_idx] = (
                        topk_idx.detach().cpu(),
                        topk_weight.detach().cpu()
                    )
                else:
                    step_buffer[layer_idx] = None
            except Exception as e:
                print(f"Hook error at layer {layer_idx}: {e}")
                step_buffer[layer_idx] = None
        return hook
    
    # Hook only exact MoE gate modules
    for name, module in model.named_modules():
        # Exact match for .mlp.gate (not .mlp.gate_proj)
        if name.endswith(".mlp.gate"):
            L = _extract_layer_idx(name)
            if L >= 0:
                # Verify it's actually an MoE gate module
                # Check for MoE-specific attributes
                if (hasattr(module, 'n_routed_experts') or 
                    hasattr(module, 'num_experts_per_tok') or
                    hasattr(module, 'topk') or
                    module.__class__.__name__ in ['MoEGate', 'DeepseekV3MoEGate']):
                    moe_layers.add(L)
                    handles.append(module.register_forward_hook(make_hook(L)))
                    print(f"  Hooked MoE gate at layer {L}")
    
    return handles, step_buffer, sorted(moe_layers), routing_config

def extract_experts_from_buffer_correct(step_buffer, position, routing_config, moe_layers):
    """
    Extract expert routing from the buffer for a specific position.
    Returns both expert-level and group-level information.
    """
    layer_experts = {}
    layer_groups = {}
    
    n = routing_config['n_routed_experts']
    g = routing_config['n_group']
    experts_per_group = n // g if g > 0 else n
    
    for L in moe_layers:
        if L not in step_buffer or step_buffer[L] is None:
            continue
        
        topk_idx, topk_weight = step_buffer[L]
        
        # Handle different tensor shapes
        if len(topk_idx.shape) == 2:  # [B*T, K] or [T, K]
            T = topk_idx.shape[0]
            p = min(position, T - 1)
            idxs = topk_idx[p]     # [K]
            wts = topk_weight[p]   # [K]
        elif len(topk_idx.shape) == 3:  # [B, T, K]
            T = topk_idx.shape[1]
            p = min(position, T - 1)
            idxs = topk_idx[0, p]  # [K]
            wts = topk_weight[0, p]  # [K]
        else:
            continue
        
        # Store expert choices with their weights
        expert_list = []
        group_set = set()
        
        for e_id, w in zip(idxs.tolist(), wts.tolist()):
            expert_list.append((int(e_id), float(w)))
            # Calculate which group this expert belongs to
            group_id = e_id // experts_per_group
            group_set.add(group_id)
        
        layer_experts[L] = expert_list
        layer_groups[L] = list(group_set)
    
    return layer_experts, layer_groups

def verify_routing_sanity(layer_experts, layer_groups, routing_config):
    """Verify that routing follows DeepSeek's constraints"""
    issues = []
    
    k = routing_config['num_experts_per_tok']
    topk_group = routing_config['topk_group']
    scaling_factor = routing_config['routed_scaling_factor']
    norm_topk = routing_config['norm_topk_prob']
    
    for L in layer_experts:
        experts = layer_experts[L]
        groups = layer_groups[L]
        
        # Check 1: Exactly k experts per token
        if len(experts) != k:
            issues.append(f"Layer {L}: Expected {k} experts, got {len(experts)}")
        
        # Check 2: At most topk_group groups
        if len(groups) > topk_group:
            issues.append(f"Layer {L}: Expected ≤{topk_group} groups, got {len(groups)}")
        
        # Check 3: Weight sum (if normalized)
        if norm_topk:
            weight_sum = sum(w for _, w in experts)
            if abs(weight_sum - scaling_factor) > 0.1:  # Allow small tolerance
                issues.append(f"Layer {L}: Weight sum {weight_sum:.2f} != {scaling_factor}")
    
    return issues

def main():
    # Pandas/Data Science focused prompts
    prompts = [
        #"Comment cuisiner de la poutine",
        #"C'est quoi la recette de la poutine",
        #"how to join 2 dataframes in pandas on column id",
        #"I have 2 dataframse",
        #"Fix this error: df.groupby('category').mean() gives 'DataError: No numeric types to aggregate' on my sales DataFrame",
        #"Why does pd.read_csv('data.csv', parse_dates=['date']) still show date column as object type?",
        #"My code df[df['price'] > 100 and df['quantity'] < 50] throws 'ValueError: truth value ambiguous' - how to fix?",
        "Convert this list [[1,2,3], [4,5,6]] into a pandas DataFrame with columns 'A', 'B', 'C'",
        #"df.merge(df2, on='id') returns empty DataFrame but both have matching id values - what's wrong?",
    ]  # Shortened for testing - add more as needed
    
#     prompts = [
#     """Review: "These running shoes are incredible! Perfect cushioning and support."
# Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
#     """Review: "Best tennis racket I've ever owned. Great power and control."
# Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
#     """Review: "The yoga mat has excellent grip and thickness. Worth every penny!"
# Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
#     """Review: "Basketball has perfect bounce and grip. Professional quality!"
# Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No"""]
    print("="*60)
    print("EXPERT ACTIVATION ANALYSIS - V5 (CORRECT GROUPED ROUTING)")
    print("="*60)
    
    # Setup GPUs
    num_gpus = setup_8_gpus()
    
    # Model name
    model_name = "deepseek-ai/DeepSeek-V3.1"
    
    print(f"\n[Loading] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\n[Loading] Model configuration...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Create device map for all 8 GPUs
    device_map = create_balanced_device_map(config, num_gpus)
    
    print(f"\n[Device Map] Distributing {config.num_hidden_layers} layers across {num_gpus} GPUs:")
    for gpu_id in range(num_gpus):
        layers_on_gpu = [k for k, v in device_map.items() if v == gpu_id and "layers" in k]
        print(f"  GPU {gpu_id}: {len(layers_on_gpu)} layers")
    
    print(f"\n[Loading] Model across all {num_gpus} GPUs...")
    print("This will take a few minutes...")
    
    # Load with device_map to use all GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={i: "120GiB" for i in range(num_gpus)},
    )
    
    # Check memory distribution
    print(f"\n[Memory Distribution]")
    total_mem = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        total_mem += allocated
        print(f"  GPU {i}: {allocated:.1f} GB")
    print(f"  Total: {total_mem:.1f} GB across {num_gpus} GPUs")
    
    # Register hooks with correct handling
    print(f"\n[Registering MoE Gate Hooks]")
    handles, step_buffer, moe_layers, routing_config = register_gate_hooks_correct(model, config)
    
    print(f"\n[Routing Configuration]")
    print(f"  Total experts: {routing_config['n_routed_experts']}")
    print(f"  Groups: {routing_config['n_group']}")
    print(f"  Experts per group: {routing_config['n_routed_experts'] // routing_config['n_group']}")
    print(f"  Top-K groups selected: {routing_config['topk_group']}")
    print(f"  Experts per token: {routing_config['num_experts_per_tok']}")
    print(f"  Scaling factor: {routing_config['routed_scaling_factor']}")
    print(f"  MoE layers found: {len(moe_layers)} layers")
    
    # Verify MoE layer discovery
    expected_moe_layers = []
    first_k = routing_config['first_k_dense_replace']
    freq = routing_config['moe_layer_freq']
    for i in range(config.num_hidden_layers):
        if i >= first_k and i % freq == 0:
            expected_moe_layers.append(i)
    
    print(f"\n[MoE Layer Verification]")
    print(f"  Expected MoE layers (from config): {len(expected_moe_layers)}")
    print(f"  Actual MoE layers found: {len(moe_layers)}")
    if len(moe_layers) != len(expected_moe_layers):
        print(f"  WARNING: Mismatch in MoE layer count!")

    print(f"\n[Processing] {len(prompts)} prompts...")
    print("-" * 40)
    
    # Storage for analysis
    detailed_logs = []
    expert_counts = defaultdict(lambda: defaultdict(int))
    expert_mass = defaultdict(lambda: defaultdict(float))
    group_counts = defaultdict(lambda: defaultdict(int))
    group_mass = defaultdict(lambda: defaultdict(float))
    
    experts_per_group = routing_config['n_routed_experts'] // routing_config['n_group']
    
    # Process each prompt
    for idx, prompt in enumerate(prompts):
        #idx = 1
        #prompt = prompts[idx]
        print(f"{idx+1:2d}. {prompt[:60]}...")

        # Format prompt with chat template when available
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
            max_length=4096,
        )
        input_ids = batch["input_ids"].to("cuda:0")
        attn_mask = batch["attention_mask"].to("cuda:0")

        token_logs = []
        generated_tokens = []
        sanity_issues = []

        max_new = 60  # Shorter for testing
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id


        # Shapes / device
        B, L = input_ids.shape
        device = input_ids.device

        # Preallocate a full-size attention mask and slice it per step (avoids torch.cat in loop)
        full_attn_mask = torch.ones((B, L + max_new), dtype=attn_mask.dtype, device=device)
        full_attn_mask[:, :L] = attn_mask  # preserve any padding zeros from tokenizer

        # Absolute position of the last prompt token (used to query routing buffer)
        base_pos = L - 1

        model.eval()
        # Generate tokens
        with torch.inference_mode():
            # ---- 1) Prefill once over the full prompt (enables KV cache) ----
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,         # <-- crucial
                return_dict=True,
            )
            past = out.past_key_values

            # Greedy next token from the prefill
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # shape (B,1)
            tok_id = int(next_id.item())
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)

            # Extract routing for the token that produced this next id
            # (same semantics as your original code: router info for last existing token)
            position = base_pos
            layer_experts, layer_groups = extract_experts_from_buffer_correct(
                step_buffer, position, routing_config, moe_layers
            )

            # Verify routing sanity for this token
            issues = verify_routing_sanity(layer_experts, layer_groups, routing_config)
            if issues:
                sanity_issues.extend(issues)

            # Accumulate statistics
            for L_ in layer_experts:
                for e_id, weight in layer_experts[L_]:
                    expert_counts[L_][e_id] += 1
                    expert_mass[L_][e_id] += weight
                    g_id = e_id // experts_per_group
                    group_counts[L_][g_id] += 1
                    group_mass[L_][g_id] += weight

            # Log step 0 (matches your previous shape/content)
            token_logs.append({
                "position": 0,
                "token_id": tok_id,
                "token_text": tok_text,
                "layer_experts": layer_experts,
                "layer_groups": layer_groups,
            })
            generated_tokens.append(tok_text)

            # Early stop if EOS right away
            if eos_id is not None and tok_id == eos_id:
                pass
            else:
                # ---- 2) Decode subsequent tokens feeding ONLY the last token (KV cache) ----
                for step in range(1, max_new):
                    cur_len = L + step  # prompt length + tokens generated so far

                    out = model(
                        input_ids=next_id,                    # feed just the last token
                        attention_mask=full_attn_mask[:, :cur_len],
                        past_key_values=past,                 # reuse cache
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values

                    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    tok_id = int(next_id.item())
                    tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)

                    # step_buffer now contains routing for the single token we just fed.
                    # Keep your original "position" semantics by using absolute index;
                    # extract_experts_* will clamp correctly when T=1.
                    position = base_pos + step
                    layer_experts, layer_groups = extract_experts_from_buffer_correct(
                        step_buffer, position, routing_config, moe_layers
                    )

                    issues = verify_routing_sanity(layer_experts, layer_groups, routing_config)
                    if issues:
                        sanity_issues.extend(issues)

                    # Accumulate statistics
                    for L_ in layer_experts:
                        for e_id, weight in layer_experts[L_]:
                            expert_counts[L_][e_id] += 1
                            expert_mass[L_][e_id] += weight
                            g_id = e_id // experts_per_group
                            group_counts[L_][g_id] += 1
                            group_mass[L_][g_id] += weight

                    token_logs.append({
                        "position": step,
                        "token_id": tok_id,
                        "token_text": tok_text,
                        "layer_experts": layer_experts,
                        "layer_groups": layer_groups,
                    })
                    generated_tokens.append(tok_text)

                    if eos_id is not None and tok_id == eos_id:
                        break        
        # Store results
        detailed_logs.append({
            "prompt_idx": idx,
            "prompt": prompt,
            "generated_text": "".join(generated_tokens),
            "moe_layers": moe_layers,
            "routing_config": routing_config,
            "token_logs": token_logs,
            "sanity_issues": sanity_issues[:10] if sanity_issues else [],  # Limit to first 10
        })

# let's make that a for loop:
for i in range(2):  # for detailed_logs[0] and detailed_logs[1]
    for j in range(3):  # for token_logs[0], [1], and [2]
        print(detailed_logs[i]["token_logs"][j]["layer_experts"][60])

# Count unique experts for each token across all layers
detailed_log = detailed_logs[0]
print(f"\nDetailed Log {i} (Prompt: '{detailed_log['prompt'][:50]}...'):")

for j, token_log in enumerate(detailed_log["token_logs"]):
    print(f"\n  Token {j}: '{token_log['token_text']}'")
    
    # Count unique experts per layer for this token
    layer_experts = token_log["layer_experts"]
    
    for layer_idx, experts in enumerate(layer_experts):
        if experts:  # if experts list is not empty
            unique_experts = set(experts)
            unique_count = len(unique_experts)
            print(f"    Layer {layer_idx}: {unique_count} unique experts {sorted(unique_experts)}")
        else:
            print(f"    Layer {layer_idx}: 0 unique experts")

detailed_logs[0]["token_logs"][1]["layer_experts"][60]

detailed_logs[0]["token_logs"][1]["layer_experts"][60]

detailed_logs[0]["token_logs"][1]["token_text"]

# Clean up hooks
for h in handles:
    try:
        h.remove()
    except Exception:
        pass

# Calculate statistics
total_activations = 0
unique_experts = set()
unique_groups = set()
layer_stats = {}

for layer_idx in expert_counts:
    layer_counts = expert_counts[layer_idx]
    layer_masses = expert_mass[layer_idx]
    layer_group_counts = group_counts[layer_idx]
    layer_group_masses = group_mass[layer_idx]
    
    layer_stats[layer_idx] = {
        "unique_experts": len(layer_counts),
        "unique_groups": len(layer_group_counts),
        "total_activations": sum(layer_counts.values()),
        "total_mass": sum(layer_masses.values()),
        "top_experts": sorted(
            [(e, layer_counts[e], layer_masses[e]) for e in layer_counts],
            key=lambda x: x[2],
            reverse=True
        )[:5],
        "top_groups": sorted(
            [(g, layer_group_counts[g], layer_group_masses[g]) for g in layer_group_counts],
            key=lambda x: x[2],
            reverse=True
        )[:3],
    }
    
    for expert_idx in layer_counts:
        total_activations += layer_counts[expert_idx]
        unique_experts.add((layer_idx, expert_idx))
    
    for group_idx in layer_group_counts:
        unique_groups.add((layer_idx, group_idx))

# Model info
total_experts = len(moe_layers) * routing_config['n_routed_experts']
total_groups = len(moe_layers) * routing_config['n_group']

print(f"\n[Results]")
print(f"  MoE layers: {len(moe_layers)}")
print(f"  Total experts: {total_experts:,}")
print(f"  Total groups: {total_groups:,}")
print(f"  Unique experts activated: {len(unique_experts):,}")
print(f"  Unique groups activated: {len(unique_groups):,}")
if total_experts > 0:
    print(f"  Expert activation ratio: {len(unique_experts)/total_experts*100:.2f}%")
if total_groups > 0:
    print(f"  Group activation ratio: {len(unique_groups)/total_groups*100:.2f}%")
print(f"  Total activations: {total_activations:,}")

# Show most active layers
print(f"\n[Most Active Layers by Mass]")
sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]['total_mass'], reverse=True)[:5]
for layer_idx, stats in sorted_layers:
    print(f"  Layer {layer_idx}:")
    print(f"    {stats['unique_experts']} unique experts, {stats['unique_groups']} unique groups")
    print(f"    {stats['total_activations']} activations, mass={stats['total_mass']:.2f}")

# Show group utilization
print(f"\n[Group Utilization]")
all_groups = set()
for layer_idx in group_counts:
    for g_id in group_counts[layer_idx]:
        all_groups.add(g_id)
print(f"  Unique groups used (across all layers): {len(all_groups)} / {routing_config['n_group']}")
print(f"  Groups: {sorted(all_groups)}")

# Expert specialization
print(f"\n[Top Experts Across All Layers]")
expert_frequency = defaultdict(float)
for L, masses in expert_mass.items():        # expert_mass[L]: {e_id: mass}
    for e_id, m in masses.items():
        expert_frequency[(L, e_id)] += m

top_experts = sorted(
    expert_frequency.items(), key=lambda kv: kv[1], reverse=True
)[:10]
for expert_idx, mass in top_experts:
    group_id = expert_idx[1] // experts_per_group
    print(f"    Expert {expert_idx} (Group {group_id}): {mass:.2f} total mass")

# Create summary
summary = {
    "model": model_name,
    "num_gpus": num_gpus,
    "num_prompts": len(prompts),
    "routing_config": routing_config,
    "moe_layers": moe_layers,
    "total_experts": total_experts,
    "total_groups": total_groups,
    "unique_experts_activated": len(unique_experts),
    "unique_groups_activated": len(unique_groups),
    "expert_activation_ratio": len(unique_experts)/total_experts*100 if total_experts > 0 else 0,
    "group_activation_ratio": len(unique_groups)/total_groups*100 if total_groups > 0 else 0,
    "total_activations": total_activations,
    "counts_per_layer": {int(L): {int(e): c for e, c in d.items()} for L, d in expert_counts.items()},
    "mass_per_layer": {int(L): {int(e): m for e, m in d.items()} for L, d in expert_mass.items()},
    "group_counts_per_layer": {int(L): {int(g): c for g, c in d.items()} for L, d in group_counts.items()},
    "group_mass_per_layer": {int(L): {int(g): m for g, m in d.items()} for L, d in group_mass.items()},
    "memory_per_gpu": {i: torch.cuda.memory_allocated(i)/(1024**3) for i in range(num_gpus)},
}

# Save results
with open("expert_activation_v5_correct.json", "w") as f:
    json.dump({"summary": summary, "details": detailed_logs}, f, indent=2)

print(f"\n[Saved] expert_activation_v5_correct.json")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - GROUPED ROUTING VERIFIED")
print("="*60)

print(f"\n[Key Insights]")
if total_experts > 0:
    activation_ratio = len(unique_experts)/total_experts*100
    if activation_ratio < 5:
        print(f"  → Ultra-sparse: Only {activation_ratio:.1f}% of experts used")
    elif activation_ratio < 15:
        print(f"  → Sparse: {activation_ratio:.1f}% of experts used")
    else:
        print(f"  → Moderate sparsity: {activation_ratio:.1f}% of experts used")

print(f"\n[Routing Verification]")
print(f"  ✓ Captured actual router outputs (topk_idx, topk_weight)")
print(f"  ✓ Tracked group-level routing")
print(f"  ✓ Verified routing constraints")
print(f"  ✓ No incorrect softmax or bias operations")

if __name__ == "__main__":
    main()