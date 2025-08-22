#!/usr/bin/env python3
# analyze_experts_8gpu_v2.py - Analyze expert activation using ALL 8 GPUs (Fixed for DeepSeek-V3.1)

import os
import json
import torch
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class DeepSeekCompatibleCache(DynamicCache):
    """Adds the seen_tokens property expected by DeepSeek-V3.1."""
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0

    @property
    def seen_tokens(self):
        # Fallback to current cached sequence length if we haven't computed it yet
        return getattr(self, "_seen_tokens", self.get_seq_length())

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Let base class update caches
        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        # Track length from layer 0 cache if available
        try:
            if len(self.key_cache) > 0 and self.key_cache[0] is not None:
                self._seen_tokens = self.key_cache[0].shape[-2]
        except Exception:
            pass
        return out

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

def enable_router_logits(model):
    """Ensure router logits are produced"""
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True

def format_for_chat(tokenizer, user_text: str) -> str:
    """Format user text for chat if chat template is available"""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_text

def detect_moe_layers(router_logits) -> List[int]:
    """Indices of layers that actually emit router logits."""
    moe_layers = []
    for i, x in enumerate(router_logits):
        if x is not None:
            moe_layers.append(i)
    return moe_layers

def extract_layer_topk_at_pos(router_logits, pos: int, top_k: int) -> Dict[int, List[tuple]]:
    """
    router_logits: list with one tensor per layer (None for dense layers).
    Returns {layer_idx: [(expert_id, prob), ...top_k]}
    """
    layer_experts = {}
    for layer_idx, layer_logits in enumerate(router_logits):
        if layer_logits is None:
            continue
        # Shapes vary across models. Handle [B, T, E] and [T, E].
        if layer_logits.dim() == 3:      # [batch, time, experts]
            t = min(pos, layer_logits.shape[1] - 1)
            pos_logits = layer_logits[0, t]
        elif layer_logits.dim() == 2:    # [time, experts]
            t = min(pos, layer_logits.shape[0] - 1)
            pos_logits = layer_logits[t]
        else:
            continue
        probs = torch.softmax(pos_logits.to(torch.float32), dim=-1)
        vals, idx = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        layer_experts[layer_idx] = [(int(i), float(v)) for i, v in zip(idx, vals)]
    return layer_experts

def main():
    # Pandas/Data Science focused prompts
    prompts = [
        "Fix this error: df.groupby('category').mean() gives 'DataError: No numeric types to aggregate' on my sales DataFrame",
        "Why does pd.read_csv('data.csv', parse_dates=['date']) still show date column as object type?",
        "My code df[df['price'] > 100 and df['quantity'] < 50] throws 'ValueError: truth value ambiguous' - how to fix?",
        "Convert this list [[1,2,3], [4,5,6]] into a pandas DataFrame with columns 'A', 'B', 'C'",
        "df.merge(df2, on='id') returns empty DataFrame but both have matching id values - what's wrong?",
        "Replace all zeros with NaN in columns ['sales', 'profit', 'cost'] of my DataFrame",
        "Why does df.loc[df['date'] == '2024-01-15'] return no rows when I can see that date in df.head()?",
        "Calculate percentage of missing values for each column in iris_dataset.csv",
        "df['new_col'] = df.apply(lambda x: x['A'] * x['B']) gives KeyError - correct syntax?",
        "Split 'full_name' column into 'first_name' and 'last_name' columns for this data: ['John Smith', 'Jane Doe']",
        "ValueError when trying df.pivot(index='date', columns='product', values='sales') - says duplicate entries",
        "Add row totals and column totals to this crosstab result: pd.crosstab(df['region'], df['category'])",
        "Memory error loading 5GB CSV file with pd.read_csv() - need chunking solution",
        "df.to_excel('output.xlsx', index=False) saves but Excel says file is corrupted",
        "Combine these 3 DataFrames: df1 has columns [id, name], df2 has [id, age], df3 has [id, city]"
    ]
    
    print("="*60)
    print("EXPERT ACTIVATION ANALYSIS - 8 GPU DISTRIBUTED (V2)")
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
        max_memory={i: "120GiB" for i in range(num_gpus)},  # Use most of each GPU
    )
    model.eval()
    
    # Check memory distribution
    print(f"\n[Memory Distribution]")
    total_mem = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        total_mem += allocated
        print(f"  GPU {i}: {allocated:.1f} GB")
    print(f"  Total: {total_mem:.1f} GB across {num_gpus} GPUs")
    
    # Enable router logits
    enable_router_logits(model)
    
    # Get top-k configuration
    top_k = (
        getattr(config, "num_experts_per_tok", None)
        or getattr(config, "moe_top_k", None)
        or getattr(config, "top_k", 8)
    )
    
    print(f"\n[Configuration]")
    print(f"  Top-K experts per token: {top_k}")
    print(f"  Total experts per layer: {getattr(config, 'n_routed_experts', 256)}")
    
    print(f"\n[Processing] {len(prompts)} prompts...")
    print("-" * 40)
    
    # Storage for detailed logs
    detailed_logs = []
    
    # Process each prompt with token-by-token generation
    for idx, prompt in enumerate(prompts):
        print(f"{idx+1:2d}. {prompt[:60]}...")
        
        # Format prompt for chat if possible
        formatted = format_for_chat(tokenizer, prompt)
        
        # Tokenize with attention mask
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
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        
        # 1) Probe once to learn which layers are MoE
        with torch.no_grad():
            out0 = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_router_logits=True,
                return_dict=True,
                use_cache=False,   # keep it simple & compatible
            )
        moe_layers = detect_moe_layers(out0.router_logits)
        
        # 2) Greedy decode max_new_tokens, logging router top-k each step
        max_new = 50
        cur_ids = input_ids
        cur_mask = attn_mask
        
        for step in range(max_new):
            with torch.no_grad():
                outputs = model(
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    output_router_logits=True,
                    return_dict=True,
                    use_cache=False,  # avoids cache ABI issues reliably
                )
                next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)  # [1,1]
                tok_id = int(next_id.item())
                tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
                
                # Position is current last index
                pos = cur_ids.shape[1] - 1
                layer_experts = extract_layer_topk_at_pos(outputs.router_logits, pos, top_k)
                
                token_logs.append({
                    "position": step,
                    "token_id": tok_id,
                    "token_text": tok_text,
                    "layer_experts": {int(k): v for k, v in layer_experts.items()},
                })
                generated_tokens.append(tok_text)
                
                if eos_id is not None and tok_id == eos_id:
                    break
                
                # Append for next step
                cur_ids  = torch.cat([cur_ids, next_id.to(cur_ids.device)], dim=1)
                cur_mask = torch.cat([cur_mask, torch.ones_like(next_id).to(cur_mask.device)], dim=1)
        
        # Save per-prompt detailed record
        detailed_logs.append({
            "prompt_idx": idx,
            "prompt": prompt,
            "generated_text": "".join(generated_tokens),
            "moe_layers": moe_layers,
            "top_k": top_k,
            "token_logs": token_logs,
        })
    
    print("-" * 40)
    
    # Aggregate simple counts/mass per (layer, expert)
    expert_counts = defaultdict(lambda: defaultdict(int))
    expert_mass   = defaultdict(lambda: defaultdict(float))
    
    for rec in detailed_logs:
        for tl in rec["token_logs"]:
            for layer_idx, pairs in tl["layer_experts"].items():
                for expert_id, prob in pairs:
                    expert_counts[layer_idx][expert_id] += 1
                    expert_mass[layer_idx][expert_id]   += prob
    
    # Calculate statistics
    total_activations = 0
    unique_experts = set()
    layer_stats = {}
    
    for layer_idx in expert_counts:
        layer_counts = expert_counts[layer_idx]
        layer_masses = expert_mass[layer_idx]
        
        layer_stats[layer_idx] = {
            "unique_experts": len(layer_counts),
            "total_activations": sum(layer_counts.values()),
            "total_mass": sum(layer_masses.values()),
            "top_experts": sorted(
                [(e, layer_counts[e], layer_masses[e]) for e in layer_counts],
                key=lambda x: x[2],  # Sort by probability mass
                reverse=True
            )[:5]
        }
        
        for expert_idx in layer_counts:
            total_activations += layer_counts[expert_idx]
            unique_experts.add((layer_idx, expert_idx))
    
    # Model info
    total_experts = len(moe_layers) * getattr(config, 'n_routed_experts', 256)
    
    print(f"\n[Results]")
    print(f"  MoE layers identified: {len(moe_layers)}")
    print(f"  Total experts in MoE layers: {total_experts:,}")
    print(f"  Unique experts activated: {len(unique_experts):,}")
    print(f"  Activation ratio: {len(unique_experts)/total_experts*100:.2f}%")
    print(f"  Total activations: {total_activations:,}")
    
    # Show most active layers
    print(f"\n[Most Active Layers]")
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]['total_mass'], reverse=True)[:5]
    for layer_idx, stats in sorted_layers:
        print(f"  Layer {layer_idx}: {stats['unique_experts']} unique experts, "
              f"{stats['total_activations']} activations, "
              f"mass={stats['total_mass']:.2f}")
    
    # Find expert specialization
    print(f"\n[Expert Specialization]")
    expert_frequency = defaultdict(float)
    for layer_idx in expert_mass:
        for expert_idx, mass in expert_mass[layer_idx].items():
            expert_frequency[expert_idx] += mass
    
    top_experts = sorted(expert_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 most frequently activated experts (by probability mass across all layers):")
    for expert_idx, mass in top_experts:
        print(f"    Expert {expert_idx}: {mass:.2f} total probability mass")
    
    # Create summary
    summary = {
        "model": model_name,
        "num_gpus": num_gpus,
        "num_prompts": len(prompts),
        "moe_layers": sorted(set(l for rec in detailed_logs for l in rec["moe_layers"])),
        "top_k": top_k,
        "total_experts": total_experts,
        "unique_experts_activated": len(unique_experts),
        "activation_ratio": len(unique_experts)/total_experts*100,
        "total_activations": total_activations,
        "counts_per_layer": {int(L): {int(e): c for e, c in d.items()} for L, d in expert_counts.items()},
        "prob_mass_per_layer": {int(L): {int(e): m for e, m in d.items()} for L, d in expert_mass.items()},
        "memory_per_gpu": {i: torch.cuda.memory_allocated(i)/(1024**3) for i in range(num_gpus)}
    }
    
    # Save results
    with open("expert_activation_detailed.json", "w") as f:
        json.dump({"summary": summary, "details": detailed_logs}, f, indent=2)
    
    print(f"\n[Saved] expert_activation_detailed.json with per-token, per-layer expert routing.")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - ALL 8 GPUs UTILIZED")
    print("="*60)
    
    print(f"\n[Key Insights]")
    activation_ratio = len(unique_experts)/total_experts*100
    if activation_ratio < 5:
        print(f"  → Ultra-sparse: Only {activation_ratio:.1f}% of experts used")
        print(f"  → Pandas/data science queries activate a very specific expert subset")
        print(f"  → High degree of specialization in the model")
    elif activation_ratio < 15:
        print(f"  → Sparse: {activation_ratio:.1f}% of experts used")
        print(f"  → Good specialization for data science domain")
    else:
        print(f"  → Moderate sparsity: {activation_ratio:.1f}% of experts used")
        print(f"  → These queries engage diverse knowledge areas")
    
    print(f"\n[Router Logits Successfully Captured]")
    print(f"  → Token-by-token expert routing logged")
    print(f"  → Probability distributions recorded for analysis")
    print(f"  → Full details in expert_activation_detailed.json")

if __name__ == "__main__":
    main()