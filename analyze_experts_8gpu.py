#!/usr/bin/env python3
# analyze_experts_8gpu.py - Analyze expert activation using ALL 8 GPUs

import os
import json
import torch
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

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

def hook_expert_activation(model):
    """Add hooks to track expert activation"""
    expert_activations = defaultdict(lambda: defaultdict(int))
    
    def create_hook(layer_idx):
        def hook_fn(module, input, output):
            # Track which experts were selected
            if hasattr(module, 'last_expert_indices'):
                for expert_idx in module.last_expert_indices:
                    expert_activations[layer_idx][int(expert_idx)] += 1
        return hook_fn
    
    # Add hooks to MoE layers
    for name, module in model.named_modules():
        if "mlp" in name and hasattr(module, "experts"):
            # Extract layer index from name
            layer_idx = int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else -1
            if layer_idx >= 0:
                module.register_forward_hook(create_hook(layer_idx))
    
    return expert_activations

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
    print("EXPERT ACTIVATION ANALYSIS - 8 GPU DISTRIBUTED")
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
    
    # Add tracking hooks
    expert_activations = hook_expert_activation(model)
    
    print(f"\n[Processing] {len(prompts)} prompts...")
    print("-" * 40)
    
    # Process each prompt
    all_outputs = []
    for idx, prompt in enumerate(prompts, 1):
        print(f"{idx:2d}. {prompt[:60]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0")
        
        # Generate (short for analysis)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,  # Short for speed
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
            )
        
        all_outputs.append(outputs)
    
    print("-" * 40)
    
    # Analyze expert activation
    print(f"\n[Expert Activation Analysis]")
    
    # Calculate statistics
    total_activations = 0
    unique_experts = set()
    layer_stats = {}
    
    for layer_idx, expert_counts in expert_activations.items():
        layer_stats[layer_idx] = {
            "unique_experts": len(expert_counts),
            "total_activations": sum(expert_counts.values()),
            "top_experts": sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        for expert_idx, count in expert_counts.items():
            total_activations += count
            unique_experts.add((layer_idx, expert_idx))
    
    # Model info
    total_experts = config.num_hidden_layers * getattr(config, 'n_routed_experts', 256)
    
    print(f"\n[Results]")
    print(f"  Total experts in model: {total_experts:,}")
    print(f"  Unique experts activated: {len(unique_experts):,}")
    print(f"  Activation ratio: {len(unique_experts)/total_experts*100:.2f}%")
    print(f"  Total activations: {total_activations:,}")
    
    # Show most active layers
    print(f"\n[Most Active Layers]")
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]['total_activations'], reverse=True)[:5]
    for layer_idx, stats in sorted_layers:
        print(f"  Layer {layer_idx}: {stats['unique_experts']} unique experts, {stats['total_activations']} activations")
    
    # Find expert specialization
    print(f"\n[Expert Specialization]")
    expert_frequency = defaultdict(int)
    for layer_idx, expert_counts in expert_activations.items():
        for expert_idx, count in expert_counts.items():
            expert_frequency[expert_idx] += count
    
    top_experts = sorted(expert_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 most frequently activated experts (across all layers):")
    for expert_idx, count in top_experts:
        print(f"    Expert {expert_idx}: {count} activations")
    
    # Save results
    results = {
        "model": model_name,
        "num_gpus": num_gpus,
        "num_prompts": len(prompts),
        "total_experts": total_experts,
        "unique_experts_activated": len(unique_experts),
        "activation_ratio": len(unique_experts)/total_experts*100,
        "total_activations": total_activations,
        "layer_stats": layer_stats,
        "memory_per_gpu": {i: torch.cuda.memory_allocated(i)/(1024**3) for i in range(num_gpus)}
    }
    
    with open("expert_activation_8gpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Saved] Results to expert_activation_8gpu_results.json")
    
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

if __name__ == "__main__":
    main()