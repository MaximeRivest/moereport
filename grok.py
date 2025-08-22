#!/usr/bin/env python3
# analyze_experts_8gpu.py - Analyze expert activation using ALL 8 GPUs

import os
import json
import torch
import time
import numpy as np
import types
from collections import defaultdict
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

class DeepSeekCompatibleCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0

    @property
    def seen_tokens(self):
        return self._seen_tokens

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx == 0:
            if len(self.key_cache) > 0 and self.key_cache[0] is not None:
                self._seen_tokens = self.key_cache[0].shape[-2]
            else:
                self._seen_tokens = key_states.shape[-2]
        return result

def setup_8_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Setup] Found {num_gpus} GPUs")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f" GPU {i}: {mem:.1f} GB")
        torch.cuda.set_device(0)
        return num_gpus
    return 0

def create_balanced_device_map(config, num_gpus=8):
    num_layers = config.num_hidden_layers
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": num_gpus - 1,
        "lm_head": num_gpus - 1,
    }
    current_layer = 0
    for gpu_id in range(num_gpus):
        gpu_layers = layers_per_gpu + (1 if gpu_id < remainder else 0)
        for i in range(gpu_layers):
            if current_layer < num_layers:
                device_map[f"model.layers.{current_layer}"] = gpu_id
                current_layer += 1
    return device_map

def main():
    prompts = [
        "Fix this error: df.groupby('category').mean() gives 'DataError: No numeric types to aggregate' on my sales DataFrame",
        "Why does pd.read_csv('data.csv', parse_dates=['date']) still show date column as object type?",
        # "My code df[df['price'] > 100 and df['quantity'] < 50] throws 'ValueError: truth value ambiguous' - how to fix?",
        # "Convert this list [[1,2,3], [4,5,6]] into a pandas DataFrame with columns 'A', 'B', 'C'",
        # "df.merge(df2, on='id') returns empty DataFrame but both have matching id values - what's wrong?",
        # "Replace all zeros with NaN in columns ['sales', 'profit', 'cost'] of my DataFrame",
        # "Why does df.loc[df['date'] == '2024-01-15'] return no rows when I can see that date in df.head()?",
        # "Calculate percentage of missing values for each column in iris_dataset.csv",
        # "df['new_col'] = df.apply(lambda x: x['A'] * x['B']) gives KeyError - correct syntax?",
        # "Split 'full_name' column into 'first_name' and 'last_name' columns for this data: ['John Smith', 'Jane Doe']",
        # "ValueError when trying df.pivot(index='date', columns='product', values='sales') - says duplicate entries",
        # "Add row totals and column totals to this crosstab result: pd.crosstab(df['region'], df['category'])",
        # "Memory error loading 5GB CSV file with pd.read_csv() - need chunking solution",
        # "df.to_excel('output.xlsx', index=False) saves but Excel says file is corrupted",
        # "Combine these 3 DataFrames: df1 has columns [id, name], df2 has [id, age], df3 has [id, city]"
    ]
   
    print("="*60)
    print("EXPERT ACTIVATION ANALYSIS - 8 GPU DISTRIBUTED")
    print("="*60)
   
    num_gpus = setup_8_gpus()
   
    model_name = "deepseek-ai/DeepSeek-V3.1"
   
    print(f"\n[Loading] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
   
    print(f"\n[Loading] Model configuration...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
   
    device_map = create_balanced_device_map(config, num_gpus)
   
    print(f"\n[Device Map] Distributing {config.num_hidden_layers} layers across {num_gpus} GPUs:")
    for gpu_id in range(num_gpus):
        layers_on_gpu = [k for k, v in device_map.items() if v == gpu_id and "layers" in k]
        print(f" GPU {gpu_id}: {len(layers_on_gpu)} layers")
   
    print(f"\n[Loading] Model across all {num_gpus} GPUs...")
    print("This will take a few minutes...")
   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={i: "120GiB" for i in range(num_gpus)},
    )
    model.eval()
   
    print(f"\n[Memory Distribution]")
    total_mem = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        total_mem += allocated
        print(f" GPU {i}: {allocated:.1f} GB")
    print(f" Total: {total_mem:.1f} GB across {num_gpus} GPUs")
   
    # Patch MoE MLPs to capture router logits
    def patched_mlp_forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        self._current_router_logits = router_logits
        return self._original_forward(hidden_states)

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            layer.mlp._original_forward = layer.mlp.forward
            layer.mlp.forward = types.MethodType(patched_mlp_forward, layer.mlp)
   
    print(f"\n[Processing] {len(prompts)} prompts...")
    print("-" * 40)
   
    all_logs = []
    max_new_tokens = 50
    num_experts_per_tok = config.num_experts_per_tok
   
    for idx, prompt in enumerate(prompts, 1):
        print(f"{idx:2d}. {prompt[:60]}...")
       
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        generated_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        generated_text = ""
        token_activations = []
       
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    generated_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False
                )
           
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
           
            next_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            generated_text += next_token_text
           
            # Collect per-layer expert activations for the last position
            this_token = {}
            for layer_idx, layer in enumerate(model.model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_current_router_logits'):
                    router_logits = layer.mlp._current_router_logits
                    if isinstance(router_logits, tuple):
                        router_logits = router_logits[0]  # Assume first element is logits tensor; try [1] if error persists
                    r_logits = router_logits[0, -1]
                    probs = torch.softmax(r_logits.to("cpu"), dim=-1)
                    top_weights, top_indices = torch.topk(probs, num_experts_per_tok)
                    this_token[layer_idx] = list(zip(top_indices.tolist(), top_weights.tolist()))
           
            token_activations.append(this_token)
           
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
           
            if next_token_id[0].item() == tokenizer.eos_token_id:
                break
       
        all_logs.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "token_activations": token_activations
        })
   
    print("-" * 40)
   
    with open("expert_activation_8gpu_detailed.json", "w") as f:
        json.dump(all_logs, f, indent=2)
    print("\n[Saved] Detailed per-token activations to expert_activation_8gpu_detailed.json")

if __name__ == "__main__":
    main()