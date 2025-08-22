#!/usr/bin/env python3
# run_8gpu_distributed.py
# Properly distribute DeepSeek-V3.1 across all 8 H200 GPUs using Accelerate

import os
import torch
import json
import time
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from safetensors import safe_open
import glob

def setup_8_gpus():
    """Ensure all 8 GPUs are visible and initialized"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Setup] Found {num_gpus} GPUs")
        
        # Initialize all GPUs
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {mem:.1f} GB")
        
        torch.cuda.set_device(0)
        return num_gpus
    return 0

def get_model_size(model_name):
    """Estimate model size from config"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Get model dimensions
    hidden_size = getattr(config, 'hidden_size', 4096)
    num_layers = getattr(config, 'num_hidden_layers', 32)
    num_experts = getattr(config, 'n_routed_experts', 64)
    intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
    
    # Estimate parameters (rough)
    # Each expert has: gate + up + down projections
    expert_params = 3 * hidden_size * intermediate_size
    total_expert_params = num_layers * num_experts * expert_params
    
    # Non-expert params (embeddings, attention, layer norms)
    non_expert_params = num_layers * (4 * hidden_size * hidden_size + 2 * hidden_size)
    
    total_params = total_expert_params + non_expert_params
    size_gb = (total_params * 2) / (1024**3)  # bf16 = 2 bytes per param
    
    print(f"\n[Model Info]")
    print(f"  Layers: {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Estimated size: {size_gb:.1f} GB")
    
    return size_gb, num_layers

def create_device_map_8gpu(model_name, num_gpus=8):
    """Create a balanced device map for 8 GPUs"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(config, 'num_hidden_layers', 32)
    
    # Calculate layers per GPU
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": num_gpus - 1,
        "lm_head": num_gpus - 1,
    }
    
    # Distribute layers across GPUs
    current_layer = 0
    for gpu_id in range(num_gpus):
        # Add one extra layer to first GPUs if there's a remainder
        gpu_layers = layers_per_gpu + (1 if gpu_id < remainder else 0)
        
        for i in range(gpu_layers):
            if current_layer < num_layers:
                device_map[f"model.layers.{current_layer}"] = gpu_id
                current_layer += 1
    
    print(f"\n[Device Map]")
    for gpu_id in range(num_gpus):
        layers_on_gpu = [k for k, v in device_map.items() if v == gpu_id and "layers" in k]
        print(f"  GPU {gpu_id}: {len(layers_on_gpu)} layers")
    
    return device_map

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run DeepSeek-V3.1 distributed across 8 GPUs")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompts", type=str, nargs="+", 
                       default=["Explain how to efficiently use 8 GPUs for large language model inference."])
    args = parser.parse_args()
    
    print("="*60)
    print("8-GPU DISTRIBUTED INFERENCE")
    print("="*60)
    
    # Setup GPUs
    num_gpus = setup_8_gpus()
    if num_gpus < 8:
        print(f"Warning: Only {num_gpus} GPUs available, expected 8")
    
    # Get model info
    size_gb, num_layers = get_model_size(args.model)
    
    # Load tokenizer
    print("\n[Loading] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create device map
    device_map = create_device_map_8gpu(args.model, num_gpus)
    
    # Load model distributed across GPUs
    print("\n[Loading] Model across 8 GPUs...")
    print("This may take a few minutes...")
    
    # Option 1: Direct loading with device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="/tmp/offload",
            offload_state_dict=True,
            max_memory={i: "100GiB" for i in range(num_gpus)},
        )
        print("[Success] Model loaded across all GPUs!")
    except Exception as e:
        print(f"[Error] Failed to load with device_map: {e}")
        
        # Option 2: Use load_checkpoint_and_dispatch
        print("\n[Fallback] Trying load_checkpoint_and_dispatch...")
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        model = load_checkpoint_and_dispatch(
            model,
            args.model,
            device_map=device_map,
            dtype=torch.bfloat16,
            no_split_module_classes=["DeepSeekDecoderLayer"],
            max_memory={i: "100GiB" for i in range(num_gpus)},
        )
        print("[Success] Model loaded with dispatch!")
    
    model.eval()
    
    # Check memory usage
    print("\n[Memory Usage]")
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"  GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
    
    # Generate
    print("\n[Generating]")
    for idx, prompt in enumerate(args.prompts, 1):
        print(f"\nPrompt {idx}: {prompt[:100]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0")  # Start on GPU 0
        
        # Generate
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_time = time.time() - start_time
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs.shape[1] - input_ids.shape[1]
        
        print(f"\nGenerated ({new_tokens} tokens in {gen_time:.2f}s):")
        print("-" * 40)
        # Only print the generated part (not the prompt)
        generated_only = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(generated_only)
        print("-" * 40)
        print(f"Speed: {new_tokens/gen_time:.1f} tokens/sec")
    
    # Final memory report
    print("\n[Final Memory Report]")
    total_allocated = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        total_allocated += allocated
        print(f"  GPU {i}: {allocated:.1f} GB")
    print(f"  Total: {total_allocated:.1f} GB across {num_gpus} GPUs")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()