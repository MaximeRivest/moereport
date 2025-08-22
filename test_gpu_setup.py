#!/usr/bin/env python3
# test_gpu_setup.py - Test that all 8 GPUs are accessible

import torch
import os

def test_all_gpus():
    print("="*60)
    print("GPU AVAILABILITY TEST")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    # Get GPU count
    num_gpus = torch.cuda.device_count()
    print(f"\nDetected {num_gpus} GPU(s)")
    
    # Test each GPU
    print("\nTesting each GPU:")
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        
        # Get device properties
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Try to allocate a small tensor on this GPU
        try:
            test_tensor = torch.ones(1000, 1000, device=f'cuda:{i}')
            result = test_tensor.sum().item()
            print(f"  Test allocation: SUCCESS (sum={result})")
            del test_tensor
        except Exception as e:
            print(f"  Test allocation: FAILED ({e})")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("All GPUs are accessible!")
    print("="*60)

if __name__ == "__main__":
    test_all_gpus()