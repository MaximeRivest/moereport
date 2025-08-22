#!/bin/bash

# Test script for multi-GPU MOE runtime with H200 GPUs (140GB each)
# Total VRAM: 1.12TB across 8 GPUs

echo "=========================================="
echo "Multi-GPU MOE Runtime Test"
echo "H200 GPUs: 8 x 140GB = 1.12TB Total VRAM"
echo "=========================================="

# Use all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo ""
echo "Testing with massive expert budget..."
echo "Allocating 100GB per GPU for experts (800GB total)"
echo ""

python moe_multi_gpu_fixed.py \
    --model deepseek-ai/DeepSeek-V3.1 \
    --total-gpu-budget 800 \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --top-p 0.9 \
    --prompts \
        "Explain how modern data centers handle exascale computing challenges." \
        "What are the key architectural innovations in H200 GPUs compared to previous generations?" \
        "Describe the role of high-bandwidth memory in large language model inference."

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="