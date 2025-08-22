#!/bin/bash

# Test script for tensor parallel MOE runtime with 8 GPUs

echo "=========================================="
echo "Testing Tensor Parallel MOE Runtime"
echo "=========================================="

# Set environment for all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Test with DeepSeek-V3.1 model
echo ""
echo "Running with DeepSeek-V3.1 across 8 GPUs..."
echo ""

python moe_tensor_parallel.py \
    --model deepseek-ai/DeepSeek-V3.1 \
    --gpu-budget-per-gpu 18 \
    --max-new-tokens 100 \
    --temperature 0.7 \
    --top-p 0.9 \
    --repetition-penalty 1.1 \
    --prompts \
        "What are the key differences between tensor parallelism and data parallelism?" \
        "Explain how mixture of experts models work." \
        "What are the advantages of using multiple GPUs for large language models?"

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="