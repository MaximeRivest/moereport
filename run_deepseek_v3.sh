#!/bin/bash

# Run DeepSeek-V3.1 with massive expert cache on single H200 GPU (140GB VRAM)

echo "=========================================="
echo "DeepSeek-V3.1 with Large Expert Cache"
echo "Using single H200 GPU with 140GB VRAM"
echo "=========================================="

# Use single GPU to avoid device mismatch issues
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Running with 100GB expert cache budget..."
echo ""

python moe_on_demand_runtime.py \
    --model deepseek-ai/DeepSeek-V3.1 \
    --gpu-expert-budget-gb 100 \
    --load-strategy empty_selective \
    --allow-cpu-fallback \
    --max-new 256 \
    --temp 0.7 \
    --prompts \
        "Explain the architecture of mixture of experts models and how they achieve efficiency through sparse activation." \
        "What are the advantages of using H200 GPUs with 140GB HBM3e memory for large language model inference?" \
        "Compare tensor parallelism, pipeline parallelism, and data parallelism strategies for distributed training."

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="