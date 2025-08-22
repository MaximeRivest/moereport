python moe_on_demand_runtime.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --weights Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --load-strategy empty_selective \
  --gpu-expert-budget-gb 15 \
  --prompts "Summarize Basel III capital requirements" "Sketch a compactness proof"