#!/usr/bin/env python3
# quick_expert_test.py - Quick test to see expert activation patterns

import torch
import json
from moe_on_demand_runtime import OnDemandMoERuntime, RuntimeConfig

# Quick test with just a few prompts
test_prompts = [
    "Fix pandas error: df.groupby('category').mean() gives 'No numeric types to aggregate'",
    "How to handle ValueError: truth value ambiguous in df[df['price'] > 100 and df['quantity'] < 50]",
    "Convert list [[1,2,3], [4,5,6]] to pandas DataFrame with columns A, B, C"
]

print("="*60)
print("QUICK EXPERT ACTIVATION TEST")
print("="*60)

# Minimal config for faster testing
cfg = RuntimeConfig(
    model_name="deepseek-ai/DeepSeek-V3.1",
    dtype="bfloat16",
    gpu_expert_budget_gb=30.0,  # Smaller budget for quick test
    max_new_tokens=50,  # Very short responses
    temperature=0.0,
    load_strategy="empty_selective",
    allow_cpu_fallback_for_disk=True
)

print("\n[Loading] Model...")
runtime = OnDemandMoERuntime(cfg)

print(f"\n[Model Stats]")
print(f"  Total experts: {runtime.report.total_experts_wrapped}")
print(f"  MoE layers: {len(runtime.report.moe_layers)}")

print(f"\n[Testing] {len(test_prompts)} prompts...")
report = runtime.generate(test_prompts)

# Analyze expert usage
print("\n[Expert Activation Summary]")
total_activations = 0
unique_experts = set()
layer_summary = {}

for layer_idx, expert_counts in runtime.stats.usage_counts.items():
    layer_summary[layer_idx] = len(expert_counts)
    for expert_idx, count in expert_counts.items():
        total_activations += count
        unique_experts.add((layer_idx, expert_idx))

print(f"  Total activations: {total_activations}")
print(f"  Unique experts used: {len(unique_experts)}")
print(f"  Experts/Total ratio: {len(unique_experts)/runtime.report.total_experts_wrapped*100:.2f}%")

# Show top layers by activity
print("\n[Most Active Layers]")
sorted_layers = sorted(layer_summary.items(), key=lambda x: x[1], reverse=True)[:5]
for layer_idx, num_experts in sorted_layers:
    print(f"  Layer {layer_idx}: {num_experts} unique experts")

# Save quick results
with open("quick_expert_test_results.json", "w") as f:
    json.dump({
        "total_experts": runtime.report.total_experts_wrapped,
        "unique_experts_used": len(unique_experts),
        "percent_used": len(unique_experts)/runtime.report.total_experts_wrapped*100,
        "total_activations": total_activations,
        "cache_stats": runtime.stats.summary()
    }, f, indent=2)

print("\n[Complete] Results saved to quick_expert_test_results.json")
print("="*60)