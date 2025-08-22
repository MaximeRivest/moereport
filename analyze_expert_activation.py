#!/usr/bin/env python3
# analyze_expert_activation.py
# Analyze which experts are activated for pandas/data science prompts in DeepSeek-V3.1

import os
import json
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Import the MOE runtime
from moe_on_demand_runtime import (
    OnDemandMoERuntime, 
    RuntimeConfig,
    human_bytes
)

@dataclass
class ExpertActivationAnalysis:
    """Store and analyze expert activation patterns"""
    prompts: List[str]
    layer_expert_counts: Dict[int, Dict[int, int]]
    total_experts: int
    total_layers: int
    
    def get_activation_matrix(self):
        """Create matrix of expert activations across layers"""
        # Create a matrix: layers x experts
        max_experts = max(max(experts.keys()) for experts in self.layer_expert_counts.values() if experts) + 1
        matrix = np.zeros((self.total_layers, min(max_experts, 256)))  # Cap at 256 for visualization
        
        for layer_idx, expert_counts in self.layer_expert_counts.items():
            for expert_idx, count in expert_counts.items():
                if expert_idx < 256:  # Cap for visualization
                    matrix[layer_idx, expert_idx] = count
        
        return matrix
    
    def get_activation_stats(self):
        """Calculate statistics about expert activation"""
        total_activations = 0
        unique_experts_per_layer = {}
        expert_usage_distribution = defaultdict(int)
        
        for layer_idx, expert_counts in self.layer_expert_counts.items():
            unique_experts_per_layer[layer_idx] = len(expert_counts)
            for expert_idx, count in expert_counts.items():
                total_activations += count
                expert_usage_distribution[expert_idx] += count
        
        # Calculate what percentage of total experts were used
        total_unique_experts = len(set(
            expert_idx 
            for experts in self.layer_expert_counts.values() 
            for expert_idx in experts.keys()
        ))
        
        # Get top experts
        top_experts = sorted(
            expert_usage_distribution.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        return {
            "total_activations": total_activations,
            "total_unique_experts_used": total_unique_experts,
            "percent_experts_used": (total_unique_experts / self.total_experts) * 100 if self.total_experts > 0 else 0,
            "avg_experts_per_layer": np.mean(list(unique_experts_per_layer.values())),
            "top_20_experts": top_experts,
            "unique_experts_per_layer": unique_experts_per_layer
        }
    
    def plot_activation_heatmap(self, output_path="expert_activation_heatmap.png"):
        """Create a heatmap of expert activations"""
        matrix = self.get_activation_matrix()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
        
        # Main heatmap
        sns.heatmap(
            matrix[:, :64],  # Show first 64 experts for clarity
            ax=ax1,
            cmap='YlOrRd',
            cbar_kws={'label': 'Activation Count'},
            xticklabels=5,
            yticklabels=5
        )
        ax1.set_xlabel('Expert Index (showing first 64 of 256)')
        ax1.set_ylabel('Layer Index')
        ax1.set_title('Expert Activation Heatmap for Pandas/Data Science Prompts')
        
        # Summary bar chart - experts per layer
        unique_per_layer = [len(self.layer_expert_counts.get(i, {})) for i in range(self.total_layers)]
        ax2.bar(range(self.total_layers), unique_per_layer, color='steelblue')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Unique Experts')
        ax2.set_title('Number of Unique Experts Activated per Layer')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] Activation heatmap to {output_path}")
        return output_path

def analyze_expert_activation(prompts: List[str]):
    """Run DeepSeek-V3.1 on prompts and analyze expert activation"""
    
    print("="*60)
    print("EXPERT ACTIVATION ANALYSIS")
    print("="*60)
    
    # Configure runtime with reasonable settings for analysis
    cfg = RuntimeConfig(
        model_name="deepseek-ai/DeepSeek-V3.1",
        weights_location=None,
        dtype="bfloat16",
        gpu_expert_budget_gb=50.0,  # 50GB for expert caching
        prefer_gpu_when_possible=True,
        allow_cpu_fallback_for_disk=True,
        max_new_tokens=100,  # Shorter for analysis
        temperature=0.0,  # Deterministic
        system_prompt="You are a helpful assistant specializing in pandas and data science.",
        load_strategy="empty_selective"
    )
    
    print("\n[Initializing] DeepSeek-V3.1 runtime...")
    print(f"  Expert cache budget: {cfg.gpu_expert_budget_gb} GB")
    print(f"  Number of prompts: {len(prompts)}")
    
    # Initialize runtime
    runtime = OnDemandMoERuntime(cfg)
    
    # Get model info
    total_experts = runtime.report.total_experts_wrapped
    total_layers = len(runtime.report.moe_layers)
    print(f"\n[Model Info]")
    print(f"  Total experts: {total_experts}")
    print(f"  Total MoE layers: {total_layers}")
    print(f"  Experts per layer: {total_experts // total_layers if total_layers > 0 else 0}")
    
    # Run generation on all prompts
    print(f"\n[Processing] Running {len(prompts)} prompts...")
    print("-" * 40)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt[:60]}...")
    
    print("-" * 40)
    print("\n[Generating] This may take a few minutes...")
    
    # Generate responses
    report = runtime.generate(prompts)
    
    # Extract expert usage data
    layer_expert_counts = runtime.stats.usage_counts
    
    # Create analysis object
    analysis = ExpertActivationAnalysis(
        prompts=prompts,
        layer_expert_counts=dict(layer_expert_counts),
        total_experts=total_experts,
        total_layers=total_layers
    )
    
    # Get statistics
    stats = analysis.get_activation_stats()
    
    print("\n" + "="*60)
    print("EXPERT ACTIVATION STATISTICS")
    print("="*60)
    
    print(f"\n[Overall Statistics]")
    print(f"  Total expert activations: {stats['total_activations']:,}")
    print(f"  Unique experts used: {stats['total_unique_experts_used']:,} / {total_experts:,}")
    print(f"  Percentage of experts used: {stats['percent_experts_used']:.2f}%")
    print(f"  Average unique experts per layer: {stats['avg_experts_per_layer']:.1f}")
    
    print(f"\n[Top 20 Most Active Experts]")
    for i, (expert_idx, count) in enumerate(stats['top_20_experts'], 1):
        percentage = (count / stats['total_activations']) * 100
        print(f"  {i:2d}. Expert {expert_idx:3d}: {count:4d} activations ({percentage:.2f}%)")
    
    print(f"\n[Cache Performance]")
    cache_stats = runtime.stats.summary()
    print(f"  GPU hits: {cache_stats['gpu_hits']:,}")
    print(f"  GPU loads: {cache_stats['gpu_loads']:,}")
    print(f"  GPU evictions: {cache_stats['gpu_evictions']:,}")
    print(f"  CPU fallbacks: {cache_stats['cpu_runs']:,}")
    print(f"  Cache efficiency: {(cache_stats['gpu_hits'] / max(cache_stats['gpu_hits'] + cache_stats['gpu_loads'], 1)) * 100:.1f}%")
    
    # Save detailed results
    results = {
        "prompts": prompts,
        "total_experts": total_experts,
        "total_layers": total_layers,
        "activation_stats": stats,
        "cache_stats": cache_stats,
        "layer_expert_counts": {
            str(k): {str(ek): ev for ek, ev in v.items()} 
            for k, v in layer_expert_counts.items()
        }
    }
    
    output_file = "expert_activation_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Detailed results to {output_file}")
    
    # Create visualization
    try:
        heatmap_path = analysis.plot_activation_heatmap()
        print(f"\n[Visualization] Created expert activation heatmap")
    except Exception as e:
        print(f"\n[Warning] Could not create visualization: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return analysis, stats

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
    
    # Run analysis
    analysis, stats = analyze_expert_activation(prompts)
    
    # Print summary
    print(f"\n[Summary]")
    print(f"For {len(prompts)} pandas/data science prompts:")
    print(f"  - Used {stats['percent_experts_used']:.1f}% of available experts")
    print(f"  - Average {stats['avg_experts_per_layer']:.0f} unique experts per layer")
    print(f"  - Total {stats['total_activations']:,} expert activations")
    
    print("\n[Key Insight]")
    if stats['percent_experts_used'] < 10:
        print("  → Very sparse activation: Less than 10% of experts used")
        print("  → This suggests high specialization in the MoE model")
        print("  → Pandas/data science queries activate a specific subset of experts")
    elif stats['percent_experts_used'] < 30:
        print("  → Moderate sparsity: 10-30% of experts used")
        print("  → Good balance between specialization and coverage")
    else:
        print("  → Broad activation: Over 30% of experts used")
        print("  → These prompts engage diverse knowledge areas")

if __name__ == "__main__":
    main()