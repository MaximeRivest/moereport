"""
MoE Expert Analysis and Pruning Suite
=====================================
A comprehensive toolkit for analyzing and pruning Mixture of Experts models.
Tracks expert activation patterns and enables targeted model pruning.

Features:
- Token-by-token expert routing logging
- Multi-prompt batch processing
- Detailed analytics and visualization
- Hard pruning with model export
- Support for various MoE architectures (Qwen, Mixtral, etc.)
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import warnings
import gc
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TokenExpertLog:
    """Records expert routing for a single token."""
    token_id: int
    token_text: str
    position: int  # Position in generation sequence
    prompt_idx: int
    layer_experts: Dict[int, List[Tuple[int, float]]]  # layer -> [(expert_id, routing_weight)]


@dataclass
class PromptCompletionLog:
    """Complete log for a single prompt completion."""
    prompt: str
    prompt_idx: int
    generated_text: str
    token_logs: List[TokenExpertLog]
    total_tokens: int


@dataclass
class ExpertUsageStats:
    """Statistics for expert usage analysis."""
    total_experts: int
    experts_used: int
    utilization_rate: float
    layer_statistics: Dict[int, Dict[str, Any]]
    unused_experts: Dict[int, Set[int]]  # layer -> set of unused expert indices
    potential_memory_savings: float


# ==============================================================================
# Core MoE Expert Logger
# ==============================================================================

class MoEExpertLogger:
    """
    Core logger for tracking expert activations during generation.
    Handles requirement 1: Log experts per activation layer per token per prompt.
    """
    
    def __init__(self, model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        """Initialize the logger with a model."""
        print(f"Initializing MoE Expert Logger for {model_name}...")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval()
        
        # Extract model configuration
        self.config = self._extract_model_config()
        self._detect_moe_layers()
        
        # Storage for logs
        self.prompt_logs: List[PromptCompletionLog] = []
        self.expert_activation_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.expert_probability_mass: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        print(f"✓ Model loaded successfully")
        print(f"  - Total experts: {self.config['total_experts']:,}")
        print(f"  - Experts per layer: {self.config['num_experts']}")
        print(f"  - Active experts per token: {self.config['top_k']}")
        print(f"  - MoE layers detected: {len(self.moe_layers)}")
    
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract MoE configuration from the model."""
        cfg = self.model.config
        config = {}
        
        # Try different config keys for number of experts
        config['num_experts'] = (
            getattr(cfg, 'num_routed_experts', None) or
            getattr(cfg, 'num_local_experts', None) or
            getattr(cfg, 'num_experts', None)
        )
        
        # Try different config keys for top-k
        config['top_k'] = (
            getattr(cfg, 'num_experts_per_tok', None) or
            getattr(cfg, 'moe_top_k', None) or
            getattr(cfg, 'top_k', 8)  # Default fallback
        )
        
        config['num_hidden_layers'] = cfg.num_hidden_layers
        
        if config['num_experts'] is None:
            raise ValueError(f"Could not detect number of experts for {self.model_name}")
        
        return config
    
    def _detect_moe_layers(self):
        """Detect which layers have MoE by probing the model."""
        print("Detecting MoE layers...")
        
        # Ensure router logits output is enabled
        if hasattr(self.model.config, 'output_router_logits'):
            self.model.config.output_router_logits = True
        
        # Probe with a simple input
        probe_input = self.tokenizer("test", return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output = self.model(**probe_input, output_router_logits=True, return_dict=True)
        
        if not hasattr(output, 'router_logits') or output.router_logits is None:
            raise ValueError(f"Model {self.model_name} does not expose router logits")
        
        # Identify MoE layers
        self.moe_layers = []
        for i, logits in enumerate(output.router_logits):
            if logits is not None:
                self.moe_layers.append(i)
        
        self.config['total_experts'] = len(self.moe_layers) * self.config['num_experts']
    
    def log_single_generation(
        self,
        prompt: str,
        prompt_idx: int = 0,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> PromptCompletionLog:
        """
        Generate completion for a single prompt and log all expert activations.
        Implements requirement 1: Function to log experts per token.
        """
        # Format prompt with chat template if available
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            formatted_text = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(self.model.device)
        
        current_ids = inputs.input_ids
        current_attention = inputs.attention_mask
        
        token_logs = []
        generated_text = ""
        
        # Get EOS token IDs
        eos_ids = self._get_eos_ids()
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=current_attention,
                    output_router_logits=True,
                    return_dict=True,
                    use_cache=False  # Disable cache for accurate position tracking
                )
                
                # Get next token
                if temperature == 0:
                    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                else:
                    probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
                
                # Ensure correct dimensions
                if next_token_id.dim() == 0:
                    next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)
                elif next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(0)
                
                token_id = next_token_id[0, 0].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                
                # Extract expert routing for this token
                position = current_ids.shape[1] - 1
                layer_experts = self._extract_expert_routing(outputs.router_logits, position)
                
                # Create token log
                token_log = TokenExpertLog(
                    token_id=token_id,
                    token_text=token_text,
                    position=step,
                    prompt_idx=prompt_idx,
                    layer_experts=layer_experts
                )
                
                token_logs.append(token_log)
                generated_text += token_text
                
                # Update activation counts and probability mass
                for layer_idx, experts_weights in layer_experts.items():
                    for expert_idx, weight in experts_weights:
                        self.expert_activation_counts[layer_idx][expert_idx] += 1
                        self.expert_probability_mass[layer_idx][expert_idx] += weight
                
                # Check for EOS
                if token_id in eos_ids:
                    break
                
                # Update input for next iteration
                current_ids = torch.cat([current_ids, next_token_id], dim=1)
                current_attention = torch.cat([
                    current_attention,
                    torch.ones((1, 1), device=current_attention.device, dtype=current_attention.dtype)
                ], dim=1)
        
        return PromptCompletionLog(
            prompt=prompt,
            prompt_idx=prompt_idx,
            generated_text=generated_text,
            token_logs=token_logs,
            total_tokens=len(token_logs)
        )
    
    def _extract_expert_routing(
        self, 
        router_logits: List[Optional[torch.Tensor]], 
        position: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Extract expert routing information for a specific position."""
        layer_experts = {}
        
        for layer_idx, layer_logits in enumerate(router_logits):
            if layer_logits is None or layer_idx not in self.moe_layers:
                continue
            
            # Handle different tensor shapes
            if layer_logits.dim() == 2:  # [seq_len, num_experts]
                if position < layer_logits.shape[0]:
                    pos_logits = layer_logits[position]
                else:
                    pos_logits = layer_logits[-1]
            elif layer_logits.dim() == 3:  # [batch, seq_len, num_experts]
                if position < layer_logits.shape[1]:
                    pos_logits = layer_logits[0, position]
                else:
                    pos_logits = layer_logits[0, -1]
            else:
                continue
            
            # Get top-k experts with their weights
            probs = torch.softmax(pos_logits.to(torch.float32), dim=-1)
            top_k_values, top_k_indices = torch.topk(probs, self.config['top_k'])
            
            experts_weights = [
                (idx.item(), weight.item()) 
                for idx, weight in zip(top_k_indices, top_k_values)
            ]
            layer_experts[layer_idx] = experts_weights
        
        return layer_experts
    
    def _get_eos_ids(self) -> Set[int]:
        """Get all possible EOS token IDs."""
        eos_ids = set()
        
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_ids.update(self.tokenizer.eos_token_id)
            else:
                eos_ids.add(self.tokenizer.eos_token_id)
        
        # Handle model-specific EOS tokens
        special_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
        for token in special_tokens:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token)
                if tid is not None and tid >= 0:
                    eos_ids.add(tid)
            except:
                pass
        
        # Qwen3 specific
        if 151645 in self.tokenizer.get_vocab().values():
            eos_ids.add(151645)
        
        return eos_ids
    
    def process_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        batch_name: str = "default",
        system_prompt: Optional[str] = None
    ) -> List[PromptCompletionLog]:
        """
        Process multiple prompts and log all expert activations.
        Implements requirement 2: Function to process lists of prompts.
        """
        print(f"\nProcessing {len(prompts)} prompts for batch '{batch_name}'...")
        
        logs = []
        for i, prompt in enumerate(prompts):
            if (i + 1) % max(1, len(prompts) // 10) == 0 or i == 0:
                print(f"  Progress: {i + 1}/{len(prompts)}")
            
            log = self.log_single_generation(
                prompt=prompt,
                prompt_idx=i,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            logs.append(log)
            self.prompt_logs.append(log)
            
            # Memory management
            if (i + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"✓ Completed processing {len(prompts)} prompts")
        return logs


# ==============================================================================
# MoE Analyzer
# ==============================================================================

class MoEAnalyzer:
    """
    Analyzer for MoE expert activation patterns.
    Implements requirements 3-4: Analytics and unused expert identification.
    """
    
    def __init__(self, logger: MoEExpertLogger):
        """Initialize analyzer with a logger instance."""
        self.logger = logger
    
    def get_expert_usage_statistics(self) -> ExpertUsageStats:
        """
        Compute comprehensive usage statistics.
        Implements requirement 3: Summary analytics.
        """
        # Calculate per-layer statistics
        layer_statistics = {}
        total_experts_used = 0
        
        for layer_idx in self.logger.moe_layers:
            layer_counts = self.logger.expert_activation_counts.get(layer_idx, {})
            layer_mass = self.logger.expert_probability_mass.get(layer_idx, {})
            num_used = len(layer_counts)
            total_experts_used += num_used
            
            if layer_counts:
                counts = list(layer_counts.values())
                masses = list(layer_mass.values())
                top_experts = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                counts = []
                masses = []
                top_experts = []
            
            layer_statistics[layer_idx] = {
                'experts_used': num_used,
                'experts_total': self.logger.config['num_experts'],
                'utilization_rate': num_used / self.logger.config['num_experts'],
                'activation_counts': dict(layer_counts),
                'probability_mass': dict(layer_mass),
                'top_10_experts': top_experts,
                'mean_activations': np.mean(counts) if counts else 0,
                'std_activations': np.std(counts) if counts else 0,
                'mean_probability': np.mean(masses) if masses else 0,
            }
        
        # Calculate global statistics
        total_possible = self.logger.config['total_experts']
        utilization_rate = total_experts_used / total_possible if total_possible > 0 else 0
        
        # Get unused experts
        unused_experts = self.get_unused_experts()
        
        return ExpertUsageStats(
            total_experts=total_possible,
            experts_used=total_experts_used,
            utilization_rate=utilization_rate,
            layer_statistics=layer_statistics,
            unused_experts=unused_experts,
            potential_memory_savings=1.0 - utilization_rate
        )
    
    def get_unused_experts(self) -> Dict[int, Set[int]]:
        """
        Identify experts that were never activated.
        Implements requirement 4: Identify never-routed experts.
        """
        unused = {}
        num_experts = self.logger.config['num_experts']
        
        for layer_idx in self.logger.moe_layers:
            used_experts = set(self.logger.expert_activation_counts.get(layer_idx, {}).keys())
            all_experts = set(range(num_experts))
            unused[layer_idx] = all_experts - used_experts
        
        return unused
    
    def get_expert_token_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Get all (layer, expert, token_position) tuples.
        Part of requirement 3: Expert/token/layer pairs.
        """
        pairs = []
        
        for prompt_log in self.logger.prompt_logs:
            for token_log in prompt_log.token_logs:
                for layer_idx, experts_weights in token_log.layer_experts.items():
                    for expert_idx, _ in experts_weights:
                        pairs.append((layer_idx, expert_idx, token_log.position))
        
        return pairs
    
    def visualize_expert_usage(self, output_path: str = "expert_usage.png"):
        """Create visualization of expert usage patterns."""
        stats = self.get_expert_usage_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Utilization rate per layer
        layers = sorted(stats.layer_statistics.keys())
        utilization = [stats.layer_statistics[l]['utilization_rate'] for l in layers]
        
        axes[0, 0].bar(layers, utilization, color='steelblue')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Utilization Rate')
        axes[0, 0].set_title('Expert Utilization Rate by Layer')
        axes[0, 0].axhline(y=np.mean(utilization), color='r', linestyle='--', label=f'Mean: {np.mean(utilization):.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of activation counts
        all_counts = []
        for layer_stats in stats.layer_statistics.values():
            all_counts.extend(layer_stats['activation_counts'].values())
        
        if all_counts:
            axes[0, 1].hist(all_counts, bins=50, edgecolor='black', color='darkgreen', alpha=0.7)
            axes[0, 1].set_xlabel('Activation Count')
            axes[0, 1].set_ylabel('Number of Experts')
            axes[0, 1].set_title('Distribution of Expert Activation Counts')
            axes[0, 1].axvline(x=np.mean(all_counts), color='r', linestyle='--', label=f'Mean: {np.mean(all_counts):.1f}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Heatmap of expert usage across layers
        usage_matrix = np.zeros((len(layers), min(20, self.logger.config['num_experts'])))
        for i, layer_idx in enumerate(layers):
            counts = stats.layer_statistics[layer_idx]['activation_counts']
            for j in range(min(20, self.logger.config['num_experts'])):
                usage_matrix[i, j] = counts.get(j, 0)
        
        im = axes[1, 0].imshow(usage_matrix, aspect='auto', cmap='YlOrRd')
        axes[1, 0].set_xlabel('Expert Index (first 20)')
        axes[1, 0].set_ylabel('Layer Index')
        axes[1, 0].set_title('Expert Activation Heatmap')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Summary statistics
        summary_text = f"""
        Global Statistics:
        
        Total Experts: {stats.total_experts:,}
        Experts Used: {stats.experts_used:,}
        Utilization: {stats.utilization_rate:.2%}
        
        Potential VRAM Savings: {stats.potential_memory_savings:.2%}
        
        Prompts Processed: {len(self.logger.prompt_logs)}
        Tokens Generated: {sum(log.total_tokens for log in self.logger.prompt_logs)}
        
        Unused Experts per Layer:
        Min: {min(len(v) for v in stats.unused_experts.values()) if stats.unused_experts else 0}
        Max: {max(len(v) for v in stats.unused_experts.values()) if stats.unused_experts else 0}
        Avg: {np.mean([len(v) for v in stats.unused_experts.values()]):.1f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {output_path}")
    
    def export_analysis(self, output_path: str = "moe_analysis.json"):
        """Export complete analysis to JSON."""
        stats = self.get_expert_usage_statistics()
        
        analysis = {
            'model_name': self.logger.model_name,
            'model_config': self.logger.config,
            'statistics': {
                'total_experts': stats.total_experts,
                'experts_used': stats.experts_used,
                'utilization_rate': stats.utilization_rate,
                'potential_memory_savings': stats.potential_memory_savings,
                'layer_statistics': stats.layer_statistics,
            },
            'unused_experts': {k: list(v) for k, v in stats.unused_experts.items()},
            'prompts_processed': len(self.logger.prompt_logs),
            'total_tokens_generated': sum(log.total_tokens for log in self.logger.prompt_logs),
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Analysis exported to {output_path}")


# ==============================================================================
# MoE Pruner
# ==============================================================================

class MoEPruner:
    """
    Pruner for creating optimized MoE models by removing unused experts.
    Implements requirement 5: Prune unused experts and save model.
    """
    
    def __init__(self, logger: MoEExpertLogger, analyzer: MoEAnalyzer):
        """Initialize pruner with logger and analyzer."""
        self.logger = logger
        self.analyzer = analyzer
    def _is_moe_param_name(self, name: str) -> bool:
        n = name.lower()
        # Only touch MoE experts and router/gate inside MLP-ish blocks.
        in_mlpish = (".mlp." in n) or ("switch" in n) or (".moe" in n) or ("sparse" in n)
        is_expert  = "experts" in n
        is_router  = ("gate" in n) or ("router" in n)
        return in_mlpish and (is_expert or is_router)
    
    def create_pruning_config(
        self,
        coverage: float = 1.0,
        min_experts_per_layer: Optional[int] = None,
        use_probability_mass: bool = True
    ) -> Dict[int, List[int]]:
        """
        Create pruning configuration specifying which experts to keep.
        
        Args:
            coverage: Keep experts covering this fraction of probability mass (1.0 = all used experts)
            min_experts_per_layer: Minimum number of experts to keep per layer
            use_probability_mass: Use probability mass instead of counts for coverage calculation
        
        Returns:
            Dict mapping layer_idx to list of expert indices to keep
        """
        if min_experts_per_layer is None:
            min_experts_per_layer = self.logger.config['top_k']
        
        pruning_config = {}
        
        for layer_idx in self.logger.moe_layers:
            if use_probability_mass:
                layer_data = self.logger.expert_probability_mass.get(layer_idx, {})
            else:
                layer_data = self.logger.expert_activation_counts.get(layer_idx, {})
            
            if not layer_data:
                # No data for this layer, keep minimum experts
                pruning_config[layer_idx] = list(range(min_experts_per_layer))
                continue
            
            # Sort experts by mass/count
            sorted_experts = sorted(layer_data.items(), key=lambda x: x[1], reverse=True)
            
            if coverage >= 1.0:
                # Keep all used experts
                kept_experts = [e[0] for e in sorted_experts]
            else:
                # Keep experts until we reach coverage threshold
                total_mass = sum(e[1] for e in sorted_experts)
                cumulative_mass = 0
                kept_experts = []
                
                for expert_idx, mass in sorted_experts:
                    kept_experts.append(expert_idx)
                    cumulative_mass += mass
                    if cumulative_mass >= coverage * total_mass:
                        break
            
            # Ensure minimum experts
            if len(kept_experts) < min_experts_per_layer:
                # Add most used experts until we have minimum
                kept_experts = [e[0] for e in sorted_experts[:min_experts_per_layer]]
            
            pruning_config[layer_idx] = sorted(kept_experts)
        
        return pruning_config
    
    def calculate_pruning_stats(self, pruning_config: Dict[int, List[int]]) -> Dict[str, Any]:
        """Calculate statistics about the pruning configuration."""
        kept_per_layer = [len(experts) for experts in pruning_config.values()]
        total_kept = sum(kept_per_layer)
        total_possible = self.logger.config['total_experts']
        
        # For uniform pruning, all layers should keep the same number
        is_uniform = len(set(kept_per_layer)) == 1
        uniform_count = min(kept_per_layer) if kept_per_layer else 0
        
        stats = {
            'total_experts_original': total_possible,
            'total_experts_kept': total_kept,
            'total_experts_pruned': total_possible - total_kept,
            'compression_ratio': total_kept / total_possible if total_possible > 0 else 0,
            'estimated_memory_reduction': 1.0 - (total_kept / total_possible) if total_possible > 0 else 0,
            'is_uniform': is_uniform,
            'uniform_experts_per_layer': uniform_count if is_uniform else None,
            'experts_per_layer': kept_per_layer,
            'per_layer_config': pruning_config
        }
        
        return stats
    
    def apply_pruning(
        self,
        pruning_config: Dict[int, List[int]],
        output_path: str = "pruned_model",
        save_safetensors: bool = True,
        force_uniform: bool = True
    ) -> str:
        """
        Apply pruning to create a new optimized model.
        
        Args:
            pruning_config: Dict mapping layer_idx to list of expert indices to keep
            output_path: Path to save the pruned model
            save_safetensors: Whether to save in safetensors format
            force_uniform: Ensure all layers keep the same number of experts (required for most frameworks)
        
        Returns:
            Path to the saved model
        """
        print(f"\nApplying pruning configuration...")
        
        # Ensure uniform pruning if required
        if force_uniform:
            kept_counts = [len(experts) for experts in pruning_config.values()]
            min_count = min(kept_counts)
            pruning_config = {
                layer_idx: experts[:min_count] 
                for layer_idx, experts in pruning_config.items()
            }
            print(f"Enforcing uniform pruning: {min_count} experts per layer")
        
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict
        state_dict = self.logger.model.state_dict()
        pruned_state_dict = {}
        
        # Statistics for reporting
        pruned_params = []
        kept_params = []
        
        for name, param in state_dict.items():
            # Try to identify layer index from parameter name
            layer_idx = self._extract_layer_idx(name)
            
            if layer_idx is not None and layer_idx in pruning_config:
                # This is an MoE layer parameter
                pruned_param = self._prune_parameter(
                    param, 
                    name, 
                    pruning_config[layer_idx],
                    self.logger.config['num_experts']
                )
                
                if pruned_param is not None and pruned_param.shape != param.shape:
                    pruned_state_dict[name] = pruned_param
                    pruned_params.append((name, param.shape, pruned_param.shape))
                else:
                    pruned_state_dict[name] = param
                    kept_params.append(name)
            else:
                # Not an MoE parameter, keep as is
                pruned_state_dict[name] = param
                kept_params.append(name)
        
        # Update model config
        new_config = self.logger.model.config.to_dict()
        
        # Update number of experts in config
        uniform_count = len(list(pruning_config.values())[0]) if pruning_config else 0
        for key in ['num_routed_experts', 'num_local_experts', 'num_experts']:
            if key in new_config:
                new_config[key] = uniform_count
        
        # Add pruning metadata
        new_config['pruning_info'] = {
            'original_num_experts': self.logger.config['num_experts'],
            'pruned_num_experts': uniform_count,
            'pruning_config': {str(k): v for k, v in pruning_config.items()},
            'pruned': True
        }
        
        # Save config
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Save model weights
        if save_safetensors:
            safetensors_path = output_path / "model.safetensors"
            save_file(pruned_state_dict, safetensors_path)
            print(f"✓ Pruned model saved to {safetensors_path}")
        else:
            torch_path = output_path / "pytorch_model.bin"
            torch.save(pruned_state_dict, torch_path)
            print(f"✓ Pruned model saved to {torch_path}")
        
        # Save tokenizer
        self.logger.tokenizer.save_pretrained(output_path)
        
        # Save pruning report
        stats = self.calculate_pruning_stats(pruning_config)
        stats['pruned_parameters'] = pruned_params
        stats['model_name'] = self.logger.model_name
        
        report_path = output_path / "pruning_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Pruning complete. Model saved to {output_path}")
        print(f"  - Original experts per layer: {self.logger.config['num_experts']}")
        print(f"  - Kept experts per layer: {uniform_count}")
        print(f"  - Memory reduction: {stats['estimated_memory_reduction']:.1%}")
        print(f"  - Pruned {len(pruned_params)} parameters")
        
        return str(output_path)
    
    def _extract_layer_idx(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        patterns = [
            r'\.layers\.(\d+)\.',
            r'\.h\.(\d+)\.',
            r'\.blocks\.(\d+)\.',
            r'\.layer\.(\d+)\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        
        return None
    
    def _prune_parameter(self, param, name, keep_indices, num_experts):
        # 1) Only prune MoE params by name
        if not self._is_moe_param_name(name):
            return None

        keep_tensor = torch.tensor(keep_indices, device=param.device, dtype=torch.long)
        ln = name.lower()

        # 2) Router / gate matrices (last or first dim == num_experts)
        if ("gate" in ln) or ("router" in ln):
            if param.ndim >= 2 and param.shape[-1] == num_experts:
                return param[..., keep_indices]
            if param.ndim >= 1 and param.shape[0] == num_experts:
                return param[keep_tensor, ...]
            # If it doesn't match router shapes, don't touch it.
            return None

        # 3) Expert weights/biases (packed by experts)
        # Only proceed if this really looks like a packed experts tensor.
        if param.ndim == 3 and param.shape[0] == num_experts:
            return param.index_select(0, keep_tensor)

        if param.ndim == 2:
            # Packed along dim 0
            if (param.shape[0] % num_experts) == 0 and (param.shape[0] // num_experts) > 1:
                chunk = param.shape[0] // num_experts
                reshaped = param.view(num_experts, chunk, -1)
                pruned = reshaped.index_select(0, keep_tensor)
                return pruned.reshape(-1, param.shape[1])
            # Packed along dim 1
            if (param.shape[1] % num_experts) == 0 and (param.shape[1] // num_experts) > 1:
                chunk = param.shape[1] // num_experts
                reshaped = param.view(param.shape[0], num_experts, chunk)
                pruned = reshaped.index_select(1, keep_tensor)
                return pruned.reshape(param.shape[0], -1)
            return None

        if param.ndim == 1:
            if (param.shape[0] % num_experts) == 0:
                chunk = param.shape[0] // num_experts
                reshaped = param.view(num_experts, chunk)
                pruned = reshaped.index_select(0, keep_tensor)
                return pruned.reshape(-1)
            return None

        return None


# ==============================================================================
# Convenience Functions
# ==============================================================================

def comprehensive_moe_analysis(
    model_name: str,
    prompts: List[str],
    task_name: str = "custom_task",
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    coverage: float = 1.0,
    output_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    save_pruned_model: bool = False
) -> Tuple[MoEExpertLogger, MoEAnalyzer, MoEPruner]:
    """
    Complete pipeline for MoE analysis and pruning.
    
    Args:
        model_name: HuggingFace model name
        prompts: List of prompts to analyze
        task_name: Name for this analysis task
        max_new_tokens: Tokens to generate per prompt
        temperature: Sampling temperature (0 for greedy)
        coverage: Fraction of probability mass to keep when pruning
        output_dir: Directory to save outputs
        system_prompt: Optional system prompt for chat models
        save_pruned_model: Whether to save the pruned model
    
    Returns:
        Tuple of (logger, analyzer, pruner) for further use
    """
    print(f"\n{'='*70}")
    print(f"MoE Expert Analysis: {task_name}")
    print(f"{'='*70}")
    
    # Create output directory
    if output_dir is None:
        output_dir = f"./{model_name.replace('/', '_')}_analysis_{task_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = MoEExpertLogger(model_name)
    
    # Process prompts
    logs = logger.process_prompts(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_name=task_name,
        system_prompt=system_prompt
    )
    
    # Analyze
    analyzer = MoEAnalyzer(logger)
    stats = analyzer.get_expert_usage_statistics()
    
    print(f"\nAnalysis Results:")
    print(f"  - Total tokens generated: {sum(log.total_tokens for log in logs)}")
    print(f"  - Global utilization: {stats.utilization_rate:.2%}")
    print(f"  - Potential VRAM savings: {stats.potential_memory_savings:.2%}")
    print(f"  - Unused experts: {sum(len(v) for v in stats.unused_experts.values())}")
    
    # Visualize
    viz_path = f"{output_dir}/{task_name}_expert_usage.png"
    analyzer.visualize_expert_usage(viz_path)
    
    # Export analysis
    analysis_path = f"{output_dir}/{task_name}_analysis.json"
    analyzer.export_analysis(analysis_path)
    
    # Create pruner
    pruner = MoEPruner(logger, analyzer)
    
    # Create pruning configuration
    pruning_config = pruner.create_pruning_config(
        coverage=coverage,
        min_experts_per_layer=logger.config['top_k'],
        use_probability_mass=True
    )
    
    pruning_stats = pruner.calculate_pruning_stats(pruning_config)
    
    print(f"\nPruning Configuration:")
    print(f"  - Coverage target: {coverage:.2%}")
    print(f"  - Experts to keep: {pruning_stats['total_experts_kept']:,}")
    print(f"  - Experts to prune: {pruning_stats['total_experts_pruned']:,}")
    print(f"  - Compression ratio: {pruning_stats['compression_ratio']:.2%}")
    print(f"  - Estimated memory reduction: {pruning_stats['estimated_memory_reduction']:.2%}")
    
    # Save pruned model if requested
    if save_pruned_model and pruning_stats['estimated_memory_reduction'] > 0.01:
        pruned_path = f"{output_dir}/pruned_model"
        pruner.apply_pruning(
            pruning_config,
            output_path=pruned_path,
            save_safetensors=True,
            force_uniform=True
        )
    
    print(f"\n✓ Analysis complete. Results saved to {output_dir}")
    
    return logger, analyzer, pruner


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Example 1: Simple analysis with a few prompts
    prompts = [
        "Write a Python function to calculate the factorial of a number",
        "Explain the difference between lists and tuples in Python",
        "How do I handle exceptions in Python?",
        "What are decorators in Python and how do they work?",
        "Write a Python script to read and process a CSV file",
    ]
    
    # Run comprehensive analysis
    logger, analyzer, pruner = comprehensive_moe_analysis(
        #model_name="Qwen/Qwen2.5-3B-Instruct",  # Use smaller model for testing
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",  # Large model
        prompts=prompts,
        task_name="python_coding",
        max_new_tokens=50,
        temperature=0.0,
        coverage=1.0,  # Keep all used experts
        save_pruned_model=True  # Save the pruned model
    )
    
    # Example 2: Analyze unused experts
    unused = analyzer.get_unused_experts()
    print(f"\nUnused experts per layer:")
    for layer_idx in sorted(unused.keys()):
        print(f"  Layer {layer_idx}: {len(unused[layer_idx])} unused experts")