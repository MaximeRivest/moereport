"""
MoE Expert Analysis and Pruning Suite
======================================
A comprehensive toolkit for analyzing and pruning Mixture of Experts models.
Tracks expert activation patterns and enables targeted model pruning.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


@dataclass
class ExpertActivation:
    """Records a single expert activation event."""
    layer_idx: int
    expert_idx: int
    token_position: int
    token_id: int
    token_text: str
    routing_weight: float
    prompt_idx: int


@dataclass
class TokenExpertLog:
    """Complete log for a single token generation."""
    token_id: int
    token_text: str
    position: int
    prompt_idx: int
    layer_experts: Dict[int, List[Tuple[int, float]]]  # layer -> [(expert_id, weight)]


@dataclass
class PromptCompletionLog:
    """Complete log for a single prompt completion."""
    prompt: str
    prompt_idx: int
    generated_text: str
    token_logs: List[TokenExpertLog]
    total_tokens: int


class MoEExpertLogger:
    """
    Core logger for tracking expert activations during generation.
    """
    
    def __init__(self, model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        """Initialize the logger with a model."""
        print(f"Initializing MoE Expert Logger for {model_name}...")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
        
        # Extract model configuration
        self.config = self._extract_model_config()
        self._validate_moe_model()
        
        # Storage for logs
        self.prompt_logs: List[PromptCompletionLog] = []
        self.expert_activation_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        print(f"✓ Model loaded successfully")
        print(f"  - Total experts: {self.config['total_experts']:,}")
        print(f"  - Experts per layer: {self.config['num_experts']}")
        print(f"  - Active experts per token: {self.config['top_k']}")
        print(f"  - MoE layers: {self.config['num_hidden_layers']}")
    
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract MoE configuration from the model."""
        config = {
            'num_experts': getattr(self.model.config, 'num_routed_experts', 
                                  getattr(self.model.config, 'num_local_experts', 128)),
            'top_k': getattr(self.model.config, 'num_experts_per_tok', 8),
            'num_hidden_layers': self.model.config.num_hidden_layers,
        }
        config['total_experts'] = config['num_experts'] * config['num_hidden_layers']
        return config
    
    def _validate_moe_model(self):
        """Validate that this is indeed an MoE model."""
        test_input = self.tokenizer("test", return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(**test_input, output_router_logits=True)
        
        if not hasattr(output, 'router_logits') or output.router_logits is None:
            raise ValueError(f"Model {self.model_name} does not appear to be an MoE model or doesn't expose router logits")
        
        # Identify which layers have MoE
        self.moe_layers = []
        for i, logits in enumerate(output.router_logits):
            if logits is not None:
                self.moe_layers.append(i)
        
        print(f"  - MoE layers identified: {len(self.moe_layers)} layers")
    
    def log_single_generation(
        self,
        prompt: str,
        prompt_idx: int = 0,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        skip_initial_whitespace: bool = True
    ) -> PromptCompletionLog:
        """
        Generate completion for a single prompt and log all expert activations.
        
        Returns:
            PromptCompletionLog containing all token-level expert activations
        """
        # Format prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
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
        generated_tokens = []
        generated_text = ""
        skipped_whitespace = False
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=current_attention,
                    output_router_logits=True,
                    return_dict=True
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
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                
                # Skip initial whitespace if requested
                if skip_initial_whitespace and not skipped_whitespace and token_text.strip() == "":
                    skipped_whitespace = True
                    current_ids = torch.cat([current_ids, next_token_id], dim=1)
                    current_attention = torch.cat([
                        current_attention,
                        torch.ones((1, 1), device=current_attention.device, dtype=current_attention.dtype)
                    ], dim=1)
                    continue
                
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
                generated_tokens.append(token_id)
                generated_text += token_text
                
                # Update activation counts
                for layer_idx, experts_weights in layer_experts.items():
                    for expert_idx, weight in experts_weights:
                        self.expert_activation_counts[layer_idx][expert_idx] += 1
                
                # Check for EOS
                if token_id == self.tokenizer.eos_token_id:
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
            if layer_logits is None:
                continue
            
            # Handle different tensor shapes
            if layer_logits.dim() == 2:  # [seq_len, num_experts]
                if position < layer_logits.shape[0]:
                    pos_logits = layer_logits[position]
                else:
                    pos_logits = layer_logits[-1]
            else:  # [batch, seq_len, num_experts]
                if position < layer_logits.shape[1]:
                    pos_logits = layer_logits[0, position]
                else:
                    pos_logits = layer_logits[0, -1]
            
            # Get top-k experts with their weights
            probs = torch.softmax(pos_logits, dim=-1)
            top_k_values, top_k_indices = torch.topk(probs, self.config['top_k'])
            
            experts_weights = [
                (idx.item(), weight.item()) 
                for idx, weight in zip(top_k_indices, top_k_values)
            ]
            layer_experts[layer_idx] = experts_weights
        
        return layer_experts
    
    def process_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        batch_name: str = "default"
    ) -> List[PromptCompletionLog]:
        """
        Process multiple prompts and log all expert activations.
        
        Args:
            prompts: List of prompts to process
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            batch_name: Name for this batch of prompts
        
        Returns:
            List of PromptCompletionLog objects
        """
        print(f"\nProcessing {len(prompts)} prompts for batch '{batch_name}'...")
        
        logs = []
        for i, prompt in enumerate(prompts):
            if (i + 1) % max(1, len(prompts) // 10) == 0:
                print(f"  Progress: {i + 1}/{len(prompts)}")
            
            log = self.log_single_generation(
                prompt=prompt,
                prompt_idx=i,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            logs.append(log)
            self.prompt_logs.append(log)
        
        print(f"✓ Completed processing {len(prompts)} prompts")
        return logs


class MoEAnalyzer:
    """
    Analyzer for MoE expert activation patterns.
    """
    
    def __init__(self, logger: MoEExpertLogger):
        """Initialize analyzer with a logger instance."""
        self.logger = logger
        self.analysis_cache = {}
    
    def get_expert_token_matrix(self) -> Dict[int, np.ndarray]:
        """
        Create expert-token activation matrices for each layer.
        
        Returns:
            Dict mapping layer_idx to matrix [num_experts, total_tokens]
        """
        # Count total tokens
        total_tokens = sum(log.total_tokens for log in self.logger.prompt_logs)
        num_experts = self.logger.config['num_experts']
        
        # Initialize matrices
        matrices = {}
        for layer_idx in self.logger.moe_layers:
            matrices[layer_idx] = np.zeros((num_experts, total_tokens))
        
        # Fill matrices
        token_idx = 0
        for prompt_log in self.logger.prompt_logs:
            for token_log in prompt_log.token_logs:
                for layer_idx, experts_weights in token_log.layer_experts.items():
                    for expert_idx, weight in experts_weights:
                        matrices[layer_idx][expert_idx, token_idx] = weight
                token_idx += 1
        
        return matrices
    
    def get_unused_experts(self) -> Dict[int, Set[int]]:
        """
        Identify experts that were never activated.
        
        Returns:
            Dict mapping layer_idx to set of unused expert indices
        """
        unused = {}
        num_experts = self.logger.config['num_experts']
        
        for layer_idx in self.logger.moe_layers:
            used_experts = set(self.logger.expert_activation_counts.get(layer_idx, {}).keys())
            all_experts = set(range(num_experts))
            unused[layer_idx] = all_experts - used_experts
        
        return unused
    
    def get_expert_usage_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive usage statistics.
        
        Returns:
            Dictionary with detailed statistics
        """
        stats = {
            'model_config': self.logger.config,
            'num_prompts': len(self.logger.prompt_logs),
            'total_tokens_generated': sum(log.total_tokens for log in self.logger.prompt_logs),
            'layer_statistics': {},
            'global_statistics': {}
        }
        
        # Per-layer statistics
        total_experts_used = 0
        for layer_idx in self.logger.moe_layers:
            layer_counts = self.logger.expert_activation_counts.get(layer_idx, {})
            num_used = len(layer_counts)
            total_experts_used += num_used
            
            if layer_counts:
                counts = list(layer_counts.values())
                top_experts = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                counts = []
                top_experts = []
            
            stats['layer_statistics'][layer_idx] = {
                'experts_used': num_used,
                'experts_total': self.logger.config['num_experts'],
                'utilization_rate': num_used / self.logger.config['num_experts'],
                'activation_counts': dict(layer_counts),
                'top_10_experts': top_experts,
                'mean_activations': np.mean(counts) if counts else 0,
                'std_activations': np.std(counts) if counts else 0
            }
        
        # Global statistics
        total_possible = self.logger.config['total_experts']
        stats['global_statistics'] = {
            'total_experts_in_model': total_possible,
            'total_experts_used': total_experts_used,
            'global_utilization_rate': total_experts_used / total_possible,
            'potential_memory_savings': 1.0 - (total_experts_used / total_possible),
            'unused_experts_by_layer': {k: len(v) for k, v in self.get_unused_experts().items()}
        }
        
        return stats
    
    def get_expert_specialization_scores(
        self,
        category_prompts: Dict[str, List[int]]
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        """
        Calculate specialization scores for experts across different prompt categories.
        
        Args:
            category_prompts: Dict mapping category names to prompt indices
        
        Returns:
            Dict[layer_idx][expert_idx] -> specialization scores per category
        """
        specialization = defaultdict(lambda: defaultdict(dict))
        
        # Count activations per category
        category_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for prompt_log in self.logger.prompt_logs:
            # Determine category
            category = None
            for cat_name, indices in category_prompts.items():
                if prompt_log.prompt_idx in indices:
                    category = cat_name
                    break
            
            if category:
                for token_log in prompt_log.token_logs:
                    for layer_idx, experts_weights in token_log.layer_experts.items():
                        for expert_idx, weight in experts_weights:
                            category_counts[layer_idx][expert_idx][category] += 1
        
        # Calculate specialization scores
        for layer_idx in category_counts:
            for expert_idx in category_counts[layer_idx]:
                expert_cats = category_counts[layer_idx][expert_idx]
                total = sum(expert_cats.values())
                
                for category in expert_cats:
                    specialization[layer_idx][expert_idx][category] = expert_cats[category] / total
        
        return dict(specialization)
    
    def visualize_expert_usage(self, output_path: str = "expert_usage.png"):
        """Create visualization of expert usage patterns."""
        stats = self.get_expert_usage_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Utilization rate per layer
        layers = sorted(stats['layer_statistics'].keys())
        utilization = [stats['layer_statistics'][l]['utilization_rate'] for l in layers]
        
        axes[0, 0].bar(layers, utilization)
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Utilization Rate')
        axes[0, 0].set_title('Expert Utilization Rate by Layer')
        axes[0, 0].axhline(y=np.mean(utilization), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # 2. Distribution of activation counts
        all_counts = []
        for layer_stats in stats['layer_statistics'].values():
            all_counts.extend(layer_stats['activation_counts'].values())
        
        if all_counts:
            axes[0, 1].hist(all_counts, bins=50, edgecolor='black')
            axes[0, 1].set_xlabel('Activation Count')
            axes[0, 1].set_ylabel('Number of Experts')
            axes[0, 1].set_title('Distribution of Expert Activation Counts')
            axes[0, 1].axvline(x=np.mean(all_counts), color='r', linestyle='--', label='Mean')
            axes[0, 1].legend()
        
        # 3. Heatmap of top experts
        top_experts_matrix = np.zeros((len(layers), 10))
        for i, layer_idx in enumerate(layers):
            top_experts = stats['layer_statistics'][layer_idx]['top_10_experts']
            for j, (expert_idx, count) in enumerate(top_experts[:10]):
                top_experts_matrix[i, j] = count
        
        im = axes[1, 0].imshow(top_experts_matrix, aspect='auto', cmap='YlOrRd')
        axes[1, 0].set_xlabel('Top Expert Rank')
        axes[1, 0].set_ylabel('Layer Index')
        axes[1, 0].set_title('Top 10 Expert Activation Counts by Layer')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Summary statistics
        summary_text = f"""
        Global Statistics:
        
        Total Experts: {stats['global_statistics']['total_experts_in_model']:,}
        Experts Used: {stats['global_statistics']['total_experts_used']:,}
        Utilization: {stats['global_statistics']['global_utilization_rate']:.2%}
        
        Potential VRAM Savings: {stats['global_statistics']['potential_memory_savings']:.2%}
        
        Prompts Processed: {stats['num_prompts']}
        Tokens Generated: {stats['total_tokens_generated']}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualization saved to {output_path}")
    
    def export_analysis(self, output_path: str = "moe_analysis.json"):
        """Export complete analysis to JSON."""
        analysis = {
            'statistics': self.get_expert_usage_statistics(),
            'unused_experts': {k: list(v) for k, v in self.get_unused_experts().items()},
            'prompt_logs': [
                {
                    'prompt': log.prompt[:100] + '...' if len(log.prompt) > 100 else log.prompt,
                    'generated': log.generated_text[:100] + '...' if len(log.generated_text) > 100 else log.generated_text,
                    'total_tokens': log.total_tokens
                }
                for log in self.logger.prompt_logs
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Analysis exported to {output_path}")


class MoEPruner:
    """
    Pruner for creating optimized MoE models by removing unused experts.
    """
    
    def __init__(self, logger: MoEExpertLogger, analyzer: MoEAnalyzer):
        """Initialize pruner with logger and analyzer."""
        self.logger = logger
        self.analyzer = analyzer
    
    def create_pruning_config(
        self,
        keep_threshold: float = 0.0,
        min_experts_per_layer: Optional[int] = None
    ) -> Dict[int, List[int]]:
        """
        Create pruning configuration specifying which experts to keep.
        
        Args:
            keep_threshold: Minimum activation count to keep an expert (0 = keep all used)
            min_experts_per_layer: Minimum number of experts to keep per layer
        
        Returns:
            Dict mapping layer_idx to list of expert indices to keep
        """
        if min_experts_per_layer is None:
            min_experts_per_layer = self.logger.config['top_k']
        
        pruning_config = {}
        
        for layer_idx in self.logger.moe_layers:
            layer_counts = self.logger.expert_activation_counts.get(layer_idx, {})
            
            # Filter by threshold
            kept_experts = [
                expert_idx for expert_idx, count in layer_counts.items()
                if count > keep_threshold
            ]
            
            # Ensure minimum experts
            if len(kept_experts) < min_experts_per_layer:
                # Add most used experts until we have minimum
                sorted_experts = sorted(
                    layer_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                kept_experts = [e[0] for e in sorted_experts[:min_experts_per_layer]]
            
            pruning_config[layer_idx] = sorted(kept_experts)
        
        return pruning_config
    
    def calculate_pruning_stats(self, pruning_config: Dict[int, List[int]]) -> Dict[str, Any]:
        """Calculate statistics about the pruning configuration."""
        total_kept = sum(len(experts) for experts in pruning_config.values())
        total_possible = self.logger.config['total_experts']
        
        stats = {
            'total_experts_original': total_possible,
            'total_experts_kept': total_kept,
            'total_experts_pruned': total_possible - total_kept,
            'compression_ratio': total_kept / total_possible,
            'estimated_memory_reduction': 1.0 - (total_kept / total_possible),
            'per_layer_stats': {}
        }
        
        for layer_idx in self.logger.moe_layers:
            kept = len(pruning_config.get(layer_idx, []))
            total = self.logger.config['num_experts']
            stats['per_layer_stats'][layer_idx] = {
                'experts_kept': kept,
                'experts_total': total,
                'compression_ratio': kept / total
            }
        
        return stats
    
    def apply_pruning(
        self,
        pruning_config: Dict[int, List[int]],
        output_path: str = "pruned_model",
        save_safetensors: bool = True
    ) -> str:
        """
        Apply pruning to create a new optimized model.
        
        Args:
            pruning_config: Dict mapping layer_idx to list of expert indices to keep
            output_path: Path to save the pruned model
            save_safetensors: Whether to save in safetensors format
        
        Returns:
            Path to the saved model
        """
        print(f"\nApplying pruning configuration...")
        
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create new model config
        new_config = self.logger.model.config.to_dict()
        
        # Update config with pruning information
        new_config['pruning_config'] = {
            str(k): v for k, v in pruning_config.items()
        }
        new_config['original_num_experts'] = self.logger.config['num_experts']
        new_config['pruned'] = True
        
        # Save config
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Create pruned model state dict
        state_dict = self.logger.model.state_dict()
        pruned_state_dict = {}
        
        for name, param in state_dict.items():
            # Check if this is an expert parameter
            is_expert_param = False
            layer_idx = None
            expert_idx = None
            
            # Parse parameter name to identify expert parameters
            # This is model-specific and may need adjustment for different architectures
            if 'expert' in name.lower() or 'moe' in name.lower():
                # Try to extract layer and expert indices
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if 'layer' in part.lower():
                        try:
                            layer_idx = int(parts[i+1]) if i+1 < len(parts) else None
                        except (ValueError, IndexError):
                            pass
                    if 'expert' in part.lower():
                        try:
                            # Extract expert index from the parameter name
                            expert_match = re.search(r'expert[_.]?(\d+)', name.lower())
                            if expert_match:
                                expert_idx = int(expert_match.group(1))
                                is_expert_param = True
                        except (ValueError, AttributeError):
                            pass
            
            # Keep parameter if it's not an expert parameter or if the expert is kept
            if not is_expert_param:
                pruned_state_dict[name] = param
            elif layer_idx in pruning_config and expert_idx in pruning_config[layer_idx]:
                pruned_state_dict[name] = param
        
        # Save pruned model
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
        report_path = output_path / "pruning_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                'pruning_config': {str(k): v for k, v in pruning_config.items()},
                'statistics': stats,
                'model_name': self.logger.model_name
            }, f, indent=2)
        
        print(f"✓ Pruning complete. Model saved to {output_path}")
        print(f"  - Original experts: {stats['total_experts_original']:,}")
        print(f"  - Kept experts: {stats['total_experts_kept']:,}")
        print(f"  - Memory reduction: {stats['estimated_memory_reduction']:.1%}")
        
        return str(output_path)
    
    def export_pruning_config(self, pruning_config: Dict[int, List[int]], output_path: str = "pruning_config.json"):
        """Export pruning configuration to JSON."""
        config = {
            'model_name': self.logger.model_name,
            'model_config': self.logger.config,
            'pruning_config': {str(k): v for k, v in pruning_config.items()},
            'statistics': self.calculate_pruning_stats(pruning_config)
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Pruning configuration exported to {output_path}")


# ============================================================================
# Example Usage Functions
# ============================================================================

def example_sentiment_analysis():
    """Example: Analyzing expert usage for sentiment classification."""
    
    # Initialize logger
    logger = MoEExpertLogger("Qwen/Qwen3-30B-A3B-Instruct-2507")
    
    # Define prompts
    positive_prompts = [
        'Review: "This product is amazing!" Sentiment:',
        'Review: "Best purchase ever!" Sentiment:',
        'Review: "Highly recommend!" Sentiment:',
        'Review: "Excellent quality!" Sentiment:',
        'Review: "Love it!" Sentiment:',
    ]
    
    negative_prompts = [
        'Review: "Terrible product!" Sentiment:',
        'Review: "Waste of money!" Sentiment:',
        'Review: "Very disappointed!" Sentiment:',
        'Review: "Poor quality!" Sentiment:',
        'Review: "Do not buy!" Sentiment:',
    ]
    
    all_prompts = positive_prompts + negative_prompts
    
    # Process prompts
    logs = logger.process_prompts(
        prompts=all_prompts,
        max_new_tokens=1,  # Single token for classification
        temperature=0.0,
        batch_name="sentiment_classification"
    )
    
    # Analyze
    analyzer = MoEAnalyzer(logger)
    
    # Define categories for specialization analysis
    categories = {
        'positive': list(range(len(positive_prompts))),
        'negative': list(range(len(positive_prompts), len(all_prompts)))
    }
    
    # Get statistics
    stats = analyzer.get_expert_usage_statistics()
    print(f"\nGlobal utilization: {stats['global_statistics']['global_utilization_rate']:.2%}")
    print(f"Potential VRAM savings: {stats['global_statistics']['potential_memory_savings']:.2%}")
    
    # Get specialization scores
    specialization = analyzer.get_expert_specialization_scores(categories)
    
    # Visualize
    analyzer.visualize_expert_usage("sentiment_expert_usage.png")
    
    # Export analysis
    analyzer.export_analysis("sentiment_analysis.json")
    
    # Create pruning configuration
    pruner = MoEPruner(logger, analyzer)
    pruning_config = pruner.create_pruning_config(keep_threshold=0, min_experts_per_layer=8)
    
    # Calculate and display pruning statistics
    pruning_stats = pruner.calculate_pruning_stats(pruning_config)
    print(f"\nPruning statistics:")
    print(f"  - Compression ratio: {pruning_stats['compression_ratio']:.2%}")
    print(f"  - Memory reduction: {pruning_stats['estimated_memory_reduction']:.2%}")
    
    # Export pruning config
    pruner.export_pruning_config(pruning_config, "sentiment_pruning_config.json")
    
    # Optionally apply pruning
    # pruned_model_path = pruner.apply_pruning(pruning_config, "sentiment_pruned_model")
    
    return logger, analyzer, pruner


def example_comprehensive_analysis(
    model_name: str,
    prompts: List[str],
    task_name: str = "custom_task",
    max_new_tokens: int = 50,
    categories: Optional[Dict[str, List[int]]] = None
):
    """
    Comprehensive analysis pipeline for any task.
    
    Args:
        model_name: HuggingFace model name
        prompts: List of prompts to analyze
        task_name: Name for this analysis task
        max_new_tokens: Tokens to generate per prompt
        categories: Optional category mapping for specialization analysis
    
    Returns:
        Tuple of (logger, analyzer, pruner) for further use
    """
    print(f"\n{'='*70}")
    print(f"Starting Comprehensive MoE Analysis: {task_name}")
    print(f"{'='*70}")
    
    # Initialize
    logger = MoEExpertLogger(model_name)
    
    # Process prompts
    logs = logger.process_prompts(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        batch_name=task_name
    )
    
    # Analyze
    analyzer = MoEAnalyzer(logger)
    stats = analyzer.get_expert_usage_statistics()
    
    print(f"\nAnalysis Results:")
    print(f"  - Total tokens generated: {stats['total_tokens_generated']}")
    print(f"  - Global utilization: {stats['global_statistics']['global_utilization_rate']:.2%}")
    print(f"  - Potential VRAM savings: {stats['global_statistics']['potential_memory_savings']:.2%}")
    
    # Specialization analysis if categories provided
    if categories:
        specialization = analyzer.get_expert_specialization_scores(categories)
        print(f"  - Category specialization computed for {len(categories)} categories")
    
    # Visualize
    viz_path = f"{task_name}_expert_usage.png"
    analyzer.visualize_expert_usage(viz_path)
    
    # Export analysis
    analysis_path = f"{task_name}_analysis.json"
    analyzer.export_analysis(analysis_path)
    
    # Pruning
    pruner = MoEPruner(logger, analyzer)
    pruning_config = pruner.create_pruning_config(
        keep_threshold=0,
        min_experts_per_layer=logger.config['top_k']
    )
    
    pruning_stats = pruner.calculate_pruning_stats(pruning_config)
    
    print(f"\nPruning Configuration:")
    print(f"  - Experts to keep: {pruning_stats['total_experts_kept']:,}")
    print(f"  - Experts to prune: {pruning_stats['total_experts_pruned']:,}")
    print(f"  - Compression ratio: {pruning_stats['compression_ratio']:.2%}")
    print(f"  - Estimated memory reduction: {pruning_stats['estimated_memory_reduction']:.2%}")
    
    # Export pruning config
    config_path = f"{task_name}_pruning_config.json"
    pruner.export_pruning_config(pruning_config, config_path)
    
    print(f"\n✓ Analysis complete. Files saved:")
    print(f"  - Visualization: {viz_path}")
    print(f"  - Analysis: {analysis_path}")
    print(f"  - Pruning config: {config_path}")
    
    return logger, analyzer, pruner


if __name__ == "__main__":
    # Run sentiment analysis example
    # logger, analyzer, pruner = example_sentiment_analysis()
    
    # Or run a custom analysis
    custom_prompts = [
        "Translate to French: Hello world",
        "What is the capital of France?",
        "Explain quantum computing",
    ]
    logger, analyzer, pruner = example_comprehensive_analysis(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts=custom_prompts,
        task_name="mixed_tasks",
        max_new_tokens=50
    )