I have asked 4 llms to do that: 

Mixture of experts for large language models like Qwen 30b a3b have several activation layers and several layers within each activation layers. For instance qwen has 128 of which only 8 experts are activated per layers per forward pass for a total of (128 * 48) 6144 experts. If the assistant produces 2 token (2 forward pass) it would hit a specific combination of 8 experts per layer possibly with 0 over load but for more then 16 token (forward passes in the assistant) there will be necessary overlap because 8*16 equal 128. Its also possible that for each token, especially for different prompt but in a very very spelialez similar task. When one know what domain they will be producing tokens for they could leverage that and save on vram and load only the required experts at each activation layers. to leverage that we need a way to measure and save which expert for each activation was used by an assistant completion. and one we have that we can run analysis and prune. I have 3 scripts from ai assistants that gets us close to it but they each have some problems. So bugs even.
 
Your task is to write 1 function that log the experts per activation layers per token per prompt that are used upon assistants forwards pass. We also, want a function the uses that and takes a lists of prompts and runthe forward passes up to the end token and store of that information. Then we need a function that can tell us the exper/token(maybe position)/layer pairs and make summary analytics of that. another tool that tells us which expert for each layers that were never routed to for all our prompts/tokens completetion. Another finally that given that information get modify the llm architecture an prune the used experts at each layers, with all the proper hf transformer info so that inference tools can load them (llm, sglang, etC).

This is the most promising: 

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
        model_name="openai/gpt-oss-120b",
        # model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts=custom_prompts,
        task_name="mixed_tasks",
        max_new_tokens=50
    )

These are the other ones:

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel, Qwen3MoeDecoderLayer, Qwen3MoeSparseMlp
from transformers.models.qwen3 import Qwen3MLP
from collections import defaultdict
from typing import List, Dict, Tuple
from torch.nn import ModuleList
import os
import json

# Custom classes for variable num_experts per layer

class CustomQwen3MoeConfig(PretrainedConfig):
    model_type = "qwen3_moe"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_routed_experts_per_layer = kwargs.pop("num_routed_experts_per_layer", [self.num_routed_experts] * self.num_hidden_layers)

class CustomQwen3MoeSparseMlp(Qwen3MoeSparseMlp):
    def __init__(self, config, num_experts=None):
        if num_experts is not None:
            config.num_routed_experts = num_experts
        super().__init__(config)

class CustomQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx, num_experts=None):
        super().__init__(config, layer_idx)
        if num_experts is not None:
            self.mlp = CustomQwen3MoeSparseMlp(config, num_experts=num_experts)

class CustomQwen3MoeModel(Qwen3MoeModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            num_exp = config.num_routed_experts_per_layer[layer_idx]
            self.layers.append(CustomQwen3MoeDecoderLayer(config, layer_idx, num_experts=num_exp))
        self.post_init()

class CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen3MoeModel(config)
        self.post_init()

# Register for Auto loading
AutoModelForCausalLM.register(CustomQwen3MoeConfig, CustomQwen3MoeForCausalLM)

# Function 1: Log experts per token for a single prompt
def log_experts_per_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> Tuple[str, List[Dict]]:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    current_ids = inputs.input_ids
    current_attention = inputs.attention_mask

    generated_ids = []
    experts_log = []  # List of dicts, one per generated token

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_attention,
                output_router_logits=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

        generated_ids.append(next_id.item())
        token_text = tokenizer.decode(next_id)

        # Log experts for this token (routing at last position governs it)
        position = current_ids.shape[1] - 1
        token_experts = {}
        if hasattr(outputs, 'router_logits'):
            for layer_idx, layer_logits in enumerate(outputs.router_logits):
                if layer_logits is None:
                    continue
                # Handle shape: [batch, seq, experts] or [seq, experts]
                if layer_logits.dim() == 3:
                    pos_logits = layer_logits[0, position]
                else:
                    pos_logits = layer_logits[position]
                probs = torch.softmax(pos_logits, dim=-1)
                top_k = model.config.num_experts_per_tok
                top_experts = torch.topk(probs, top_k).indices.tolist()
                token_experts[layer_idx] = top_experts

        experts_log.append({
            'token_id': next_id.item(),
            'token_text': token_text,
            'position': len(generated_ids),
            'experts': token_experts
        })

        current_ids = torch.cat([current_ids, next_id], dim=1)
        current_attention = torch.cat([current_attention, torch.ones((1, 1), device=current_attention.device, dtype=current_attention.dtype)], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids)
    return generated_text, experts_log

# Function 2: Profile multiple prompts
def profile_multiple_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> List[Tuple[str, List[Dict]]]:
    results = []
    for prompt in prompts:
        gen_text, exp_log = log_experts_per_token(model, tokenizer, prompt, max_new_tokens, temperature)
        results.append((gen_text, exp_log))
    return results

# Function 3: Analyze expert usage
def analyze_expert_usage(profile_results: List[Tuple[str, List[Dict]]]) -> Dict:
    usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # layer -> expert -> position -> count
    total_activations = defaultdict(lambda: defaultdict(int))  # layer -> expert -> total count

    for _, exp_log in profile_results:
        for token_data in exp_log:
            pos = token_data['position']
            for layer, experts in token_data['experts'].items():
                for exp in experts:
                    usage[layer][exp][pos] += 1
                    total_activations[layer][exp] += 1

    summary = {
        'per_layer_usage_counts': {layer: dict(sorted(total_activations[layer].items(), key=lambda x: x[1], reverse=True)) for layer in total_activations},
        'per_layer_unique_experts': {layer: len(total_activations[layer]) for layer in total_activations},
        'per_position_per_layer': dict(usage),
    }
    return summary

# Function 4: Get unused experts per layer
def get_unused_experts(analysis: Dict, num_experts: int) -> Dict[int, Set[int]]:
    unused = {}
    for layer in analysis['per_layer_usage_counts']:
        used = set(analysis['per_layer_usage_counts'][layer].keys())
        unused[layer] = set(range(num_experts)) - used
    return unused

# Function 5: Prune model based on kept experts per layer
def prune_model(
    original_model: AutoModelForCausalLM,
    model_name: str,
    per_layer_kept: Dict[int, List[int]],  # layer -> sorted list of kept expert indices
    output_dir: str,
):
    config = CustomQwen3MoeConfig.from_pretrained(model_name)
    config.num_routed_experts_per_layer = [len(per_layer_kept.get(i, [])) for i in range(config.num_hidden_layers)]
    new_model = CustomQwen3MoeForCausalLM(config)

    # Copy non-layer parts
    new_model.model.embed_tokens.load_state_dict(original_model.model.embed_tokens.state_dict())
    new_model.model.norm.load_state_dict(original_model.model.norm.state_dict())
    new_model.lm_head.load_state_dict(original_model.lm_head.state_dict())

    for layer_idx in range(config.num_hidden_layers):
        orig_layer = original_model.model.layers[layer_idx]
        new_layer = new_model.model.layers[layer_idx]

        # Copy non-mlp
        new_layer.self_attn.load_state_dict(orig_layer.self_attn.state_dict())
        new_layer.input_layernorm.load_state_dict(orig_layer.input_layernorm.state_dict())
        new_layer.post_attention_layernorm.load_state_dict(orig_layer.post_attention_layernorm.state_dict())

        # Prune mlp
        kept = per_layer_kept.get(layer_idx, [])
        if not kept:
            continue

        orig_mlp = orig_layer.mlp
        new_mlp = new_layer.mlp

        # Gate: slice weight to kept
        new_mlp.gate.weight.data.copy_(orig_mlp.gate.weight[kept, :])

        # Experts: copy selected
        for new_i, orig_i in enumerate(kept):
            new_mlp.experts[new_i].load_state_dict(orig_mlp.experts[orig_i].state_dict())

        # Shared expert and gate
        new_mlp.shared_expert.load_state_dict(orig_mlp.shared_expert.state_dict())
        new_mlp.shared_expert_gate.load_state_dict(orig_mlp.shared_expert_gate.state_dict())

    os.makedirs(output_dir, exist_ok=True)
    new_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    # Save custom config explicitly
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    return output_dir


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
import json
import gc
from typing import List, Dict, Tuple, Optional, Any, Set
import os
import copy

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Utilities and Aggregation
# ==============================================================================

class LayerAggregator:
    """Accumulates probability mass per expert."""
    def __init__(self, moe_layers: List[int], num_experts: int):
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        # Track probability mass assigned to each expert per layer (on CPU to save VRAM)
        self.mass = {L: torch.zeros(num_experts, dtype=torch.float64) for L in moe_layers}
        self.tokens_seen = 0

    # Requirement 1: Logging function
    def update_from_router_logits(
        self,
        router_logits: List[Optional[torch.Tensor]],
        last_pos: int
    ):
        """
        Update stats based on router logits at the last position.
        """
        self.tokens_seen += 1
        for L, layer_logits in enumerate(router_logits):
            if layer_logits is None or L not in self.moe_layers:
                continue
            
            # Handle different shapes ([seq_len, num_experts] or [batch, seq_len, num_experts])
            try:
                if layer_logits.dim() == 2:
                    seq_len = layer_logits.shape[0]
                    target_pos = min(last_pos, seq_len - 1)
                    v = layer_logits[target_pos]
                elif layer_logits.dim() == 3:
                    # Assuming batch size 1 during profiling
                    seq_len = layer_logits.shape[1]
                    target_pos = min(last_pos, seq_len - 1)
                    v = layer_logits[0, target_pos]
                else:
                    continue
            except IndexError:
                continue
            
            # Calculate probabilities (softmax). Use float32 for stability.
            if v.numel() > 0:
                # Check for numerical stability
                if not torch.isfinite(v).all():
                    print(f"Warning: Non-finite values in router logits Layer {L}. Skipping token.")
                    continue
                
                probs = torch.softmax(v.to(torch.float32), dim=-1)
                # Update mass
                self.mass[L] += probs.to(torch.float64).cpu()

# ==============================================================================
# 2. MoE Profiler (Handles Requirements 1-4)
# ==============================================================================

class MoEProfiler:
    """
    Profiles MoE expert usage for specific tasks and generates pruning plans.
    """
    def __init__(self, model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        print(f"Loading model: {model_name}...")
        # trust_remote_code is essential for Qwen3 MoE architectures
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval()
        print("Model loaded.")
        
        self._detect_config()
        self.aggregator = LayerAggregator(self.moe_layers, self.num_experts)

    def _detect_config(self):
        """Detects MoE configuration robustly across various architectures."""
        cfg = self.model.config
        self.num_experts = None
        self.top_k = None

        # 1. Check nested configurations (Standardized approach in newer models)
        potential_moe_keys = ['moe_config', 'ffn_config', 'router_config']
        for key in potential_moe_keys:
            if hasattr(cfg, key) and getattr(cfg, key) is not None:
                moe_cfg = getattr(cfg, key)
                # Handle dictionary or object-like access
                if isinstance(moe_cfg, dict):
                    if self.num_experts is None:
                        self.num_experts = moe_cfg.get('num_experts', moe_cfg.get('num_routed_experts'))
                    if self.top_k is None:
                        self.top_k = moe_cfg.get('num_experts_per_tok', moe_cfg.get('moe_top_k'))
                else:
                    if self.num_experts is None:
                        self.num_experts = getattr(moe_cfg, 'num_experts', getattr(moe_cfg, 'num_routed_experts', None))
                    if self.top_k is None:
                        self.top_k = getattr(moe_cfg, 'num_experts_per_tok', getattr(moe_cfg, 'moe_top_k', None))

        # 2. Check top-level attributes (Common in many existing models, including Qwen3-30B-A3B)
        if self.num_experts is None:
            self.num_experts = getattr(cfg, 'num_experts', None)
        if self.num_experts is None:
            self.num_experts = getattr(cfg, 'num_routed_experts', None)
        if self.num_experts is None:
            self.num_experts = getattr(cfg, 'num_local_experts', None)
            
        if self.top_k is None:
             self.top_k = getattr(cfg, 'num_experts_per_tok', None)

        # 3. Final check and fallback
        if self.num_experts is None:
            print("Loaded Model Config Keys:")
            try: print(list(cfg.to_dict().keys()))
            except: pass
            raise ValueError(f"Could not detect the number of experts for {self.model_name}. Checked top-level and nested configs.")

        if self.top_k is None:
            self.top_k = 8 # Default fallback
            print(f"Warning: Could not detect Top-K (num_experts_per_tok). Defaulting to {self.top_k}.")

        # Probe to find which layers are MoE
        print("Probing model structure to identify MoE layers...")
        try:
             # Ensure the model configuration allows outputting router logits
            if hasattr(self.model.config, 'output_router_logits'):
                self.model.config.output_router_logits = True

            probe_inputs = self.tokenizer("probe test", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                probe_out = self.model(
                    **probe_inputs, output_router_logits=True, return_dict=True, use_cache=False
                )
            
            if not hasattr(probe_out, 'router_logits') or probe_out.router_logits is None:
                 raise ValueError("Model did not return 'router_logits' during probe.")

            router_logits_output = probe_out.router_logits

            # Handle potential variations in output structure (List vs Tuple of Tuples)
            if isinstance(router_logits_output, (list, tuple)):
                 # Standard structure: list where index corresponds to layer index, None if not MoE
                self.moe_layers = [L for L, x in enumerate(router_logits_output) if x is not None]
            else:
                raise TypeError(f"Unexpected type for router_logits: {type(router_logits_output)}")

        except Exception as e:
            raise RuntimeError(f"Failed to probe model for MoE layers: {e}")
            
        if not self.moe_layers:
            raise ValueError("No MoE layers were detected during probing.")

        self.total_experts = len(self.moe_layers) * self.num_experts

        print(f"\nModel Configuration Detected:")
        print(f"  • Total Layers: {getattr(cfg, 'num_hidden_layers', 'N/A')}")
        print(f"  • MoE Layers Detected: {len(self.moe_layers)}")
        print(f"  • Experts per Layer: {self.num_experts}")
        print(f"  • Experts per Token (Top-K): {self.top_k}")

    # Requirement 2: Prompt processing function
    def profile_task(self, prompts: List[str], task_name: str = "Task", max_new_tokens: int = 64):
        """
        Profile expert usage for a given task by running generation step-by-step.
        """
        print(f"\n{'='*70}\nPROFILING: {task_name} ({len(prompts)} prompts)\n{'='*70}")
        
        # Ensure the model is configured to output logits during the run
        if hasattr(self.model.config, 'output_router_logits'):
            self.model.config.output_router_logits = True

        for i, prompt in enumerate(prompts):
            if (i + 1) % max(1, len(prompts) // 5) == 0 or i == 0:
                print(f"  Progress: {i + 1}/{len(prompts)}")
            
            self._generate_and_log(prompt, max_new_tokens)
            
            # Memory management during long profiling runs
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"Profiling complete. Total tokens analyzed: {self.aggregator.tokens_seen}")

    def _get_eos_ids(self):
        """Helper to identify EOS token IDs robustly."""
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_ids.update(self.tokenizer.eos_token_id)
            else:
                eos_ids.add(self.tokenizer.eos_token_id)

        # Handle specific tokens (e.g., Qwen <|im_end|>)
        try:
            # Qwen3 often uses 151645 (<|im_end|>) as EOS for chat
            if 151645 in self.tokenizer.get_vocab():
                eos_ids.add(151645)
        except Exception:
            pass
        return eos_ids

    def _generate_and_log(self, prompt: str, max_new_tokens: int):
        """Step-by-step generation for a single prompt, logging router probs."""
        # Apply chat template if available
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_text = prompt
        
        # Set a reasonable max length for inputs
        inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        seq = inputs.input_ids
        attn = inputs.attention_mask

        eos_ids = self._get_eos_ids()

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(
                    input_ids=seq,
                    attention_mask=attn,
                    output_router_logits=True,
                    return_dict=True,
                    use_cache=False # Disable cache for easier position tracking and memory management
                )
                
                # Router logits at the last input position govern the NEXT token
                last_pos = seq.shape[1] - 1
                
                # Log the routing decisions for this step
                self.aggregator.update_from_router_logits(
                    out.router_logits, last_pos
                )

                # Get next token (greedy decoding for deterministic profiling)
                logits = out.logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1)
                
                # Ensure shape is [B, 1] for concatenation
                if next_id.dim() == 1:
                    next_id = next_id.unsqueeze(1)
                elif next_id.dim() == 0:
                    next_id = next_id.view(1, 1)
                
                # Check batch size consistency (assuming batch size 1)
                if next_id.shape[0] != seq.shape[0]:
                     next_id = next_id[0].view(1, 1)

                next_id_item = int(next_id[0, 0].item())

                # Stop on EOS
                if next_id_item in eos_ids:
                    break
                
                # Update sequence for next iteration
                seq = torch.cat([seq, next_id], dim=1)
                attn = torch.cat(
                    [attn, torch.ones((seq.shape[0], 1), device=attn.device, dtype=attn.dtype)],
                    dim=1,
                )

    # Requirement 3 & 4: Analysis and identifying unused experts
    def generate_uniform_pruning_plan(self, coverage: float = 0.99):
        """
        Analyzes the profiling data and generates a UNIFORM pruning plan (Union method).
        """
        print(f"\n{'='*70}\nGENERATING UNIFORM PRUNING PLAN (Target Coverage: {coverage*100:.2f}%)\n{'='*70}")

        if self.aggregator.tokens_seen == 0:
            print("No tokens profiled. Cannot generate plan."); return None

        plan = self._build_uniform_plan(self.aggregator, coverage, self.top_k)
        
        self._report_plan(plan)
        return plan

    def _compute_layer_requirements(self, mass_vec: torch.Tensor, coverage: float, top_k: int) -> Set[int]:
        """
        Determine experts required for a single layer to meet target coverage.
        Ensures at least 'top_k' experts are identified.
        """
        if mass_vec.numel() == 0:
            return set()
            
        total = float(mass_vec.sum().item())
        epsilon = 1e-9 # Threshold for numerical stability
        
        if total <= epsilon:
            # If a layer was never used, keep the first top_k experts for stability.
            return set(range(min(top_k, len(mass_vec))))
        
        # Sort experts by mass (descending)
        order = torch.argsort(mass_vec, descending=True)
        
        # If coverage is 1.0, we look for non-zero mass
        if coverage >= 1.0:
            non_zero_mask = mass_vec[order] > epsilon
            keep_n = max(top_k, int(non_zero_mask.sum().item()))
        else:
            # Calculate cumulative sum and find the target index
            cumsum = torch.cumsum(mass_vec[order], dim=0)
            target = coverage * total
            # Handle potential floating point inaccuracies
            target_tensor = torch.tensor(target, dtype=cumsum.dtype)
            idx = int(torch.searchsorted(cumsum, target_tensor).item())
            # Ensure we keep at least top_k experts
            keep_n = max(top_k, idx + 1)
        
        # Ensure keep_n does not exceed the number of experts
        keep_n = min(keep_n, len(mass_vec))
        
        keep_indices = order[:keep_n].cpu().tolist()
        return set(keep_indices)

    def _build_uniform_plan(self, agg: LayerAggregator, coverage: float, top_k: int) -> Dict[str, Any]:
        """
        Build a uniform pruning plan: identify the UNION of experts required
        across all layers.
        """
        required_experts_union = set()
        per_layer_requirements_count = {}

        # 1. Identify required experts per layer
        for L in agg.moe_layers:
            required_indices = self._compute_layer_requirements(agg.mass[L], coverage, top_k)
            required_experts_union.update(required_indices)
            per_layer_requirements_count[L] = len(required_indices)

        # 2. Create the uniform plan
        uniform_keep_indices = sorted(list(required_experts_union))
        uniform_keep_count = len(uniform_keep_indices)

        # 3. (Requirement 4) Identify globally unused experts
             
        # Recalculate union for 100% coverage to find truly unused experts
        if coverage < 1.0:
            # We need the union of experts required if coverage was 1.0
            union_100 = set()
            for L in agg.moe_layers:
                required_100 = self._compute_layer_requirements(agg.mass[L], 1.0, top_k)
                union_100.update(required_100)
            unused_100_coverage = set(range(agg.num_experts)) - union_100
        else:
            unused_100_coverage = set(range(agg.num_experts)) - required_experts_union


        plan = {
            "strategy": "uniform_union",
            "coverage_target": coverage,
            "uniform_keep_indices": uniform_keep_indices,
            "uniform_keep_count": uniform_keep_count,
            "original_num_experts": agg.num_experts,
            "per_layer_requirements_count": per_layer_requirements_count,
            "globally_unused_experts_count_100_coverage": len(unused_100_coverage)
        }
        return plan

    def _report_plan(self, plan: Dict[str, Any]):
        """Generate a human-readable report of the pruning plan. (Requirement 3)"""
        keep_count = plan["uniform_keep_count"]
        original_count = plan["original_num_experts"]
        
        if original_count == 0:
            savings_ratio = 0.0
        else:
            savings_ratio = 1.0 - (keep_count / original_count)
        
        print("\n--- Pruning Plan Summary ---")
        print(f"Strategy: Uniform (Union Method) - Maximizes compatibility")
        print(f"Original Experts per Layer: {original_count}")
        print(f"Experts Kept per Layer:     {keep_count}")
        print(f"💡 Pruning Rate (VRAM Savings Estimate for MoE layers): {savings_ratio*100:.2f}%")
        print(f"Experts with Zero Activations (at 100% coverage): {plan['globally_unused_experts_count_100_coverage']}")

        print("\n--- Per-Layer Requirements Analysis ---")
        
        requirements = list(plan['per_layer_requirements_count'].values())
        if not requirements: return

        avg_req = sum(requirements) / len(requirements)
        
        print(f"Min required: {min(requirements)}")
        print(f"Max required: {max(requirements)}")
        print(f"Avg required: {avg_req:.1f}")

# ==============================================================================
# 3. MoE Pruner (Requirement 5: Architectural Modification)
# ==============================================================================

class MoEPruner:
    """
    Applies a uniform pruning plan to a Hugging Face MoE model.
    Physically removes unused experts and updates the model configuration.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, pruning_plan: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.plan = pruning_plan
        self.config = model.config

    def prune(self):
        """
        Executes the pruning process in-place based on the uniform plan.
        """
        if self.plan is None:
            print("Pruning plan is missing. Aborting."); return self.model

        experts_to_keep_indices = self.plan["uniform_keep_indices"]
        new_num_experts = self.plan["uniform_keep_count"]
        original_num_experts = self.plan["original_num_experts"]

        if new_num_experts >= original_num_experts:
            print("No significant pruning possible."); return self.model

        print(f"\n{'='*70}\nSTARTING UNIFORM HARD PRUNING\n{'='*70}")
        print(f"Keeping {new_num_experts}/{original_num_experts} experts across all MoE layers.")

        # Identify decoder layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
             layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
             layers = self.model.layers
        else:
            raise RuntimeError("Could not identify the decoder layers (expected model.layers or model.model.layers).")

        # Identify MoE layer indices
        moe_layer_indices = sorted([int(k) for k in self.plan["per_layer_requirements_count"].keys()])
        
        # Convert keep indices to tensor for slicing
        keep_indices_tensor = torch.tensor(experts_to_keep_indices, dtype=torch.long).to(self.model.device)

        for layer_idx in moe_layer_indices:
            if layer_idx >= len(layers): continue

            print(f"Processing Layer {layer_idx}...")
            layer = layers[layer_idx]

            # Identify the MoE block within the layer
            moe_block = self._find_moe_block(layer)
            if moe_block is None:
                print(f"Warning: Could not find MoE block in layer {layer_idx}. Skipping.")
                continue
                
            # 1. Prune the experts (MLPs)
            self._prune_experts(moe_block, experts_to_keep_indices, keep_indices_tensor, original_num_experts)

            # 2. Prune the router (gate)
            gate, gate_name_path = self._find_router(moe_block, original_num_experts)
            
            if gate is not None:
                self._prune_router(moe_block, gate, gate_name_path, keep_indices_tensor, new_num_experts)
            else:
                # Handle cases where a router might not be present if only shared experts exist
                if self._handle_shared_expert_check(moe_block):
                     print(f"  Router not found, but shared expert detected. Proceeding.")
                else:
                    raise RuntimeError(f"Could not find router/gate in layer {layer_idx}. Pruning aborted.")

            # 3. Update internal attributes
            self._update_moe_block_attributes(moe_block, new_num_experts)


        # 4. Update the global configuration
        self._update_config(new_num_experts)

        print(f"\n{'='*70}\nPRUNING COMPLETE.\n{'='*70}")
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.model

    def _find_moe_block(self, layer):
        """Heuristically finds the MoE block within a decoder layer."""
        potential_names = ['mlp', 'block_sparse_moe', 'feed_forward']
        for name in potential_names:
            if hasattr(layer, name):
                module = getattr(layer, name)
                # Check for MoE indicators (gate, router, experts, fused weights, shared expert)
                if hasattr(module, 'gate') or hasattr(module, 'router') or hasattr(module, 'experts') or hasattr(module, 'w1') or hasattr(module, 'shared_expert'):
                     return module
        return None

    def _find_router(self, moe_block, num_experts, potential_names=['gate', 'router']):
        """Heuristically finds the router (gate) within the MoE block. Handles wrapped routers."""
        for name in potential_names:
            if hasattr(moe_block, name):
                module = getattr(moe_block, name)
                
                # Case 1: Direct nn.Linear
                if isinstance(module, nn.Linear) and module.out_features == num_experts:
                    return module, name
                
                # Case 2: Wrapped Linear (e.g., custom Router class containing the Linear layer)
                for internal_name in ['layer', 'dense', 'linear']:
                    if hasattr(module, internal_name) and isinstance(getattr(module, internal_name), nn.Linear):
                        internal_linear = getattr(module, internal_name)
                        if internal_linear.out_features == num_experts:
                            # Return the actual Linear layer and its full path
                            return internal_linear, f"{name}.{internal_name}"
                            
        return None, None

    def _handle_shared_expert_check(self, moe_block):
        """Checks if the block contains shared expert components."""
        return hasattr(moe_block, 'shared_expert')

    def _prune_experts(self, moe_block, keep_indices_list, keep_indices_tensor, original_num_experts):
        """Prunes the experts, handling both ModuleList and Fused Tensor implementations."""

        # Case 1: Experts stored as a ModuleList (e.g., Mixtral)
        if hasattr(moe_block, 'experts') and isinstance(moe_block.experts, nn.ModuleList):
            print("  Detected ModuleList implementation (e.g., Mixtral). Pruning...")
            new_experts = nn.ModuleList()
            for idx in keep_indices_list:
                new_experts.append(copy.deepcopy(moe_block.experts[idx]))
            moe_block.experts = new_experts
            return

        # Case 2: Fused/Concatenated Weights (e.g., Qwen MoE)
        pruned_weights = False
        weight_names = ['w1', 'w2', 'w3', 'gate_proj', 'up_proj', 'down_proj']
        
        # Determine the container
        container = moe_block
        if hasattr(moe_block, 'experts') and not isinstance(moe_block.experts, nn.ModuleList) and isinstance(moe_block.experts, nn.Module):
             container = moe_block.experts

        for name in weight_names:
            if hasattr(container, name):
                param = getattr(container, name)
                if isinstance(param, nn.Parameter):
                    weight = param.data
                    # Assume the first dimension is the expert dimension
                    if weight.shape[0] == original_num_experts:
                        print(f"  Detected Fused Tensor implementation (e.g., Qwen). Pruning {name}...")
                        new_weight = weight.index_select(0, keep_indices_tensor)
                        # Update the parameter in place
                        param.data = new_weight
                        pruned_weights = True
        
        if not pruned_weights:
             if self._handle_shared_expert_check(moe_block):
                 print("  No routed expert weights found (shared expert detected).")
             else:
                print(f"Warning: Could not identify routed expert weights structure in {type(container).__name__}.")

    def _prune_router(self, moe_block, gate, gate_name_path, keep_indices_tensor, new_num_experts):
        """Creates a new Linear layer for the router with reduced dimensions. Handles nested paths."""
        print(f"  Pruning router: {gate_name_path}...")
        
        # Navigate to the parent module if the path is nested (e.g., "router.layer")
        parent_module = moe_block
        path_parts = gate_name_path.split('.')
        gate_name = path_parts[-1]
        
        for part in path_parts[:-1]:
             parent_module = getattr(parent_module, part)

        # Slice the weights and bias along the output dimension (dim 0)
        new_gate_weight = gate.weight.data.index_select(0, keep_indices_tensor)
        new_gate_bias = gate.bias.data.index_select(0, keep_indices_tensor) if gate.bias is not None else None

        # Create the new gate module
        new_gate = nn.Linear(gate.in_features, new_num_experts, bias=new_gate_bias is not None)
        new_gate.weight.data = new_gate_weight
        if new_gate_bias is not None:
            new_gate.bias.data = new_gate_bias

        # Replace the old gate in the parent module
        setattr(parent_module, gate_name, new_gate.to(gate.weight.device, dtype=gate.weight.dtype))

    def _update_moe_block_attributes(self, moe_block, new_num_experts):
        """Updates internal attributes of the MoE block and its router."""
        # Update expert counts
        if hasattr(moe_block, 'num_experts'):
            moe_block.num_experts = new_num_experts
        if hasattr(moe_block, 'num_routed_experts'):
            moe_block.num_routed_experts = new_num_experts
            
        # Update Top-K on the block itself
        if hasattr(moe_block, 'top_k') and moe_block.top_k > new_num_experts:
             moe_block.top_k = new_num_experts

        # Update Top-K on the router module if it exists (e.g. Qwen3MoeRouter)
        for router_name in ['gate', 'router']:
             if hasattr(moe_block, router_name):
                  router_module = getattr(moe_block, router_name)
                  if hasattr(router_module, 'top_k') and router_module.top_k > new_num_experts:
                       router_module.top_k = new_num_experts

    def _update_config(self, new_num_experts):
        """Updates the global model configuration robustly."""
        print("Updating global model configuration...")
        
        cfg = self.config
        
        # 1. Update nested configurations
        potential_moe_keys = ['moe_config', 'ffn_config', 'router_config']
        for key in potential_moe_keys:
            if hasattr(cfg, key) and getattr(cfg, key) is not None:
                moe_cfg = getattr(cfg, key)
                # Update expert counts in nested config
                for expert_key in ['num_experts', 'num_routed_experts']:
                    if isinstance(moe_cfg, dict):
                        if expert_key in moe_cfg: moe_cfg[expert_key] = new_num_experts
                    else:
                        if hasattr(moe_cfg, expert_key): setattr(moe_cfg, expert_key, new_num_experts)
                
                # Adjust Top-K in nested config
                for tk_key in ['num_experts_per_tok', 'moe_top_k']:
                    current_tk = None
                    if isinstance(moe_cfg, dict):
                        current_tk = moe_cfg.get(tk_key)
                    else:
                        current_tk = getattr(moe_cfg, tk_key, None)

                    if current_tk is not None and current_tk > new_num_experts:
                        print(f"Adjusting Top-K ({key}.{tk_key}) to {new_num_experts}.")
                        if isinstance(moe_cfg, dict):
                            moe_cfg[tk_key] = new_num_experts
                        else:
                            setattr(moe_cfg, tk_key, new_num_experts)

        # 2. Update top-level configuration keys
        if hasattr(cfg, "num_experts"):
             cfg.num_experts = new_num_experts
        if hasattr(cfg, "num_routed_experts"):
            cfg.num_routed_experts = new_num_experts
        if hasattr(cfg, "num_local_experts"):
            cfg.num_local_experts = new_num_experts
            
        # Adjust Top-K at top level
        if hasattr(cfg, "num_experts_per_tok") and cfg.num_experts_per_tok > new_num_experts:
            print(f"Adjusting Top-K (num_experts_per_tok) to {new_num_experts}.")
            cfg.num_experts_per_tok = new_num_experts


    def save_pruned_model(self, output_dir: str):
        """Saves the pruned model and tokenizer to the specified directory."""
        print(f"\nSaving pruned model and tokenizer to {output_dir}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            # Ensure the model's internal config attribute is updated before saving
            self.model.config = self.config
            
            # Save model, tokenizer, and config. Use smaller shard size for large models.
            self.model.save_pretrained(output_dir, max_shard_size="4GB")
            self.tokenizer.save_pretrained(output_dir)
            
            print("✅ Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

# ==============================================================================
# 4. Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    # Model: Qwen3-30B-A3B (Requires significant VRAM, e.g., A100 80GB)
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
    
    # If you want to test with a smaller model first:
    # MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat" 
    
    TASK_NAME = "Python Coding Generation"
    MAX_NEW_TOKENS = 64  # Reduced for faster execution on large models
    PRUNING_COVERAGE = 1.0 # Keep experts covering 100% of the probability mass (all used experts)
    # PRUNING_COVERAGE = 0.99 # Keep experts covering 99% of the probability mass

    OUTPUT_DIR = f"./{MODEL_NAME.replace('/', '_')}-Pruned-Python"

    # Define prompts representative of the target task (Reduced count for faster execution)
    prompts = [
        "Write a Python function to calculate the Fibonacci sequence recursively.",
        "How do I use list comprehensions in Python? Provide an example.",
        "Explain the difference between '==' and 'is' in Python with code snippets.",
        "Write a Python script to read a CSV file using pandas and calculate the average.",
        "What are decorators in Python and how can I implement a simple timing decorator?",
    ]

    # Determine dtype
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            DTYPE = torch.bfloat16
            print("Using bfloat16.")
        else:
            DTYPE = torch.float16
            print("Using float16 (bfloat16 not supported).")
    else:
        DTYPE = torch.float32
        print("CUDA not available, using float32 on CPU (Warning: This will be extremely slow).")


    # --- Pipeline Execution ---
    try:
        # --- Step 1: Initialize Profiler (Loads the model) ---
        profiler = MoEProfiler(model_name=MODEL_NAME, device_map="auto", dtype=DTYPE)

        # --- Step 2: Profile the Task ---
        profiler.profile_task(prompts=prompts, task_name=TASK_NAME, max_new_tokens=MAX_NEW_TOKENS)

        # --- Step 3: Analyze and Generate Pruning Plan ---
        plan = profiler.generate_uniform_pruning_plan(coverage=PRUNING_COVERAGE)

        if plan:
            # --- Step 4: Apply Pruning ---
            pruner = MoEPruner(model=profiler.model, tokenizer=profiler.tokenizer, pruning_plan=plan)
            
            if plan["uniform_keep_count"] < plan["original_num_experts"]:
                pruned_model = pruner.prune()

                # --- Step 5: Save the Pruned Model ---
                pruner.save_pruned_model(OUTPUT_DIR)
                
                # --- Step 6: Verification (Optional) ---
                print(f"\nVerifying pruned model loading from {OUTPUT_DIR}...")
                
                # Clean up memory before loading the verification model
                del profiler, pruner, pruned_model
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    # Load the pruned model to ensure structural integrity
                    model_v = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto", torch_dtype=DTYPE, trust_remote_code=True)
                    tokenizer_v = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
                    print("Verification successful: Model loaded correctly.")
                    
                    # Test inference
                    test_prompt = "Write a python function to sort a list."
                    try:
                        messages = [{"role": "user", "content": test_prompt}]
                        formatted_text = tokenizer_v.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        formatted_text = test_prompt

                    test_input = tokenizer_v(formatted_text, return_tensors="pt").to(model_v.device)
                    
                    # Generate output (using greedy decoding for verification)
                    output = model_v.generate(**test_input, max_new_tokens=100, do_sample=False, eos_token_id=list(profiler._get_eos_ids()))
                    
                    # Decode only the generated part
                    input_len = test_input.input_ids.shape[1]
                    generated_output = output[0][input_len:]
                    print("\nTest Output:\n", tokenizer_v.decode(generated_output, skip_special_tokens=True))
                    
                except Exception as e:
                    print(f"Verification FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                print("\nSkipping pruning as the task utilizes nearly all experts at the target coverage.")

    except Exception as e:
        print(f"\nAn error occurred during the pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            print("\nHint: The model might be too large for your available VRAM/RAM. Try reducing MAX_NEW_TOKENS or the number of prompts.")

    print("\n✅ Script finished!")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# moe_log_and_prune.py

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class TokenRouting:
    gen_pos: int                 # 0-based index within generated tokens
    token_id: int
    piece: str
    per_layer_topk: Dict[int, List[int]]                  # {layer_idx: [expert_ids]}
    per_layer_probs: Optional[Dict[int, List[float]]] = None  # Optional top-k probs per layer


@dataclass
class PromptRouting:
    prompt: str
    generated_text: str
    tokens: List[TokenRouting] = field(default_factory=list)


@dataclass
class RoutingLog:
    model_name: str
    num_experts: int             # experts per MoE layer
    top_k: int                   # experts per token (router top-k)
    moe_layers: List[int]        # indices (within num_hidden_layers) that are MoE
    num_hidden_layers: int
    prompts: List[PromptRouting] = field(default_factory=list)


# ---------------------------
# 1) Core single-step logger
# ---------------------------

def log_experts_for_step(
    router_logits: List[Optional[torch.Tensor]],
    *,
    last_pos: int,
    moe_layers: List[int],
    top_k: int,
    with_probs: bool = True,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, List[float]]]]:
    """
    Log the routed experts for a SINGLE generation step (the token predicted from `last_pos`).

    Args:
        router_logits: list (len=num_hidden_layers) of tensors or None.
                       Each tensor is either [batch, seq_len, num_experts] or [seq_len, num_experts].
        last_pos:      int, index of the last input position used to predict the next token.
        moe_layers:    indices of layers that are MoE (others will be ignored).
        top_k:         how many experts are routed per token (from model.config).
        with_probs:    also return top-k probabilities per layer (softmax over experts).

    Returns:
        (per_layer_topk, per_layer_probs) where:
            per_layer_topk: {layer_idx: [expert_ids]}
            per_layer_probs: {layer_idx: [probabilities]} or None
    """
    per_layer_topk: Dict[int, List[int]] = {}
    per_layer_probs: Optional[Dict[int, List[float]]] = {} if with_probs else None

    for L, logits in enumerate(router_logits):
        if L not in moe_layers or logits is None:
            continue

        # Normalize shape -> [num_experts]
        if logits.dim() == 3:
            # [batch, seq_len, num_experts]
            v = logits[0, min(last_pos, logits.shape[1] - 1)]
        elif logits.dim() == 2:
            # [seq_len, num_experts]
            v = logits[min(last_pos, logits.shape[0] - 1)]
        else:
            # Unexpected
            continue

        probs = torch.softmax(v.to(torch.float32), dim=-1)
        top = torch.topk(probs, k=top_k, dim=-1)
        per_layer_topk[L] = top.indices.cpu().tolist()
        if with_probs and per_layer_probs is not None:
            per_layer_probs[L] = top.values.cpu().tolist()

    return per_layer_topk, per_layer_probs


# ---------------------------
# Helpers: model probing
# ---------------------------

def _detect_moe_layers(model) -> List[int]:
    """Run a cheap probe to see which transformer layers expose router logits."""
    probe = {"input_ids": torch.tensor([[model.config.eos_token_id or 1]], device=model.device)}
    with torch.no_grad():
        out = model(**probe, output_router_logits=True, return_dict=True)
    return [i for i, t in enumerate(out.router_logits) if t is not None]


def _collect_eos_ids(tokenizer) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    # Chat models often have chat-specific end tokens
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.append(tid)
        except Exception:
            pass
    return sorted(set(ids))


# ---------------------------
# 2) Runner over prompts
# ---------------------------

def capture_routing_for_prompts(
    model_name: str,
    prompts: List[str],
    *,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    temperature: float = 0.0,       # greedy by default
    top_p: Optional[float] = None,   # if you want sampling
    with_probs: bool = True,         # record top-k probabilities per layer
) -> RoutingLog:
    """
    Generate for each prompt while capturing MoE routing per token, per layer.

    Returns:
        RoutingLog (see dataclass)
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"[load] {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype
    )
    print("[ok] model loaded.")

    top_k = getattr(model.config, "num_experts_per_tok", 8)
    num_hidden = model.config.num_hidden_layers
    num_experts = getattr(model.config, "num_routed_experts",
                          getattr(model.config, "num_local_experts", 128))
    moe_layers = _detect_moe_layers(model)

    def apply_chat(msg: str) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": msg})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    eos_ids = set(_collect_eos_ids(tok))

    log = RoutingLog(
        model_name=model_name,
        num_experts=num_experts,
        top_k=top_k,
        moe_layers=moe_layers,
        num_hidden_layers=num_hidden,
    )

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] capture …")
        formatted = apply_chat(prompt)
        inputs = tok(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        seq = inputs.input_ids
        attn = inputs.attention_mask

        pr = PromptRouting(prompt=prompt, generated_text="", tokens=[])

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=seq,
                    attention_mask=attn,
                    output_router_logits=True,
                    return_dict=True,
                )

                # Next-token distribution
                logits = out.logits[:, -1, :]
                if temperature and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    if top_p is not None and 0.0 < top_p < 1.0:
                        # nucleus sampling
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cdf = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = (cdf > top_p).float().argmax(dim=-1)
                        # mask tail per batch=1
                        k = int(cutoff.item()) + 1
                        probs = torch.zeros_like(probs).scatter_(
                            1, sorted_idx[:, :k], sorted_probs[:, :k]
                        )
                        probs.div_(probs.sum(dim=-1, keepdim=True))
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                next_id_item = int(next_id[0, 0].item())
                piece = tok.decode([next_id_item], skip_special_tokens=False)

                # Log MoE routing for THIS step
                last_pos = seq.shape[1] - 1
                per_layer_topk, per_layer_probs = log_experts_for_step(
                    out.router_logits,
                    last_pos=last_pos,
                    moe_layers=moe_layers,
                    top_k=top_k,
                    with_probs=with_probs,
                )

                pr.tokens.append(TokenRouting(
                    gen_pos=step,
                    token_id=next_id_item,
                    piece=tok.decode([next_id_item], skip_special_tokens=True),
                    per_layer_topk=per_layer_topk,
                    per_layer_probs=per_layer_probs,
                ))

                # Update sequence
                seq = torch.cat([seq, next_id], dim=1)
                attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
                pr.generated_text += tok.decode([next_id_item], skip_special_tokens=True)

                if next_id_item in eos_ids:
                    break

        log.prompts.append(pr)

    return log


# ---------------------------
# 3) Summary analytics
# ---------------------------

@dataclass
class Summary:
    # Aggregates
    counts_per_layer_expert: Dict[int, Dict[int, int]]
    mass_per_layer_expert: Optional[Dict[int, Dict[int, float]]]  # if probs present
    # Simple derived metrics
    per_position_unique_experts: Dict[int, int]
    utilization_percent: float
    unique_pairs: Set[Tuple[int, int]]  # (layer, expert)
    top_per_layer: Dict[int, List[Tuple[int, int]]]  # [(expert, count)]


def summarize_routing(log: RoutingLog) -> Summary:
    counts: Dict[int, Dict[int, int]] = {}              # layer -> expert -> count
    mass: Optional[Dict[int, Dict[int, float]]] = {}    # if probs were recorded
    per_pos_unique: Dict[int, Set[Tuple[int, int]]] = {}  # pos -> {(layer, expert)}

    # Initialize
    for L in log.moe_layers:
        counts[L] = {}
        if mass is not None:
            mass[L] = {}

    # Traverse
    for pr in log.prompts:
        for t in pr.tokens:
            pos = t.gen_pos
            if pos not in per_pos_unique:
                per_pos_unique[pos] = set()
            for L, experts in t.per_layer_topk.items():
                for e_idx, e in enumerate(experts):
                    counts[L][e] = counts[L].get(e, 0) + 1
                    per_pos_unique[pos].add((L, e))
                    if mass is not None and t.per_layer_probs and L in t.per_layer_probs:
                        prob = float(t.per_layer_probs[L][e_idx])
                        mass[L][e] = mass[L].get(e, 0.0) + prob

    unique_pairs = set()
    for L, es in counts.items():
        for e in es.keys():
            unique_pairs.add((L, e))

    total_possible = len(log.moe_layers) * log.num_experts
    utilization_percent = (len(unique_pairs) / total_possible * 100.0) if total_possible else 0.0
    per_position_unique_experts = {pos: len(s) for pos, s in per_pos_unique.items()}

    top_per_layer: Dict[int, List[Tuple[int, int]]] = {}
    for L, es in counts.items():
        top_per_layer[L] = sorted(es.items(), key=lambda x: x[1], reverse=True)

    return Summary(
        counts_per_layer_expert=counts,
        mass_per_layer_expert=mass if mass != {} else None,
        per_position_unique_experts=per_position_unique_experts,
        utilization_percent=utilization_percent,
        unique_pairs=unique_pairs,
        top_per_layer=top_per_layer,
    )


# ---------------------------
# 4) Unused experts per layer
# ---------------------------

def find_never_routed_experts(
    log: RoutingLog,
    summary: Optional[Summary] = None
) -> Dict[int, Set[int]]:
    """
    Returns {layer_idx: {expert_ids never routed}}.
    """
    if summary is None:
        summary = summarize_routing(log)

    unused: Dict[int, Set[int]] = {}
    for L in log.moe_layers:
        used = set(summary.counts_per_layer_expert.get(L, {}).keys())
        all_e = set(range(log.num_experts))
        unused[L] = all_e - used
    return unused


# ---------------------------
# 5) Build keep plans & prune exporter
# ---------------------------

def build_keep_map_from_summary(
    summary: Summary,
    *,
    num_experts: int,
    moe_layers: Iterable[int],
    coverage: float = 1.00,
    min_keep: Optional[int] = None,
    use_mass_if_available: bool = True,
) -> Dict[int, List[int]]:
    """
    For each MoE layer, choose a subset of experts to KEEP that covers the requested
    probability mass (if available) or routed-count mass.

    Returns:
        {layer_idx: [expert_ids_to_keep]}
    """
    keep_map: Dict[int, List[int]] = {}

    for L in moe_layers:
        if use_mass_if_available and summary.mass_per_layer_expert is not None:
            vec = summary.mass_per_layer_expert.get(L, {})
        else:
            vec = summary.counts_per_layer_expert.get(L, {})

        # Build dense vector of length num_experts
        dense = torch.zeros(num_experts, dtype=torch.float64)
        for e, v in vec.items():
            dense[e] = float(v)

        total = float(dense.sum().item())
        if total <= 0.0:
            # nothing routed for this layer; keep at least one (or min_keep/top_k)
            k = min_keep or 1
            keep_map[L] = list(range(k))
            continue

        order = torch.argsort(dense, descending=True)
        cumsum = torch.cumsum(dense[order], dim=0)
        target = coverage * total
        idx = int(torch.searchsorted(cumsum, torch.tensor(target)).item())
        keep_n = idx + 1
        if min_keep is not None:
            keep_n = max(keep_n, min_keep)
        keep_n = min(keep_n, num_experts)

        keep_map[L] = order[:keep_n].cpu().tolist()

    return keep_map


def _uniformize_keep_map(keep_map: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], int]:
    """
    Ensure all layers keep the same count. We trim each layer to the MIN count.
    """
    if not keep_map:
        return keep_map, 0
    counts = [len(v) for v in keep_map.values()]
    m = min(counts)
    uniform: Dict[int, List[int]] = {L: v[:m] for L, v in keep_map.items()}
    return uniform, m


# ---- pruning helpers (generic, Mixtral/Qwen-style packed expert dims) ----

_PATTERNS_LAYER_IDX = [
    r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\.",
    r"\.model\.(\d+)\.", r"\.transformer\.(\d+)\."
]

def _infer_layer_idx_from_name(name: str) -> Optional[int]:
    for pat in _PATTERNS_LAYER_IDX:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    # fallback: first plain number token
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _slice_packed_param(
    tensor: torch.Tensor,
    keep: List[int],
    num_experts: int,
    *,
    name: str,
) -> torch.Tensor:
    """
    Try to slice expert-packed parameters. Handles common patterns:

    - 3D with expert axis at dim 0: [E, *]
    - 2D packed along rows: [E * d, in]  -> reshape [E, d, in], slice, reshape back
    - 2D packed along cols: [out, E * d] -> reshape [out, E, d], slice on dim=1, reshape back
    - 1D biases packed: [E * d]          -> reshape [E, d], slice, reshape back
    - Router with explicit E dimension [E, in] or bias [E]

    If the shape doesn’t look like any of these, it’s returned unchanged.
    """
    E = num_experts
    dims = tensor.dim()

    # 3D with experts along dim 0
    if dims >= 3 and tensor.shape[0] == E:
        return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))

    # 2D
    if dims == 2:
        rows, cols = tensor.shape

        # Router-like: [E, in] (slice rows)
        if rows == E:
            return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))

        # Row-packed: [E * d, in]  -> slice rows
        if rows % E == 0 and rows // E > 1:
            d = rows // E
            t = tensor.view(E, d, cols)
            t = t.index_select(0, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(len(keep) * d, cols)

        # Col-packed: [out, E * d] -> slice columns
        if cols % E == 0 and cols // E > 1:
            d = cols // E
            t = tensor.view(rows, E, d)
            t = t.index_select(1, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(rows, len(keep) * d)

        # Router-like: [out, E] (slice cols)
        if cols == E:
            return tensor.index_select(1, torch.as_tensor(keep, device=tensor.device))

    # 1D bias packed: [E * d] or [E]
    if dims == 1:
        n = tensor.shape[0]
        if n == E:
            return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))
        if n % E == 0 and n // E > 1:
            d = n // E
            t = tensor.view(E, d)
            t = t.index_select(0, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(len(keep) * d)

    # Not recognized – leave unchanged
    return tensor


def export_pruned_checkpoint(
    model_name: str,
    keep_map: Dict[int, List[int]],
    *,
    output_dir: str,
    dtype: str = "bfloat16",
    device_map: str = "cpu",
    strict_uniform: bool = True,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Export a *hard-pruned* checkpoint containing only the kept experts.

    Constraints:
      - All MoE layers must keep the same number of experts (HF configs assume uniform E).
      - This function recognizes common packed expert layouts (Qwen/Mixtral). For other
        layouts you may need to extend `_slice_packed_param`.

    Returns:
      A small report dict describing parameter changes. If dry_run=True, no files are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"[load] {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    print("[ok] model loaded.")

    # Validate uniformity
    keep_map_u, keep_count = _uniformize_keep_map(keep_map)
    if strict_uniform:
        lens = {len(v) for v in keep_map_u.values()}
        if len(lens) != 1:
            raise ValueError(
                "Non-uniform keep counts per layer. Use a uniform keep plan "
                "or set strict_uniform=False (not recommended)."
            )

    # Probe to know which layers are MoE and num_experts
    moe_layers = _detect_moe_layers(model)
    num_experts = getattr(model.config, "num_routed_experts",
                          getattr(model.config, "num_local_experts", 128))
    print(f"[prune] MoE layers: {moe_layers}")
    print(f"[prune] experts per layer: {num_experts}, keep: {keep_count}")

    sd = model.state_dict()
    new_sd = {}
    changes: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for name, tensor in sd.items():
        L = _infer_layer_idx_from_name(name)
        if L is None or L not in keep_map_u or L not in moe_layers:
            # Not a MoE layer param (or not identifiable) -> keep as-is
            new_sd[name] = tensor
            continue

        kept = keep_map_u[L]
        new_tensor = _slice_packed_param(tensor, kept, num_experts, name=name)
        if new_tensor.shape != tensor.shape:
            changes.append((name, tuple(tensor.shape), tuple(new_tensor.shape)))
        new_sd[name] = new_tensor

    # Update config – set experts per layer to keep_count (uniform)
    cfg = model.config.to_dict()
    for key in ("num_routed_experts", "num_local_experts"):
        if key in cfg:
            cfg[key] = keep_count

    report = {
        "model_name": model_name,
        "moe_layers": moe_layers,
        "original_num_experts": num_experts,
        "kept_per_layer": keep_count,
        "param_changes": changes,
        "output_dir": output_dir,
        "dry_run": dry_run,
    }

    if dry_run:
        print("\n[dry-run] The following parameters would be sliced:")
        for n, old, new in changes:
            print(f"  - {n}: {old} -> {new}")
        print("\n[dry-run] No files written.")
        return report

    # Write files
    torch.save(new_sd, os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Save tokenizer artifacts (for convenience)
    try:
        tok.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"[ok] wrote pruned checkpoint to: {output_dir}")
    return report


# ---------------------------
# Convenience I/O
# ---------------------------

def save_routing_log(log: RoutingLog, path: str) -> None:
    def _tok(tok: TokenRouting) -> Dict[str, Any]:
        return {
            "gen_pos": tok.gen_pos,
            "token_id": tok.token_id,
            "piece": tok.piece,
            "per_layer_topk": tok.per_layer_topk,
            "per_layer_probs": tok.per_layer_probs,
        }

    payload = {
        "model_name": log.model_name,
        "num_experts": log.num_experts,
        "top_k": log.top_k,
        "moe_layers": log.moe_layers,
        "num_hidden_layers": log.num_hidden_layers,
        "prompts": [
            {
                "prompt": p.prompt,
                "generated_text": p.generated_text,
                "tokens": [_tok(t) for t in p.tokens],
            }
            for p in log.prompts
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_routing_log(path: str) -> RoutingLog:
    with open(path, "r") as f:
        data = json.load(f)

    log = RoutingLog(
        model_name=data["model_name"],
        num_experts=data["num_experts"],
        top_k=data["top_k"],
        moe_layers=data["moe_layers"],
        num_hidden_layers=data["num_hidden_layers"],
    )
    for p in data["prompts"]:
        pr = PromptRouting(prompt=p["prompt"], generated_text=p["generated_text"])
        for t in p["tokens"]:
            pr.tokens.append(TokenRouting(
                gen_pos=t["gen_pos"],
                token_id=t["token_id"],
                piece=t.get("piece", ""),
                per_layer_topk={int(k): v for k, v in t["per_layer_topk"].items()},
                per_layer_probs={int(k): v for k, v in (t.get("per_layer_probs") or {}).items()} if t.get("per_layer_probs") else None,
            ))
        log.prompts.append(pr)
    return log


# ---------------------------
# Example (optional)
# ---------------------------

if __name__ == "__main__":
    prompts = [
        'Review: "This product is amazing!" Sentiment:',
        'Translate to French: "Hello, how are you?"',
    ]

    # 1) Capture routing
    log = capture_routing_for_prompts(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts,
        max_new_tokens=16,
        temperature=0.0,
        with_probs=True,
    )
    save_routing_log(log, "routing_log.json")

    # 2) Summarize
    summ = summarize_routing(log)
    print(f"\nUtilization: {summ.utilization_percent:.2f}% "
          f"({len(summ.unique_pairs)} unique (layer, expert) pairs)")

    # 3) Unused experts
    never = find_never_routed_experts(log, summ)
    for L in sorted(never):
        print(f"Layer {L}: {len(never[L])} never routed experts")

    # 4) Build a 99% mass keep plan (using probs if available) and export
    keep = build_keep_map_from_summary(
        summ,
        num_experts=log.num_experts,
        moe_layers=log.moe_layers,
        coverage=0.99,
        min_keep=log.top_k,            # never keep fewer than top_k
        use_mass_if_available=True,
    )
    # Export (dry run first)
    export_pruned_checkpoint(
        log.model_name,
        keep,
        output_dir="./pruned-qwen-a3b-99",
        strict_uniform=True,
        dry_run=True,                   # set to False after you inspect changes
    )


#################

Considering the task and the ai assistants answer and using your own judgment make a final complete script that accomplishes this goal of analyzing and pruning MoE models for specific tasks.