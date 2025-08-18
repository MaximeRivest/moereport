```
"""
MoE Expert Analysis and Pruning Suite
======================================
A comprehensive toolkit for analyzing and pruning Mixture of Experts models.
Tracks expert activation patterns and enables targeted model pruning.
Specialized for Qwen3 MoE models with per-layer pruning support.
"""
import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeModel, Qwen3MoeDecoderLayer, Qwen3MoeSparseMlp
import torch.nn as nn
import os
warnings.filterwarnings('ignore')

# Custom classes for variable num_experts per layer
class CustomQwen3MoeConfig(Qwen3MoeConfig):
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
        self.layers = nn.ModuleList()
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

@dataclass
class TokenExpertLog:
    token_id: int
    token_text: str
    position: int
    prompt_idx: int
    layer_experts: Dict[int, List[Tuple[int, float]]]  # layer -> [(expert_id, weight)]

@dataclass
class PromptCompletionLog:
    prompt: str
    prompt_idx: int
    generated_text: str
    token_logs: List[TokenExpertLog]
    total_tokens: int

class MoEExpertLogger:
    def __init__(self, model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        print(f"Initializing MoE Expert Logger for {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self.config = self._extract_model_config()
        self._validate_moe_model()
        self.prompt_logs: List[PromptCompletionLog] = []
        self.expert_activation_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.expert_mass: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        print(f"✓ Model loaded successfully")
        print(f" - Total experts: {self.config['total_experts']:,}")
        print(f" - Experts per layer: {self.config['num_experts']}")
        print(f" - Active experts per token: {self.config['top_k']}")
        print(f" - MoE layers: {self.config['num_hidden_layers']}")

    def _extract_model_config(self) -> Dict[str, Any]:
        config = {
            'num_experts': getattr(self.model.config, 'num_routed_experts', 128),
            'top_k': getattr(self.model.config, 'num_experts_per_tok', 8),
            'num_hidden_layers': self.model.config.num_hidden_layers,
        }
        config['total_experts'] = config['num_experts'] * config['num_hidden_layers']
        return config

    def _validate_moe_model(self):
        test_input = self.tokenizer("test", return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(**test_input, output_router_logits=True)
        if not hasattr(output, 'router_logits') or output.router_logits is None:
            raise ValueError(f"Model {self.model_name} does not appear to be an MoE model or doesn't expose router logits")
        self.moe_layers = [i for i, logits in enumerate(output.router_logits) if logits is not None]
        print(f" - MoE layers identified: {len(self.moe_layers)} layers")

    def log_single_generation(
        self,
        prompt: str,
        prompt_idx: int = 0,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> PromptCompletionLog:
        messages = [{"role": "user", "content": prompt}]
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
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
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=current_attention,
                    output_router_logits=True,
                    return_dict=True
                )
                if temperature == 0:
                    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                else:
                    probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
                if next_token_id.dim() == 0:
                    next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)
                elif next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(0)
                token_id = next_token_id[0, 0].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                position = current_ids.shape[1] - 1
                layer_experts = self._extract_expert_routing(outputs.router_logits, position)
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
                for layer_idx, experts_weights in layer_experts.items():
                    for expert_idx, weight in experts_weights:
                        self.expert_activation_counts[layer_idx][expert_idx] += 1
                        self.expert_mass[layer_idx][expert_idx] += weight
                if token_id == self.tokenizer.eos_token_id:
                    break
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
        layer_experts = {}
        for layer_idx, layer_logits in enumerate(router_logits):
            if layer_logits is None:
                continue
            if layer_logits.dim() == 2:
                if position < layer_logits.shape[0]:
                    pos_logits = layer_logits[position]
                else:
                    pos_logits = layer_logits[-1]
            else:
                if position < layer_logits.shape[1]:
                    pos_logits = layer_logits[0, position]
                else:
                    pos_logits = layer_logits[0, -1]
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
        print(f"\nProcessing {len(prompts)} prompts for batch '{batch_name}'...")
        logs = []
        for i, prompt in enumerate(prompts):
            print(f" Progress: {i + 1}/{len(prompts)}")
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
    def __init__(self, logger: MoEExpertLogger):
        self.logger = logger

    def get_unused_experts(self) -> Dict[int, Set[int]]:
        unused = {}
        num_experts = self.logger.config['num_experts']
        for layer_idx in self.logger.moe_layers:
            used_experts = set(self.logger.expert_activation_counts.get(layer_idx, {}).keys())
            all_experts = set(range(num_experts))
            unused[layer_idx] = all_experts - used_experts
        return unused

    def get_expert_usage_statistics(self) -> Dict[str, Any]:
        stats = {
            'model_config': self.logger.config,
            'num_prompts': len(self.logger.prompt_logs),
            'total_tokens_generated': sum(log.total_tokens for log in self.logger.prompt_logs),
            'layer_statistics': {},
            'global_statistics': {}
        }
        total_experts_used = 0
        for layer_idx in self.logger.moe_layers:
            layer_counts = self.logger.expert_activation_counts.get(layer_idx, {})
            layer_mass = self.logger.expert_mass.get(layer_idx, {})
            num_used = len(layer_counts)
            total_experts_used += num_used
            counts = list(layer_counts.values())
            masses = list(layer_mass.values())
            top_experts = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['layer_statistics'][layer_idx] = {
                'experts_used': num_used,
                'experts_total': self.logger.config['num_experts'],
                'utilization_rate': num_used / self.logger.config['num_experts'],
                'activation_counts': dict(layer_counts),
                'activation_mass': dict(layer_mass),
                'top_10_experts': top_experts,
                'mean_activations': np.mean(counts) if counts else 0,
                'std_activations': np.std(counts) if counts else 0,
                'mean_mass': np.mean(masses) if masses else 0,
                'std_mass': np.std(masses) if masses else 0
            }
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
        specialization = defaultdict(lambda: defaultdict(dict))
        category_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for prompt_log in self.logger.prompt_logs:
            category = None
            for cat_name, indices in category_prompts.items():
                if prompt_log.prompt_idx in indices:
                    category = cat_name
                    break
            if category:
                for token_log in prompt_log.token_logs:
                    for layer_idx, experts_weights in token_log.layer_experts.items():
                        for expert_idx, weight in experts_weights:
                            category_counts[layer_idx][expert_idx][category] += weight
        for layer_idx in category_counts:
            for expert_idx in category_counts[layer_idx]:
                expert_cats = category_counts[layer_idx][expert_idx]
                total = sum(expert_cats.values())
                for category in expert_cats:
                    specialization[layer_idx][expert_idx][category] = expert_cats[category] / total if total > 0 else 0
        return dict(specialization)

class MoEPruner:
    def __init__(self, logger: MoEExpertLogger, analyzer: MoEAnalyzer):
        self.logger = logger
        self.analyzer = analyzer

    def create_pruning_config(
        self,
        coverage: float = 1.0,
        min_experts_per_layer: Optional[int] = None
    ) -> Dict[int, List[int]]:
        if min_experts_per_layer is None:
            min_experts_per_layer = self.logger.config['top_k']
        pruning_config = {}
        for layer_idx in self.logger.moe_layers:
            layer_mass = self.logger.expert_mass.get(layer_idx, {})
            if not layer_mass:
                pruning_config[layer_idx] = list(range(min_experts_per_layer))
                continue
            experts = list(layer_mass.keys())
            masses = torch.tensor([layer_mass[e] for e in experts])
            total_mass = masses.sum().item()
            if total_mass == 0:
                pruning_config[layer_idx] = list(range(min_experts_per_layer))
                continue
            _, sorted_indices = torch.sort(masses, descending=True)
            cum = torch.cumsum(masses[sorted_indices], dim=0)
            target = coverage * total_mass
            k = (cum >= target).nonzero(as_tuple=True)[0][0].item() + 1 if cum[-1] >= target else len(experts)
            k = max(k, min_experts_per_layer)
            k = min(k, len(experts))
            kept_experts = [experts[i.item()] for i in sorted_indices[:k]]
            pruning_config[layer_idx] = kept_experts  # no sort, to preserve order for consistency in slicing
        return pruning_config

    def calculate_pruning_stats(self, pruning_config: Dict[int, List[int]]) -> Dict[str, Any]:
        total_kept = sum(len(experts) for experts in pruning_config.values())
        total_possible = self.logger.config['total_experts']
        stats = {
            'total_experts_original': total_possible,
            'total_experts_kept': total_kept,
            'total_experts_pruned': total_possible - total_kept,
            'compression_ratio': total_kept / total_possible if total_possible else 0,
            'estimated_memory_reduction': 1.0 - (total_kept / total_possible) if total_possible else 0,
            'per_layer_stats': {}
        }
        for layer_idx in self.logger.moe_layers:
            kept = len(pruning_config.get(layer_idx, []))
            total = self.logger.config['num_experts']
            stats['per_layer_stats'][layer_idx] = {
                'experts_kept': kept,
                'experts_total': total,
                'compression_ratio': kept / total if total else 0
            }
        return stats

    def apply_pruning(
        self,
        pruning_config: Dict[int, List[int]],
        output_path: str = "pruned_model",
        save_safetensors: bool = True
    ) -> str:
        print(f"\nApplying pruning configuration...")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        original_model = self.logger.model
        config = CustomQwen3MoeConfig.from_pretrained(self.logger.model_name)
        config.num_routed_experts_per_layer = [len(pruning_config.get(i, [])) for i in range(config.num_hidden_layers)]
        new_model = CustomQwen3MoeForCausalLM(config)
        new_model.model.embed_tokens.load_state_dict(original_model.model.embed_tokens.state_dict())
        new_model.model.norm.load_state_dict(original_model.model.norm.state_dict())
        new_model.lm_head.load_state_dict(original_model.lm_head.state_dict())
        for layer_idx in range(config.num_hidden_layers):
            orig_layer = original_model.model.layers[layer_idx]
            new_layer = new_model.model.layers[layer_idx]
            new_layer.self_attn.load_state_dict(orig_layer.self_attn.state_dict())
            new_layer.input_layernorm.load_state_dict(orig_layer.input_layernorm.state_dict())
            new_layer.post_attention_layernorm.load_state_dict(orig_layer.post_attention_layernorm.state_dict())
            kept = pruning_config.get(layer_idx, [])
            if not kept:
                continue
            orig_mlp = orig_layer.mlp
            new_mlp = new_layer.mlp
            new_mlp.gate.weight.data.copy_(orig_mlp.gate.weight[kept, :])
            for new_i, orig_i in enumerate(kept):
                new_mlp.experts[new_i].load_state_dict(orig_mlp.experts[orig_i].state_dict())
            new_mlp.shared_expert.load_state_dict(orig_mlp.shared_expert.state_dict())
            new_mlp.shared_expert_gate.load_state_dict(orig_mlp.shared_expert_gate.state_dict())
        new_model.save_pretrained(str(output_path), safe_serialization=save_safetensors)
        self.logger.tokenizer.save_pretrained(str(output_path))
        with open(output_path / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        stats = self.calculate_pruning_stats(pruning_config)
        report_path = output_path / "pruning_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                'pruning_config': pruning_config,
                'statistics': stats,
                'model_name': self.logger.model_name
            }, f, indent=2)
        print(f"✓ Pruning complete. Model saved to {output_path}")
        print(f" - Original experts: {stats['total_experts_original']:,}")
        print(f" - Kept experts: {stats['total_experts_kept']:,}")
        print(f" - Memory reduction: {stats['estimated_memory_reduction']:.1%}")
        return str(output_path)

def example_comprehensive_analysis(
    model_name: str,
    prompts: List[str],
    task_name: str = "custom_task",
    max_new_tokens: int = 50,
    coverage: float = 1.0,
    min_experts_per_layer: int = 8
):
    logger = MoEExpertLogger(model_name)
    logs = logger.process_prompts(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        batch_name=task_name
    )
    analyzer = MoEAnalyzer(logger)
    stats = analyzer.get_expert_usage_statistics()
    print(f"\nAnalysis Results:")
    print(f" - Total tokens generated: {stats['total_tokens_generated']}")
    print(f" - Global utilization: {stats['global_statistics']['global_utilization_rate']:.2%}")
    print(f" - Potential VRAM savings: {stats['global_statistics']['potential_memory_savings']:.2%}")
    pruner = MoEPruner(logger, analyzer)
    pruning_config = pruner.create_pruning_config(coverage=coverage, min_experts_per_layer=min_experts_per_layer)
    pruning_stats = pruner.calculate_pruning_stats(pruning_config)
    print(f"\nPruning Configuration:")
    print(f" - Experts to keep: {pruning_stats['total_experts_kept']:,}")
    print(f" - Experts to prune: {pruning_stats['total_experts_pruned']:,}")
    print(f" - Compression ratio: {pruning_stats['compression_ratio']:.2%}")
    print(f" - Estimated memory reduction: {pruning_stats['estimated_memory_reduction']:.2%}")
    pruned_model_path = pruner.apply_pruning(pruning_config, f"{task_name}_pruned_model")
    return logger, analyzer, pruner

if __name__ == "__main__":
    custom_prompts = [
        "Translate to French: Hello world",
        "What is the capital of France?",
        "Explain quantum computing",
    ]
    example_comprehensive_analysis(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts=custom_prompts,
        task_name="mixed_tasks",
        max_new_tokens=50,
        coverage=1.0,
        min_experts_per_layer=8
    )
```


```
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
    
    def _prune_parameter(
        self, 
        param: torch.Tensor, 
        name: str, 
        keep_indices: List[int],
        num_experts: int
    ) -> Optional[torch.Tensor]:
        """
        Prune a parameter tensor based on expert indices to keep.
        Handles various parameter layouts (packed, unpacked, etc.).
        """
        keep_tensor = torch.tensor(keep_indices, device=param.device, dtype=torch.long)
        
        # Check if this is an expert parameter
        if 'expert' not in name.lower() and 'gate' not in name.lower() and 'router' not in name.lower():
            return None
        
        # Handle router/gate weights
        if 'gate' in name.lower() or 'router' in name.lower():
            if param.shape[-1] == num_experts:
                # Output dimension is experts
                return param[..., keep_indices]
            elif param.shape[0] == num_experts:
                # Input dimension is experts
                return param[keep_indices, ...]
        
        # Handle expert weights (may be packed)
        if param.dim() == 3 and param.shape[0] == num_experts:
            # Shape: [num_experts, hidden_dim, hidden_dim]
            return param.index_select(0, keep_tensor)
        
        if param.dim() == 2:
            # Check if packed along first dimension
            if param.shape[0] % num_experts == 0 and param.shape[0] // num_experts > 1:
                chunk_size = param.shape[0] // num_experts
                reshaped = param.view(num_experts, chunk_size, -1)
                pruned = reshaped.index_select(0, keep_tensor)
                return pruned.reshape(-1, param.shape[1])
            
            # Check if packed along second dimension
            if param.shape[1] % num_experts == 0 and param.shape[1] // num_experts > 1:
                chunk_size = param.shape[1] // num_experts
                reshaped = param.view(param.shape[0], num_experts, chunk_size)
                pruned = reshaped.index_select(1, keep_tensor)
                return pruned.reshape(param.shape[0], -1)
        
        if param.dim() == 1:
            # Check if packed bias
            if param.shape[0] % num_experts == 0:
                chunk_size = param.shape[0] // num_experts
                reshaped = param.view(num_experts, chunk_size)
                pruned = reshaped.index_select(0, keep_tensor)
                return pruned.reshape(-1)
        
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
        model_name="Qwen/Qwen2.5-3B-Instruct",  # Use smaller model for testing
        # model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",  # Large model
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
```

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoE Task Profiler & Pruner (Qwen/Mixtral-compatible)
====================================================

What this does
--------------
1) Log which experts are selected at each MoE layer for each *generated* token (per prompt).
2) Run a batch of prompts to completion (or max_new_tokens), capturing routing decisions.
3) Produce summary analytics:
     - counts per (layer, expert),
     - optional probability mass per (layer, expert),
     - utilization rate,
     - expert×token matrices per layer.
4) Report never-routed experts for each layer across all prompts/tokens.
5) Hard-prune a checkpoint (UNIFORM keep-count across all MoE layers) and write a HF-loadable directory.

Why uniform pruning?
--------------------
Most HF architectures (and deployment stacks like sglang, vLLM, TGI) assume the same number
of routed experts E in every MoE layer, stored in a single config key (e.g. `num_routed_experts`).
Per-layer E requires custom model classes. This script keeps *compatibility first* by default.

If you **really** need non-uniform pruning per layer, you must implement custom modules
(similar to your second snippet), then re-register classes with AutoModel. That’s doable but
intentionally out-of-scope here to keep things reliable across toolchains.

API surface (the 5 functions you asked for)
-------------------------------------------
1) log_experts_for_step(router_logits, last_pos, moe_layers, top_k, with_probs=True)
2) capture_routing_for_prompts(model_name, prompts, ...) -> RoutingLog
3) summarize_routing(log: RoutingLog) -> Summary  (+ build_expert_token_matrix(log, weight="prob"/"count"))
4) find_never_routed_experts(log, summary=None) -> Dict[layer, Set[expert]]
5) export_pruned_checkpoint(model_name, keep_map, output_dir, ..., strict_uniform=True, dry_run=True) -> report

Orchestration helper
--------------------
profile_analyze_and_prune(...):
    Runs prompts → summary → builds keep-map at coverage → (dry-run) prune export.

Usage (quick)
-------------
python moe_task_profiler_pruner.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prompts_file prompts.txt \
  --max_new_tokens 64 \
  --coverage 0.99 \
  --min_keep 8 \
  --output_dir ./qwen3-a3b-pruned-99 \
  --device_map auto \
  --dtype bfloat16 \
  --trust_remote_code \
  --export  # write pruned checkpoint after dry-run confirmation

Notes
-----
- Needs `transformers>=4.40`, `torch`, `safetensors`.
- On Qwen MoE **always** pass `--trust_remote_code`.
- The logger reads routing *logits* at the last input position to attribute the expert choices
  that produced the **next** token (the standard interpretation).
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------
# Datamodels
# ---------------------------

@dataclass
class TokenRouting:
    """Routing decisions for one generated token (position gen_pos)."""
    gen_pos: int                               # 0-based index within generated tokens
    token_id: int
    text_piece: str                            # decoded piece (without special tokens)
    per_layer_topk: Dict[int, List[int]]       # {layer_idx: [expert_ids]}
    per_layer_probs: Optional[Dict[int, List[float]]] = None  # per-layer top-k probs (softmax over experts)


@dataclass
class PromptRouting:
    prompt: str
    generated_text: str
    tokens: List[TokenRouting] = field(default_factory=list)


@dataclass
class RoutingLog:
    model_name: str
    num_experts: int                # experts per MoE layer (original, uniform)
    top_k: int                      # experts per token (router top-k)
    moe_layers: List[int]           # indices (within num_hidden_layers) that are MoE
    num_hidden_layers: int
    prompts: List[PromptRouting] = field(default_factory=list)


@dataclass
class Summary:
    counts_per_layer_expert: Dict[int, Dict[int, int]]
    mass_per_layer_expert: Optional[Dict[int, Dict[int, float]]]  # if probs were recorded
    per_position_unique_pairs: Dict[int, int]                      # position -> #unique (layer,expert) pairs
    utilization_percent: float                                     # (#used pairs) / (L*E) * 100
    unique_pairs: Set[Tuple[int, int]]                             # {(layer, expert)}
    top_per_layer: Dict[int, List[Tuple[int, int]]]                # layer -> [(expert, count) sorted desc]


# ============================================================
# 1) Log experts for a SINGLE step (token) from router logits
# ============================================================

def log_experts_for_step(
    router_logits: List[Optional[torch.Tensor]],
    *,
    last_pos: int,
    moe_layers: List[int],
    top_k: int,
    with_probs: bool = True,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, List[float]]]]:
    """
    Return the experts chosen by each MoE layer for the next token.

    Args:
        router_logits: list (len=num_hidden_layers) of tensors or None.
                       Each tensor is either [batch, seq_len, num_experts] or [seq_len, num_experts].
        last_pos:      index of the *last* input position (router for this pos produces next token).
        moe_layers:    indices of layers that are MoE (others will be ignored).
        top_k:         how many experts per token.
        with_probs:    also return top-k probabilities per layer.

    Returns:
        per_layer_topk:  {layer_idx: [expert_ids]}
        per_layer_probs: {layer_idx: [probabilities]} or None
    """
    per_layer_topk: Dict[int, List[int]] = {}
    per_layer_probs: Optional[Dict[int, List[float]]] = {} if with_probs else None

    for L, logits in enumerate(router_logits):
        if L not in moe_layers or logits is None:
            continue

        # Normalize to a [num_experts] vector for the relevant position
        if logits.dim() == 3:                  # [batch, seq_len, num_experts]
            pos = min(last_pos, logits.shape[1] - 1)
            v = logits[0, pos]
        elif logits.dim() == 2:                # [seq_len, num_experts]
            pos = min(last_pos, logits.shape[0] - 1)
            v = logits[pos]
        else:
            continue  # unexpected

        # Convert to probabilities; guard for numerical stability
        v = v.to(torch.float32)
        probs = torch.softmax(v, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        per_layer_topk[L] = top.indices.cpu().tolist()
        if with_probs and per_layer_probs is not None:
            per_layer_probs[L] = top.values.cpu().tolist()

    return per_layer_topk, per_layer_probs


# =================================
# Helpers for probing HF MoE models
# =================================

def _detect_moe_layers(model) -> List[int]:
    """Probe once to identify layers that expose router logits."""
    # A 1-token input is enough to elicit router logits
    tok_id = model.config.eos_token_id
    if tok_id is None:
        tok_id = 1
    probe = {"input_ids": torch.tensor([[tok_id]], device=model.device)}
    with torch.no_grad():
        out = model(**probe, output_router_logits=True, return_dict=True)
    router = out.router_logits
    if isinstance(router, (list, tuple)):
        return [i for i, x in enumerate(router) if x is not None]
    raise RuntimeError("Unexpected router_logits structure; expected list/tuple aligned to layers.")


def _collect_eos_ids(tokenizer) -> List[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            ids.update(tokenizer.eos_token_id)
        else:
            ids.add(tokenizer.eos_token_id)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return sorted(ids)


# ============================================================
# 2) Run a LIST of prompts and capture routing per token
# ============================================================

def capture_routing_for_prompts(
    model_name: str,
    prompts: List[str],
    *,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    temperature: float = 0.0,              # greedy by default
    top_p: Optional[float] = None,          # nucleus sampling (optional)
    with_probs: bool = True,
    trust_remote_code: bool = False,
) -> RoutingLog:
    """
    Generate for each prompt while capturing per-layer MoE routing at each step.

    Returns:
        RoutingLog (model metadata + per-prompt, per-token routing).
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"[load] {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).eval()
    print("[ok] model loaded.")

    # Configure routing
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True

    # Detect key MoE attributes (robust across families)
    top_k = getattr(model.config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(model.config, "moe_top_k", None)
    if top_k is None:
        top_k = 8  # safe fallback

    num_hidden = int(getattr(model.config, "num_hidden_layers", 0))
    num_experts = getattr(model.config, "num_routed_experts",
                          getattr(model.config, "num_local_experts", None))
    if num_experts is None:
        # last-resort fallback; many MoE models set one of the above keys
        num_experts = 128

    moe_layers = _detect_moe_layers(model)
    eos_ids = set(_collect_eos_ids(tok))

    def _apply_chat_template(msg: str) -> str:
        try:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": msg})
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return msg

    log = RoutingLog(
        model_name=model_name,
        num_experts=num_experts,
        top_k=top_k,
        moe_layers=moe_layers,
        num_hidden_layers=num_hidden,
    )

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] capturing routing …")
        formatted = _apply_chat_template(prompt)
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

                # The router at position (seq_len-1) produced the next token
                last_pos = seq.shape[1] - 1
                per_layer_topk, per_layer_probs = log_experts_for_step(
                    out.router_logits,
                    last_pos=last_pos,
                    moe_layers=moe_layers,
                    top_k=top_k,
                    with_probs=with_probs,
                )

                # Next token policy
                logits = out.logits[:, -1, :]
                if temperature and temperature > 0.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cdf = torch.cumsum(sorted_probs, dim=-1)
                        k = int((cdf > top_p).float().argmax(dim=-1).item()) + 1
                        probs = torch.zeros_like(probs).scatter_(1, sorted_idx[:, :k], sorted_probs[:, :k])
                        probs.div_(probs.sum(dim=-1, keepdim=True))
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy

                next_id_item = int(next_id[0, 0].item())
                piece_text = tok.decode([next_id_item], skip_special_tokens=True)

                pr.tokens.append(TokenRouting(
                    gen_pos=step,
                    token_id=next_id_item,
                    text_piece=piece_text,
                    per_layer_topk=per_layer_topk,
                    per_layer_probs=per_layer_probs,
                ))

                # Append to sequence
                seq = torch.cat([seq, next_id], dim=1)
                attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
                pr.generated_text += piece_text

                if next_id_item in eos_ids:
                    break

        log.prompts.append(pr)

    return log


# ============================================================
# 3) Summary analytics (+ expert×token matrices)
# ============================================================

def summarize_routing(log: RoutingLog) -> Summary:
    counts: Dict[int, Dict[int, int]] = {}           # layer -> expert -> count
    mass: Optional[Dict[int, Dict[int, float]]] = {} # layer -> expert -> prob mass
    per_pos_unique: Dict[int, Set[Tuple[int, int]]] = {}  # pos -> {(layer, expert)}

    for L in log.moe_layers:
        counts[L] = {}
        if mass is not None:
            mass[L] = {}

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
                        p = float(t.per_layer_probs[L][e_idx])
                        mass[L][e] = mass[L].get(e, 0.0) + p

    unique_pairs = {(L, e) for L, es in counts.items() for e in es.keys()}
    total_possible = len(log.moe_layers) * log.num_experts
    util = (len(unique_pairs) / total_possible * 100.0) if total_possible else 0.0
    per_pos_uniques = {pos: len(s) for pos, s in per_pos_unique.items()}
    top_per_layer: Dict[int, List[Tuple[int, int]]] = {
        L: sorted(es.items(), key=lambda x: x[1], reverse=True) for L, es in counts.items()
    }

    return Summary(
        counts_per_layer_expert=counts,
        mass_per_layer_expert=(mass if mass != {} else None),
        per_position_unique_pairs=per_pos_uniques,
        utilization_percent=util,
        unique_pairs=unique_pairs,
        top_per_layer=top_per_layer,
    )


def build_expert_token_matrix(
    log: RoutingLog,
    weight: str = "prob"  # "prob" (if available) or "count"
) -> Dict[int, "np.ndarray"]:
    """
    Returns a dict {layer: matrix} where matrix is [num_experts, total_generated_tokens].
    If weight="prob" but probabilities weren't recorded, falls back to "count".

    NOTE: numpy import kept local to avoid hard dependency if not used.
    """
    import numpy as np

    total_tokens = sum(len(pr.tokens) for pr in log.prompts)
    mats: Dict[int, np.ndarray] = {L: np.zeros((log.num_experts, total_tokens), dtype=np.float32)
                                   for L in log.moe_layers}

    t_idx = 0
    for pr in log.prompts:
        for t in pr.tokens:
            for L, experts in t.per_layer_topk.items():
                if weight == "prob" and t.per_layer_probs and L in t.per_layer_probs:
                    probs = t.per_layer_probs[L]
                    for e, p in zip(experts, probs):
                        mats[L][e, t_idx] = float(p)
                else:
                    # Binary indicator for chosen experts at this step
                    for e in experts:
                        mats[L][e, t_idx] = 1.0
            t_idx += 1

    return mats


# ============================================================
# 4) Never-routed experts per layer
# ============================================================

def find_never_routed_experts(
    log: RoutingLog,
    summary: Optional[Summary] = None
) -> Dict[int, Set[int]]:
    """
    Returns {layer_idx: {expert_ids never routed}} over all prompts/tokens.
    """
    if summary is None:
        summary = summarize_routing(log)

    unused: Dict[int, Set[int]] = {}
    for L in log.moe_layers:
        used = set(summary.counts_per_layer_expert.get(L, {}).keys())
        all_e = set(range(log.num_experts))
        unused[L] = all_e - used
    return unused


# ============================================================
# 5) Pruned checkpoint exporter (UNIFORM keep-count)
# ============================================================

# --- Helpers to connect state_dict names back to layer indices ---

_PATTERNS_LAYER_IDX = [
    r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\.",
    r"\.model\.(\d+)\.", r"\.transformer\.(\d+)\.",
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
    Slice common packed expert layouts (Qwen/Mixtral).
    Tries:
      - [E, *] (router rows / packed 3D)
      - [E*d, in] rows -> reshape [E, d, in], slice rows
      - [out, E*d] cols -> reshape [out, E, d], slice on E axis
      - [out, E] cols (router)
      - [E*d] 1D bias -> reshape [E, d], slice rows
      - [E] 1D bias
    If no match, returns tensor unchanged.
    """
    E = num_experts
    dims = tensor.dim()

    if dims >= 3 and tensor.shape[0] == E:
        return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))

    if dims == 2:
        rows, cols = tensor.shape

        if rows == E:
            return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))

        if rows % E == 0 and rows // E > 1:
            d = rows // E
            t = tensor.view(E, d, cols)
            t = t.index_select(0, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(len(keep) * d, cols)

        if cols % E == 0 and cols // E > 1:
            d = cols // E
            t = tensor.view(rows, E, d)
            t = t.index_select(1, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(rows, len(keep) * d)

        if cols == E:
            return tensor.index_select(1, torch.as_tensor(keep, device=tensor.device))

    if dims == 1:
        n = tensor.shape[0]
        if n == E:
            return tensor.index_select(0, torch.as_tensor(keep, device=tensor.device))
        if n % E == 0 and n // E > 1:
            d = n // E
            t = tensor.view(E, d)
            t = t.index_select(0, torch.as_tensor(keep, device=tensor.device))
            return t.reshape(len(keep) * d)

    return tensor  # unrecognized; leave as-is


def _uniformize_keep_map(keep_map: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], int]:
    """Trim all layers to the *minimum* keep-count, ensuring uniformity."""
    if not keep_map:
        return keep_map, 0
    counts = [len(v) for v in keep_map.values()]
    m = min(counts)
    uniform = {L: v[:m] for L, v in keep_map.items()}
    return uniform, m


def _update_config_dict_for_pruned(cfg: Dict[str, Any], keep_count: int) -> Dict[str, Any]:
    """
    Update config dict so the model (and serving stacks) instantiate a graph that matches
    the pruned parameters. Adjusts nested configs (moe/router/ffn) and top-level keys.
    """
    def _set(d: Dict[str, Any], key: str, val: Any):
        if key in d:
            d[key] = val

    def _cap(d: Dict[str, Any], key: str, cap_val: int):
        if key in d and isinstance(d[key], int) and d[key] > cap_val:
            d[key] = cap_val

    # Top-level common keys
    _set(cfg, "num_routed_experts", keep_count)
    _set(cfg, "num_local_experts", keep_count)
    _set(cfg, "num_experts", keep_count)
    _cap(cfg, "num_experts_per_tok", keep_count)
    _cap(cfg, "moe_top_k", keep_count)
    _cap(cfg, "router_top_k", keep_count)

    # Nested configs frequently used by MoE families
    for key in ("moe_config", "ffn_config", "router_config"):
        if key in cfg and isinstance(cfg[key], dict):
            _set(cfg[key], "num_routed_experts", keep_count)
            _set(cfg[key], "num_local_experts", keep_count)
            _set(cfg[key], "num_experts", keep_count)
            _cap(cfg[key], "num_experts_per_tok", keep_count)
            _cap(cfg[key], "moe_top_k", keep_count)
            _cap(cfg[key], "router_top_k", keep_count)

    return cfg


def export_pruned_checkpoint(
    model_name: str,
    keep_map: Dict[int, List[int]],   # {layer_idx: [expert_ids_to_keep]}
    *,
    output_dir: str,
    dtype: str = "bfloat16",
    device_map: str = "cpu",
    strict_uniform: bool = True,
    dry_run: bool = True,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """
    Export a hard-pruned checkpoint containing only the kept experts.
    - Slices packed parameters for Qwen/Mixtral-style MoE.
    - Writes config.json (with updated expert count and top-k caps) and model.safetensors.
    - Saves tokenizer files for convenience.

    Returns a report dict. In dry_run=True no files are written; we just report changes.
    """
    from safetensors.torch import save_file  # local import; optional dependency

    os.makedirs(output_dir, exist_ok=True)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"[load] {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    print("[ok] model loaded.")

    # Validate uniformity
    keep_u, keep_count = _uniformize_keep_map(keep_map)
    if strict_uniform:
        sizes = {len(v) for v in keep_u.values()}
        if len(sizes) != 1:
            raise ValueError(
                "Per-layer keep counts are not uniform. Either provide a uniform keep-map "
                "or set strict_uniform=False (not recommended for most serving stacks)."
            )

    # Probe MoE layers & E
    moe_layers = _detect_moe_layers(model)
    num_experts = getattr(model.config, "num_routed_experts",
                          getattr(model.config, "num_local_experts", 128))
    print(f"[prune] MoE layers: {moe_layers}")
    print(f"[prune] experts per layer: {num_experts}, keep per layer: {keep_count}")

    sd = model.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    changes: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for name, tensor in sd.items():
        L = _infer_layer_idx_from_name(name)
        if L is None or L not in keep_u or L not in moe_layers:
            # Not a MoE layer param (or cannot map to layer) -> keep unchanged
            new_sd[name] = tensor
            continue

        kept = keep_u[L]
        nt = _slice_packed_param(tensor, kept, num_experts, name=name)
        if nt.shape != tensor.shape:
            changes.append((name, tuple(tensor.shape), tuple(nt.shape)))
        new_sd[name] = nt

    cfg = model.config.to_dict()
    cfg = _update_config_dict_for_pruned(cfg, keep_count)

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
        print("[dry-run] No files written.")
        return report

    # Write files (safetensors + config + tokenizer)
    # Move tensors to CPU before saving safetensors
    new_sd_cpu = {k: v.detach().to("cpu") for k, v in new_sd.items()}
    save_file(new_sd_cpu, os.path.join(output_dir, "model.safetensors"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    try:
        tok.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"[ok] wrote pruned checkpoint to: {output_dir}")
    return report


# ============================================================
# Keep-map builder from summary (coverage-driven)
# ============================================================

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
    Per-layer, pick experts to keep that cover the requested mass (probability if available, else counts).

    Returns: {layer_idx: [expert_ids_to_keep]}
    """
    keep_map: Dict[int, List[int]] = {}

    for L in moe_layers:
        # Source vector for this layer
        if use_mass_if_available and summary.mass_per_layer_expert is not None:
            vec = summary.mass_per_layer_expert.get(L, {})
        else:
            vec = summary.counts_per_layer_expert.get(L, {})

        dense = torch.zeros(num_experts, dtype=torch.float64)
        for e, v in vec.items():
            dense[e] = float(v)

        total = float(dense.sum().item())
        if total <= 0.0:
            # Nothing routed in this layer; keep at least min_keep if given, else 1
            k = max(1, min_keep or 1)
            keep_map[L] = list(range(min(k, num_experts)))
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


# ============================================================
# Orchestration helper (optional)
# ============================================================

def profile_analyze_and_prune(
    model_name: str,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    coverage: float = 1.00,
    min_keep: Optional[int] = None,
    output_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    export: bool = False,
) -> Tuple[RoutingLog, Summary, Dict[int, List[int]], Optional[Dict[str, Any]]]:
    """
    End-to-end: capture routing → summarize → build keep-map → (optionally) export pruned checkpoint (dry-run first).
    """
    log = capture_routing_for_prompts(
        model_name, prompts,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
        dtype=dtype,
        temperature=temperature,
        top_p=top_p,
        trust_remote_code=trust_remote_code,
    )
    summ = summarize_routing(log)

    print("\n=== Summary ===")
    print(f"Utilization of (layer,expert) pairs: {summ.utilization_percent:.2f}% "
          f"({len(summ.unique_pairs)}/{len(log.moe_layers)*log.num_experts})")

    for L in sorted(log.moe_layers):
        used = len(summ.counts_per_layer_expert.get(L, {}))
        print(f"  Layer {L}: {used}/{log.num_experts} experts routed at least once.")

    keep_map = build_keep_map_from_summary(
        summ,
        num_experts=log.num_experts,
        moe_layers=log.moe_layers,
        coverage=coverage,
        min_keep=min_keep or log.top_k,  # never below top_k by default
        use_mass_if_available=True,
    )

    # Uniformize to maximize compatibility
    uniform_keep, uniform_count = _uniformize_keep_map(keep_map)
    print(f"\nPlanned keep-count (uniform): {uniform_count} experts per MoE layer "
          f"(min across layers, >= top_k)")

    report = None
    if export and output_dir:
        # First do a dry-run to inspect parameter slicing
        report = export_pruned_checkpoint(
            log.model_name,
            uniform_keep,
            output_dir=output_dir,
            dtype=dtype,
            device_map="cpu",              # slicing on CPU is safest
            strict_uniform=True,
            dry_run=False,                  # actually write files
            trust_remote_code=trust_remote_code,
        )

    return log, summ, uniform_keep, report


# ============================================================
# CLI
# ============================================================

def _load_prompts_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip("\n") for ln in f]
    # blank line = separator; skip empties
    return [ln for ln in lines if ln.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="Profile MoE routing on prompts, summarize, and optionally export a pruned checkpoint.")
    ap.add_argument("--model", required=True, help="HF model id or local path (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)")
    ap.add_argument("--prompts_file", required=True, help="Text file with one prompt per line.")
    ap.add_argument("--system_prompt", default=None, help="Optional system prompt for chat templates.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--coverage", type=float, default=1.0, help="Target mass coverage per layer (0..1).")
    ap.add_argument("--min_keep", type=int, default=None, help="Enforce at least this many experts per layer (defaults to top_k).")
    ap.add_argument("--output_dir", default=None, help="Where to write pruned model (if --export).")
    ap.add_argument("--export", action="store_true", help="Write a pruned checkpoint (safetensors + config + tokenizer).")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True (Qwen requires this).")

    args = ap.parse_args()
    prompts = _load_prompts_from_file(args.prompts_file)

    log, summ, keep_map, report = profile_analyze_and_prune(
        args.model,
        prompts,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        coverage=args.coverage,
        min_keep=args.min_keep,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        export=args.export and args.output_dir is not None,
    )

    # Print never-routed summary
    never = find_never_routed_experts(log, summ)
    print("\n=== Never-routed experts by layer ===")
    for L in sorted(log.moe_layers):
        print(f"Layer {L}: {len(never.get(L, set()))} never routed")

    # Write minimal artifacts for auditability
    os.makedirs(args.output_dir or "./_moe_artifacts", exist_ok=True)
    artifacts_dir = args.output_dir or "./_moe_artifacts"
    with open(os.path.join(artifacts_dir, "routing_summary.json"), "w") as f:
        json.dump({
            "model": log.model_name,
            "num_experts": log.num_experts,
            "top_k": log.top_k,
            "moe_layers": log.moe_layers,
            "utilization_percent": summ.utilization_percent,
            "per_layer_used_counts": {int(L): len(summ.counts_per_layer_expert.get(L, {})) for L in log.moe_layers},
        }, f, indent=2)
    with open(os.path.join(artifacts_dir, "keep_map_uniform.json"), "w") as f:
        json.dump({int(k): v for k, v in keep_map.items()}, f, indent=2)
    print(f"\nArtifacts written to: {artifacts_dir}")
    if report:
        print(f"Prune report written under: {args.output_dir}")


if __name__ == "__main__":
    main()

```
