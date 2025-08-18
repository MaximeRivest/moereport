import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, Qwen2Config
from transformers.models.qwen2_moe import Qwen2MoEForCausalLM, Qwen2MoEModel, Qwen2MoEDecoderLayer, Qwen2SparseMlp
from transformers.models.qwen2 import Qwen2MLP
from collections import defaultdict
from typing import List, Dict, Tuple
from torch.nn import Linear, ModuleList, Sigmoid
import os
import json

# Custom classes for variable num_experts per layer

class CustomQwen2MoEConfig(PretrainedConfig):
    model_type = "qwen2_moe"  # To match original for loading
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_routed_experts_per_layer = kwargs.pop("num_routed_experts_per_layer", [self.num_routed_experts] * self.num_hidden_layers)

class CustomQwen2SparseMlp(Qwen2SparseMlp):
    def __init__(self, config, num_experts=None):
        if num_experts is not None:
            config.num_routed_experts = num_experts
        super().__init__(config)

class CustomQwen2MoEDecoderLayer(Qwen2MoEDecoderLayer):
    def __init__(self, config, layer_idx, num_experts=None):
        super().__init__(config, layer_idx)
        if num_experts is not None:
            self.mlp = CustomQwen2SparseMlp(config, num_experts=num_experts)

class CustomQwen2MoEModel(Qwen2MoEModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            num_exp = config.num_routed_experts_per_layer[layer_idx]
            self.layers.append(CustomQwen2MoEDecoderLayer(config, layer_idx, num_experts=num_exp))
        self.post_init()

class CustomQwen2MoEForCausalLM(Qwen2MoEForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen2MoEModel(config)
        self.post_init()

# Register for Auto loading
AutoModelForCausalLM.register(CustomQwen2MoEConfig, CustomQwen2MoEForCausalLM)

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
    config = CustomQwen2MoEConfig.from_pretrained(model_name)
    config.num_routed_experts_per_layer = [len(per_layer_kept.get(i, [])) for i in range(config.num_hidden_layers)]
    new_model = CustomQwen2MoEForCausalLM(config)

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