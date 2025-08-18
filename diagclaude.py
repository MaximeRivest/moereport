import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Model loading
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
print(f"Loading model: {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float8_e4m3fn if "FP8" in model_name else torch.bfloat16,
    )
    print("Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Get model configuration
num_experts_per_layer = model.config.num_routed_experts if hasattr(model.config, 'num_routed_experts') else 128
top_k = model.config.num_experts_per_tok if hasattr(model.config, 'num_experts_per_tok') else 8
num_hidden_layers = model.config.num_hidden_layers

print(f"\nModel Configuration:")
print(f"  - Hidden layers: {num_hidden_layers}")
print(f"  - Experts per layer: {num_experts_per_layer}")
print(f"  - Experts activated per token (Top-K): {top_k}")
print()

def diagnose_router_behavior(prompts, prompt_names):
    """
    Diagnose router behavior with detailed inspection
    """
    print("=" * 70)
    print("ROUTER DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    all_activations = {}
    
    for prompt_idx, (prompt, name) in enumerate(zip(prompts, prompt_names)):
        print(f"\n[Prompt {prompt_idx+1}] {name}")
        print(f"Content: '{prompt[:50]}...'")
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        # Show input shape
        print(f"Input shape: {inputs.input_ids.shape}")
        
        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_router_logits=True,
                    return_dict=True
                )
                
                # Get predicted token
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                predicted_text = tokenizer.decode([next_token_id]).strip()
                print(f"Predicted token: '{predicted_text}'")
                
                # Diagnose router logits
                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    print(f"Router logits available: Yes")
                    print(f"Number of layers with logits: {len(outputs.router_logits)}")
                    
                    # Analyze first few layers in detail
                    layer_activations = {}
                    
                    for layer_idx in range(min(3, len(outputs.router_logits))):
                        layer_logits = outputs.router_logits[layer_idx]
                        
                        if layer_logits is None:
                            print(f"  Layer {layer_idx}: None")
                            continue
                        
                        print(f"\n  Layer {layer_idx}:")
                        print(f"    Shape: {layer_logits.shape}")
                        
                        # Handle different shapes
                        if len(layer_logits.shape) == 3:  # (batch, seq, experts)
                            # Check multiple positions
                            seq_len = layer_logits.shape[1]
                            print(f"    Sequence length: {seq_len}")
                            
                            # Check first, middle, and last token positions
                            positions_to_check = [0, seq_len//2, -1]
                            
                            for pos in positions_to_check:
                                pos_logits = layer_logits[0, pos, :]
                                probs = torch.softmax(pos_logits, dim=-1)
                                top_probs, top_indices = torch.topk(probs, min(top_k, 10))
                                
                                print(f"    Position {pos}:")
                                print(f"      Top experts: {top_indices.tolist()}")
                                print(f"      Top probs: {[f'{p:.3f}' for p in top_probs.tolist()]}")
                                
                                # Check if logits vary
                                logit_std = pos_logits.std().item()
                                logit_min = pos_logits.min().item()
                                logit_max = pos_logits.max().item()
                                print(f"      Logit stats: min={logit_min:.2f}, max={logit_max:.2f}, std={logit_std:.2f}")
                                
                            # Store last position for comparison
                            last_pos_logits = layer_logits[0, -1, :]
                            probs = torch.softmax(last_pos_logits, dim=-1)
                            top_experts = torch.topk(probs, top_k).indices.tolist()
                            layer_activations[layer_idx] = sorted(top_experts)
                            
                        elif len(layer_logits.shape) == 2:  # (batch, experts)
                            pos_logits = layer_logits[0]
                            probs = torch.softmax(pos_logits, dim=-1)
                            top_probs, top_indices = torch.topk(probs, min(top_k, 10))
                            
                            print(f"    Top experts: {top_indices.tolist()}")
                            print(f"    Top probs: {[f'{p:.3f}' for p in top_probs.tolist()]}")
                            
                            # Check logit statistics
                            logit_std = pos_logits.std().item()
                            logit_min = pos_logits.min().item()
                            logit_max = pos_logits.max().item()
                            print(f"    Logit stats: min={logit_min:.2f}, max={logit_max:.2f}, std={logit_std:.2f}")
                            
                            top_experts = torch.topk(probs, top_k).indices.tolist()
                            layer_activations[layer_idx] = sorted(top_experts)
                    
                    all_activations[name] = layer_activations
                    
                    # Check if router logits are actually changing
                    print("\n  Router Logit Variation Check:")
                    if len(outputs.router_logits) > 0 and outputs.router_logits[0] is not None:
                        first_layer_logits = outputs.router_logits[0]
                        if len(first_layer_logits.shape) == 3:
                            # Compare first and last position
                            if first_layer_logits.shape[1] > 1:
                                first_pos = first_layer_logits[0, 0, :]
                                last_pos = first_layer_logits[0, -1, :]
                                diff = (first_pos - last_pos).abs().mean().item()
                                print(f"    Mean absolute difference between first and last position: {diff:.4f}")
                                if diff < 0.001:
                                    print("    ⚠️ WARNING: Router logits appear static across positions!")
                else:
                    print("Router logits NOT available!")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
    
    # Compare activations across prompts
    print("\n" + "=" * 70)
    print("CROSS-PROMPT COMPARISON")
    print("=" * 70)
    
    if len(all_activations) > 1:
        prompt_names_list = list(all_activations.keys())
        
        for layer_idx in range(min(3, num_hidden_layers)):
            print(f"\nLayer {layer_idx} expert selection across prompts:")
            
            for name in prompt_names_list:
                if name in all_activations and layer_idx in all_activations[name]:
                    experts = all_activations[name][layer_idx]
                    print(f"  {name:30s}: {experts}")
            
            # Check if all prompts use same experts
            layer_experts_sets = []
            for name in prompt_names_list:
                if name in all_activations and layer_idx in all_activations[name]:
                    layer_experts_sets.append(set(all_activations[name][layer_idx]))
            
            if len(layer_experts_sets) > 1:
                if all(s == layer_experts_sets[0] for s in layer_experts_sets):
                    print("  ⚠️ WARNING: All prompts activate IDENTICAL experts at this layer!")
                else:
                    union = set.union(*layer_experts_sets)
                    intersection = set.intersection(*layer_experts_sets)
                    print(f"  Total unique experts used: {len(union)}")
                    print(f"  Experts used by ALL prompts: {sorted(list(intersection))}")

# Test with diverse prompts
test_prompts = [
    "What is 2+2?",
    "Translate 'hello' to French",
    "Explain quantum physics",
    "Write a poem about cats",
    "What's the capital of France?",
]

prompt_names = [
    "Simple Math",
    "Translation",
    "Complex Science",
    "Creative Writing",
    "Geography",
]

print("\n" + "=" * 70)
print("TESTING WITH DIVERSE PROMPTS")
print("=" * 70)

diagnose_router_behavior(test_prompts, prompt_names)

# Additional diagnostic: Check if we're looking at the right token position
print("\n" + "=" * 70)
print("TOKEN POSITION DIAGNOSTIC")
print("=" * 70)

test_prompt = "The capital of France is"
messages = [{"role": "user", "content": test_prompt}]
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
print(f"Test prompt: '{test_prompt}'")
print(f"Tokenized length: {inputs.input_ids.shape[1]}")

# Decode tokens to see what positions correspond to what
tokens = inputs.input_ids[0].tolist()
for i, token_id in enumerate(tokens[-10:]):  # Show last 10 tokens
    token_text = tokenizer.decode([token_id])
    print(f"  Position {len(tokens)-10+i}: '{token_text}' (ID: {token_id})")

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_router_logits=True,
        return_dict=True
    )
    
    if hasattr(outputs, 'router_logits') and outputs.router_logits and outputs.router_logits[0] is not None:
        layer_0_logits = outputs.router_logits[0]
        print(f"\nLayer 0 router logits shape: {layer_0_logits.shape}")
        
        if len(layer_0_logits.shape) == 3:
            print("Checking if different token positions have different expert selections:")
            
            # Sample a few positions
            num_positions = layer_0_logits.shape[1]
            positions_to_check = [0, num_positions//4, num_positions//2, 3*num_positions//4, -1]
            
            position_experts = []
            for pos in positions_to_check:
                logits = layer_0_logits[0, pos, :]
                probs = torch.softmax(logits, dim=-1)
                top_experts = torch.topk(probs, top_k).indices.tolist()
                position_experts.append(set(top_experts))
                print(f"  Position {pos}: {sorted(top_experts)}")
            
            # Check if all positions have same experts
            if all(s == position_experts[0] for s in position_experts):
                print("\n⚠️ CRITICAL: All token positions select the SAME experts!")
                print("This suggests the router might be:")
                print("  1. Not properly trained/initialized")
                print("  2. Using cached/static routing")
                print("  3. Bug in router logit extraction")
            else:
                print("\n✓ Different positions select different experts (expected behavior)")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)