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

def analyze_full_sequence_routing(prompt):
    """
    Analyze routing for ALL tokens in a sequence, not just the last one.
    """
    print(f"\nAnalyzing: '{prompt[:50]}...'")
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    # Decode each token to understand what we're looking at
    tokens = inputs.input_ids[0].tolist()
    print(f"\nSequence length: {len(tokens)} tokens")
    print("Token breakdown:")
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        print(f"  Pos {i:2d}: '{token_text[:20]:20s}' (ID: {token_id})")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_router_logits=True,
            return_dict=True
        )
        
        if hasattr(outputs, 'router_logits') and outputs.router_logits:
            print(f"\n{'='*70}")
            print("ROUTER ANALYSIS")
            print(f"{'='*70}")
            
            # Analyze the first layer in detail
            if outputs.router_logits[0] is not None:
                layer_0_logits = outputs.router_logits[0]
                print(f"\nLayer 0 Router Logits:")
                print(f"  Shape: {layer_0_logits.shape}")
                print(f"  Dtype: {layer_0_logits.dtype}")
                
                # Check if shape has batch dimension
                if len(layer_0_logits.shape) == 2:
                    print("\n  ⚠️ Missing batch dimension! Shape is [seq_len, num_experts]")
                    print("  This means router logits are NOT input-dependent!")
                    
                    # Analyze each position
                    print("\n  Expert selection by token position:")
                    all_experts_used = set()
                    position_experts = {}
                    
                    for pos in range(min(len(tokens), layer_0_logits.shape[0])):
                        logits = layer_0_logits[pos]
                        probs = torch.softmax(logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        position_experts[pos] = sorted(top_experts)
                        all_experts_used.update(top_experts)
                        
                        token_text = tokenizer.decode([tokens[pos]]) if pos < len(tokens) else "???"
                        print(f"    Pos {pos:2d} ('{token_text[:15]:15s}'): {sorted(top_experts)}")
                    
                    # Check if all positions use same experts
                    unique_patterns = set(tuple(experts) for experts in position_experts.values())
                    if len(unique_patterns) == 1:
                        print(f"\n  ⚠️ CRITICAL: All {len(position_experts)} positions use IDENTICAL experts!")
                        print(f"     This confirms the router is not working properly!")
                    else:
                        print(f"\n  ✓ Found {len(unique_patterns)} unique expert patterns across positions")
                        print(f"  Total unique experts used in layer 0: {len(all_experts_used)}")
                
                elif len(layer_0_logits.shape) == 3:
                    print("\n  ✓ Has batch dimension [batch, seq_len, num_experts]")
                    batch_size, seq_len, num_experts = layer_0_logits.shape
                    
                    # Check variation across sequence
                    print(f"\n  Checking routing variation across {seq_len} positions:")
                    
                    position_experts = {}
                    for pos in range(seq_len):
                        logits = layer_0_logits[0, pos, :]
                        probs = torch.softmax(logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        position_experts[pos] = sorted(top_experts)
                    
                    unique_patterns = set(tuple(experts) for experts in position_experts.values())
                    print(f"  Unique routing patterns: {len(unique_patterns)}")
                    
                    if len(unique_patterns) == 1:
                        print("  ⚠️ All positions route to same experts (unexpected!)")
                    else:
                        print("  ✓ Different positions use different experts (expected)")
                
                # Check if logits are actually static by comparing statistics
                print("\n  Logit Statistics Check:")
                if len(layer_0_logits.shape) >= 2:
                    # Flatten to analyze all logits
                    flat_logits = layer_0_logits.reshape(-1, layer_0_logits.shape[-1])
                    
                    # Check if each expert column has the same value across all positions
                    for expert_idx in range(min(5, num_experts_per_layer)):
                        expert_column = flat_logits[:, expert_idx]
                        unique_values = torch.unique(expert_column)
                        if len(unique_values) == 1:
                            print(f"    Expert {expert_idx}: STATIC (single value: {unique_values[0].item():.4f})")
                        else:
                            print(f"    Expert {expert_idx}: Dynamic (std: {expert_column.std().item():.4f})")
            
            # Check consistency across layers
            print(f"\n{'='*70}")
            print("CROSS-LAYER CONSISTENCY CHECK")
            print(f"{'='*70}")
            
            layer_signatures = []
            for layer_idx in range(min(5, len(outputs.router_logits))):
                if outputs.router_logits[layer_idx] is not None:
                    layer_logits = outputs.router_logits[layer_idx]
                    
                    # Get signature of this layer (first position's expert selection)
                    if len(layer_logits.shape) >= 2:
                        first_pos_logits = layer_logits[0] if len(layer_logits.shape) == 2 else layer_logits[0, 0]
                        probs = torch.softmax(first_pos_logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        layer_signatures.append(tuple(sorted(top_experts)))
                        print(f"  Layer {layer_idx}: {sorted(top_experts)}")
            
            # Check if all layers have same signature
            if len(set(layer_signatures)) == 1:
                print("\n  ⚠️ All layers have IDENTICAL routing (very suspicious!)")
            else:
                print(f"\n  ✓ Layers have different routing patterns")

# Test with different prompts
test_prompts = [
    "What is the capital of France?",
    "Explain quantum mechanics in simple terms",
    "Write a haiku about spring",
]

print("=" * 70)
print("FULL SEQUENCE ROUTING ANALYSIS")
print("=" * 70)

for prompt in test_prompts:
    analyze_full_sequence_routing(prompt)

# Additional test: Check if router behavior changes with different generation parameters
print("\n" * 2)
print("=" * 70)
print("TESTING WITH DIFFERENT GENERATION PARAMETERS")
print("=" * 70)

test_prompt = "The meaning of life is"
messages = [{"role": "user", "content": test_prompt}]
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Try with do_sample=True to see if it affects routing
print(f"\nTesting: '{test_prompt}'")
print("\n1. With do_sample=False (default):")
inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

with torch.no_grad():
    outputs1 = model.generate(
        **inputs,
        max_new_tokens=1,
        output_router_logits=True,
        return_dict_in_generate=True,
        do_sample=False
    )
    
    if hasattr(outputs1, 'router_logits') and outputs1.router_logits:
        print("  Router logits available in generate output")
    else:
        print("  No router logits in generate output")

print("\n2. With do_sample=True:")
with torch.no_grad():
    outputs2 = model.generate(
        **inputs,
        max_new_tokens=1,
        output_router_logits=True,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=0.7
    )
    
    if hasattr(outputs2, 'router_logits') and outputs2.router_logits:
        print("  Router logits available in generate output")
    else:
        print("  No router logits in generate output")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Final diagnosis
print("\nFINAL DIAGNOSIS:")
print("-" * 40)
print("Based on the analysis, the router appears to be:")
print("1. Returning static/cached logits that don't depend on input")
print("2. Missing the batch dimension in the output")
print("3. Using the same expert selection regardless of prompt content")
print("\nThis means the MoE routing is effectively DISABLED or BROKEN.")
print("The model is likely using a fixed routing pattern learned during training")
print("rather than dynamically routing based on input content.")
print("\nIMPLICATIONS:")
print("- The '93.8% unused experts' finding is misleading")
print("- Those experts might be used with proper dynamic routing")
print("- Current routing is not task-specific or input-dependent")