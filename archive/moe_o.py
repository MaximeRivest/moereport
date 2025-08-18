import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Model setup (use FP8 for VRAM efficiency; ~30GB footprint)
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Or original: "Qwen/Qwen3-30B-A3B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Multi-GPU
    torch_dtype=torch.float8_e4m3fn if "FP8" in model_name else torch.bfloat16,  # FP8 or bfloat16
)

# Assuming model is already loaded
# model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(...)

# Get model configuration
num_experts = model.config.num_routed_experts if hasattr(model.config, 'num_routed_experts') else 128
top_k = model.config.num_experts_per_tok if hasattr(model.config, 'num_experts_per_tok') else 8
num_hidden_layers = model.config.num_hidden_layers

print(f"Model Configuration:")
print(f"  - Number of experts: {num_experts}")
print(f"  - Experts per token: {top_k}")
print(f"  - Hidden layers: {num_hidden_layers}")
print()

# First, let's explore the model structure to find MoE layers
def explore_model_structure():
    """Explore model structure to find MoE-related modules"""
    print("Exploring Model Structure...")
    print("-" * 70)
    
    all_modules = []
    moe_related = []
    
    for name, module in model.named_modules():
        all_modules.append(name)
        # Check for various possible MoE layer names
        if any(keyword in name.lower() for keyword in ['moe', 'expert', 'gate', 'router']):
            moe_related.append(name)
            
    # Print first few module names to understand structure
    print(f"Total modules: {len(all_modules)}")
    print(f"\nFirst 20 module names:")
    for name in all_modules[:20]:
        print(f"  - {name}")
        
    # Look for patterns in layer names
    layer_patterns = set()
    for name in all_modules:
        if 'layers' in name:
            parts = name.split('.')
            if len(parts) > 2:
                layer_patterns.add('.'.join(parts[:3]))
    
    print(f"\nLayer patterns found:")
    for pattern in sorted(layer_patterns)[:10]:
        print(f"  - {pattern}")
    
    # Check specific layer structure
    print(f"\nChecking a specific layer (layer 0):")
    for name in all_modules:
        if 'layers.0.' in name and 'layers.0.self' not in name:
            print(f"  - {name}")
            if len([n for n in all_modules if 'layers.0.' in n]) > 15:
                print("  ... (truncated)")
                break
    
    if moe_related:
        print(f"\nMoE-related modules found:")
        for name in moe_related[:10]:
            print(f"  - {name}")
    else:
        print(f"\nNo obvious MoE module names found. Checking for mlp/ffn variations...")
        mlp_modules = [name for name in all_modules if any(k in name.lower() for k in ['mlp', 'ffn', 'feed'])]
        if mlp_modules[:5]:
            print("MLP/FFN modules (first 5):")
            for name in mlp_modules[:5]:
                print(f"  - {name}")
    
    return all_modules, moe_related

# Explore the model
all_modules, moe_modules = explore_model_structure()

def track_with_forward_pass(prompts, max_new_tokens=30):
    """
    Track expert routing by doing forward passes and examining outputs.
    """
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    
    print("\nAttempting forward pass tracking...")
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"  Processing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt[:50]}...'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        input_ids = inputs.input_ids
        
        # Generate tokens one by one
        for step in range(min(max_new_tokens, 10)):  # Limit for testing
            with torch.no_grad():
                # Try different approaches to get router information
                try:
                    # Approach 1: Standard forward with output_router_logits
                    outputs = model(
                        input_ids=input_ids,
                        output_router_logits=True,
                        return_dict=True
                    )
                    
                    # Check what's in the outputs
                    if hasattr(outputs, 'router_logits') and outputs.router_logits:
                        print(f"    Step {step}: Found router_logits")
                        for layer_idx, logits in enumerate(outputs.router_logits):
                            if logits is not None:
                                # Handle the tensor properly
                                if len(logits.shape) == 3:  # (batch, seq, experts)
                                    logits = logits[:, -1, :]  # Get last token
                                elif len(logits.shape) == 2:  # (batch, experts)
                                    pass
                                else:
                                    continue
                                    
                                probs = torch.softmax(logits[0], dim=-1)
                                k = min(top_k, len(probs))
                                top_experts = torch.topk(probs, k).indices.tolist()
                                
                                for exp in top_experts:
                                    expert_counts[layer_idx][exp] += 1
                                total_selections[layer_idx] += k
                    
                except Exception as e:
                    if step == 0:  # Only print error once
                        print(f"    Error in forward pass: {e}")
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    return expert_counts, total_selections

def inspect_single_forward():
    """
    Do a single forward pass and inspect the output structure in detail.
    """
    print("\nDetailed Single Forward Pass Inspection")
    print("-" * 70)
    
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Try with output_router_logits
        try:
            outputs = model(**inputs, output_router_logits=True, return_dict=True)
            
            print("Output attributes:")
            for attr in dir(outputs):
                if not attr.startswith('_'):
                    value = getattr(outputs, attr)
                    if value is not None and not callable(value):
                        if isinstance(value, torch.Tensor):
                            print(f"  - {attr}: Tensor with shape {value.shape}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"  - {attr}: {type(value).__name__} with {len(value)} items")
                            if isinstance(value[0], torch.Tensor):
                                print(f"      First item shape: {value[0].shape if value[0] is not None else 'None'}")
                        else:
                            print(f"  - {attr}: {type(value).__name__}")
        except Exception as e:
            print(f"Error with output_router_logits=True: {e}")
        
        # Try without output_router_logits
        print("\nTrying without output_router_logits...")
        try:
            outputs = model(**inputs, return_dict=True)
            print("Success! Output type:", type(outputs))
            
            # Check if model has any router/gate related attributes
            print("\nChecking model attributes for routing info:")
            for attr in ['router_logits', 'gate_logits', 'expert_weights', 'routing_weights']:
                if hasattr(model, attr):
                    print(f"  Found: model.{attr}")
                    
        except Exception as e:
            print(f"Error: {e}")

def manual_hook_tracking(prompts, max_new_tokens=30):
    """
    Use hooks to track routing decisions at the MLP/layer level.
    """
    print("\nSetting up hook-based tracking...")
    
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    layer_outputs = {}
    
    def create_hook(layer_name):
        def hook_fn(module, input, output):
            # Store output for analysis
            layer_outputs[layer_name] = {
                'input': input,
                'output': output,
                'module': module
            }
            
            # Check for any routing-related information
            if hasattr(output, 'router_logits'):
                print(f"  Found router_logits in {layer_name}")
            if hasattr(module, 'gate') or hasattr(module, 'router'):
                print(f"  Found gate/router in {layer_name}")
                
        return hook_fn
    
    # Register hooks on layers that might have MoE
    hooks = []
    target_layers = []
    
    # Try to hook into layer blocks
    for name, module in model.named_modules():
        if 'layers.' in name and name.count('.') == 2:  # e.g., model.layers.0
            hooks.append(module.register_forward_hook(create_hook(name)))
            target_layers.append(name)
            if len(target_layers) >= 3:  # Just hook first 3 layers for testing
                break
    
    print(f"Registered hooks on {len(hooks)} layers")
    
    try:
        # Run a forward pass
        prompt = prompts[0] if prompts else "Hello, world!"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            
        print(f"\nCaptured outputs from {len(layer_outputs)} layers")
        
        # Analyze captured outputs
        for layer_name, data in list(layer_outputs.items())[:3]:
            print(f"\n{layer_name}:")
            output = data['output']
            if isinstance(output, tuple):
                print(f"  Output is tuple with {len(output)} elements")
                for i, elem in enumerate(output):
                    if isinstance(elem, torch.Tensor):
                        print(f"    Element {i}: Tensor shape {elem.shape}")
            elif isinstance(output, torch.Tensor):
                print(f"  Output is Tensor with shape {output.shape}")
            else:
                print(f"  Output type: {type(output)}")
                
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return expert_counts, total_selections



# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING QWEN3 MOE ANALYSIS")
    print("=" * 70)
    
    # First inspect the model structure
    inspect_single_forward()
    
    # Try tracking
    test_prompts = [
        "What is 2 + 2?",
        "Hello, how are you?",
    ]
    
    print("\n" + "=" * 70)
    print("ATTEMPTING EXPERT TRACKING")
    print("=" * 70)
    
    # Try forward pass tracking
    counts, totals = track_with_forward_pass(test_prompts, max_new_tokens=10)
    
    if counts:
        print(f"\nSuccessfully tracked {len(counts)} layers")
        for layer_idx in sorted(counts.keys())[:3]:
            print(f"  Layer {layer_idx}: {len(counts[layer_idx])} unique experts used")
    else:
        print("\nForward pass tracking didn't capture expert data")
        print("Trying hook-based approach...")
        counts, totals = manual_hook_tracking(test_prompts, max_new_tokens=10)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)