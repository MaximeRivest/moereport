import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Model loading
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print("Model loaded.\n")

num_experts_per_layer = 128
top_k = 8
num_hidden_layers = 48

def get_experts_at_position(prompt, position_desc="last"):
    """Get experts activated at a specific position"""
    messages = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            output_router_logits=True,
            return_dict=True
        )
        
        if hasattr(outputs, 'router_logits') and outputs.router_logits:
            # Get the position we're interested in
            if position_desc == "last":
                position = inputs.input_ids.shape[1] - 1
            else:
                position = position_desc
            
            experts_by_layer = {}
            
            for layer_idx in range(min(5, len(outputs.router_logits))):
                if outputs.router_logits[layer_idx] is not None:
                    layer_logits = outputs.router_logits[layer_idx]
                    
                    if len(layer_logits.shape) >= 2:
                        if position < layer_logits.shape[0]:
                            pos_logits = layer_logits[position]
                        else:
                            pos_logits = layer_logits[-1]
                        
                        probs = torch.softmax(pos_logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        experts_by_layer[layer_idx] = sorted(top_experts)
            
            # Also get the predicted token
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            next_token = tokenizer.decode([next_token_id])
            
            return experts_by_layer, next_token, position
    
    return {}, "", -1

print("="*70)
print("DIAGNOSTIC: POSITION-SPECIFIC EXPERT ROUTING")
print("="*70)

# Test 1: Check if the EXACT SAME position with DIFFERENT content uses same experts
print("\nTEST 1: Different prompts, checking position 14 (after 'assistant\\n')")
print("-"*70)

test_prompts = [
    "What is 2+2?",
    "Translate 'hello' to French",
    "Write a poem about cats",
    "Explain quantum physics",
    "What's the capital of France?"
]

position_14_experts = []
for prompt in test_prompts:
    experts, token, pos = get_experts_at_position(prompt, "last")
    print(f"\nPrompt: '{prompt}'")
    print(f"Position {pos} generates: '{token}'")
    print(f"Layer 0 experts: {experts.get(0, [])}")
    position_14_experts.append(experts.get(0, []))

# Check if all are identical
if all(exp == position_14_experts[0] for exp in position_14_experts):
    print("\n⚠️ All prompts use IDENTICAL experts at position 14!")
else:
    print("\n✓ Different prompts use different experts at position 14")

# Test 2: Check if generating MULTIPLE tokens shows variation
print("\n" + "="*70)
print("TEST 2: Generating multiple tokens from same starting point")
print("-"*70)

test_prompt = "Explain how to make a sandwich"
messages = [{"role": "user", "content": test_prompt}]
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

initial_inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True).to(model.device)
print(f"Starting prompt: '{test_prompt}'")
print(f"Initial sequence length: {initial_inputs.input_ids.shape[1]}")

# Generate 5 tokens and track experts at each new position
current_ids = initial_inputs.input_ids
generated_text = ""
position_experts_map = {}

for step in range(5):
    with torch.no_grad():
        outputs = model(
            input_ids=current_ids,
            output_router_logits=True,
            return_dict=True
        )
        
        # Get next token
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        next_token = tokenizer.decode(next_token_id[0].item())
        generated_text += next_token
        
        # Get experts for the generating position
        position = current_ids.shape[1] - 1
        
        if outputs.router_logits and outputs.router_logits[0] is not None:
            layer_logits = outputs.router_logits[0]
            if position < layer_logits.shape[0]:
                pos_logits = layer_logits[position]
            else:
                pos_logits = layer_logits[-1]
            
            probs = torch.softmax(pos_logits, dim=-1)
            top_experts = torch.topk(probs, top_k).indices.tolist()
            position_experts_map[position] = sorted(top_experts)
            
            print(f"\nStep {step+1}: Position {position} -> '{next_token}'")
            print(f"  Experts: {sorted(top_experts)}")
        
        # Append for next iteration
        current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

print(f"\nGenerated: '{generated_text}'")

# Test 3: Check if the SAME tokens in DIFFERENT contexts use different experts
print("\n" + "="*70)
print("TEST 3: Same token sequence, different preceding context")
print("-"*70)

# Create two prompts that will have "assistant\n" at different positions
short_prompt = "Hi"
long_prompt = "This is a much longer prompt that will push the assistant token to a different position in the sequence"

for prompt_name, prompt in [("Short", short_prompt), ("Long", long_prompt)]:
    messages = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True).to(model.device)
    tokens = inputs.input_ids[0].tolist()
    
    print(f"\n{prompt_name} prompt: '{prompt[:30]}...'")
    print(f"Sequence length: {len(tokens)}")
    
    # Find where "assistant" token is
    assistant_token_id = tokenizer.encode("assistant")[0] if len(tokenizer.encode("assistant")) > 0 else 77091
    assistant_positions = [i for i, t in enumerate(tokens) if t == assistant_token_id]
    
    if assistant_positions:
        assistant_pos = assistant_positions[0]
        print(f"'assistant' token at position: {assistant_pos}")
        
        # Get experts at the position after assistant (the \n)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                output_router_logits=True,
                return_dict=True
            )
            
            if outputs.router_logits and outputs.router_logits[0] is not None:
                layer_logits = outputs.router_logits[0]
                if assistant_pos + 1 < layer_logits.shape[0]:
                    pos_logits = layer_logits[assistant_pos + 1]
                    probs = torch.softmax(pos_logits, dim=-1)
                    top_experts = torch.topk(probs, top_k).indices.tolist()
                    print(f"Experts at position {assistant_pos + 1} (after 'assistant'): {sorted(top_experts)}")

# Test 4: Compare expert selection at DIFFERENT absolute positions
print("\n" + "="*70)
print("TEST 4: Expert selection by absolute position")
print("-"*70)

test_prompt = "Write a story"
messages = [{"role": "user", "content": test_prompt}]
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True).to(model.device)

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        output_router_logits=True,
        return_dict=True
    )
    
    if outputs.router_logits and outputs.router_logits[0] is not None:
        layer_logits = outputs.router_logits[0]
        
        print(f"Prompt: '{test_prompt}'")
        print(f"Checking Layer 0 experts at different positions:\n")
        
        positions_to_check = [0, 5, 10, layer_logits.shape[0]-1]
        
        for pos in positions_to_check:
            if pos < layer_logits.shape[0]:
                token_id = inputs.input_ids[0, pos].item() if pos < inputs.input_ids.shape[1] else -1
                token_text = tokenizer.decode([token_id]) if token_id != -1 else "???"
                
                pos_logits = layer_logits[pos]
                probs = torch.softmax(pos_logits, dim=-1)
                top_experts = torch.topk(probs, top_k).indices.tolist()
                
                print(f"Position {pos:2d} ('{token_text:15s}'): {sorted(top_experts)}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)