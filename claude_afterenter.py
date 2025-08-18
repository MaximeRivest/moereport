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

def get_second_token_experts(prompt):
    """
    Generate TWO tokens and analyze the routing for the SECOND token
    (skipping the predictable newline)
    """
    messages = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    initial_length = inputs.input_ids.shape[1]
    
    # Generate first token (should be newline)
    with torch.no_grad():
        outputs1 = model(
            input_ids=inputs.input_ids,
            output_router_logits=True,
            return_dict=True
        )
        
        first_token_id = torch.argmax(outputs1.logits[:, -1, :], dim=-1)
        first_token = tokenizer.decode(first_token_id[0].item())
        
        # Get routing for first token generation
        first_token_experts = {}
        if outputs1.router_logits and outputs1.router_logits[0] is not None:
            for layer_idx in range(min(5, len(outputs1.router_logits))):
                if outputs1.router_logits[layer_idx] is not None:
                    layer_logits = outputs1.router_logits[layer_idx]
                    pos_logits = layer_logits[-1] if len(layer_logits.shape) == 2 else layer_logits[0, -1]
                    probs = torch.softmax(pos_logits, dim=-1)
                    top_experts = torch.topk(probs, top_k).indices.tolist()
                    first_token_experts[layer_idx] = sorted(top_experts)
        
        # Now generate second token (the actual content)
        new_input_ids = torch.cat([inputs.input_ids, first_token_id.unsqueeze(0)], dim=1)
        
        outputs2 = model(
            input_ids=new_input_ids,
            output_router_logits=True,
            return_dict=True
        )
        
        second_token_id = torch.argmax(outputs2.logits[:, -1, :], dim=-1)
        second_token = tokenizer.decode(second_token_id[0].item())
        
        # Get routing for second token generation
        second_token_experts = {}
        second_position = new_input_ids.shape[1] - 1
        
        if outputs2.router_logits and outputs2.router_logits[0] is not None:
            for layer_idx in range(min(5, len(outputs2.router_logits))):
                if outputs2.router_logits[layer_idx] is not None:
                    layer_logits = outputs2.router_logits[layer_idx]
                    
                    # Get the position that's generating the second token
                    if len(layer_logits.shape) == 2:
                        if second_position < layer_logits.shape[0]:
                            pos_logits = layer_logits[second_position]
                        else:
                            pos_logits = layer_logits[-1]
                    else:
                        pos_logits = layer_logits[0, -1]
                    
                    probs = torch.softmax(pos_logits, dim=-1)
                    top_experts = torch.topk(probs, top_k).indices.tolist()
                    second_token_experts[layer_idx] = sorted(top_experts)
    
    return first_token, first_token_experts, second_token, second_token_experts, second_position

print("="*70)
print("SECOND TOKEN EXPERT ROUTING ANALYSIS")
print("="*70)
print("Analyzing the token AFTER the initial newline\n")

# Test with diverse prompts
test_prompts = [
    "What is 2+2?",
    "Translate 'hello' to French",
    "Write a poem about cats",
    "Explain quantum physics",
    "What's the capital of France?",
    "How do I cook pasta?",
    "Tell me a joke",
    "What's the weather like?",
    "Solve x^2 + 5x + 6 = 0",
    "List three colors"
]

# Track results
all_first_token_experts = []
all_second_token_experts = []
all_positions = []

print("Analyzing first TWO tokens for each prompt:")
print("-"*70)

for i, prompt in enumerate(test_prompts):
    first_token, first_experts, second_token, second_experts, position = get_second_token_experts(prompt)
    
    print(f"\n{i+1}. Prompt: '{prompt}'")
    print(f"   1st token: '{repr(first_token)}' at position {position-1}")
    print(f"   2nd token: '{repr(second_token)}' at position {position}")
    print(f"   Layer 0 experts for 1st token: {first_experts.get(0, [])}")
    print(f"   Layer 0 experts for 2nd token: {second_experts.get(0, [])}")
    
    all_first_token_experts.append(first_experts.get(0, []))
    all_second_token_experts.append(second_experts.get(0, []))
    all_positions.append(position)

# Analysis of first tokens
print("\n" + "="*70)
print("FIRST TOKEN ANALYSIS (the newline)")
print("-"*70)

unique_first_patterns = set(tuple(exp) for exp in all_first_token_experts if exp)
print(f"Unique expert patterns for first token: {len(unique_first_patterns)}")

if len(unique_first_patterns) == 1:
    print("⚠️ All prompts use IDENTICAL experts for the first token (newline)")
    print(f"   Pattern: {all_first_token_experts[0]}")
else:
    print("✓ Different expert patterns found for first token")
    for pattern in list(unique_first_patterns)[:3]:
        print(f"   Pattern: {list(pattern)}")

# Analysis of second tokens
print("\n" + "="*70)
print("SECOND TOKEN ANALYSIS (actual content)")
print("-"*70)

unique_second_patterns = set(tuple(exp) for exp in all_second_token_experts if exp)
print(f"Unique expert patterns for second token: {len(unique_second_patterns)}")

if len(unique_second_patterns) == 1:
    print("⚠️ All prompts use IDENTICAL experts for the second token!")
    print(f"   Pattern: {all_second_token_experts[0]}")
else:
    print("✓ Different expert patterns found for second token!")
    for i, pattern in enumerate(list(unique_second_patterns)[:5]):
        # Find which prompts use this pattern
        prompts_with_pattern = [test_prompts[j] for j, exp in enumerate(all_second_token_experts) 
                                if tuple(exp) == pattern]
        print(f"\n   Pattern {i+1}: {list(pattern)}")
        print(f"   Used by: {prompts_with_pattern[:3]}")

# Check if it's purely position-based
print("\n" + "="*70)
print("POSITION DEPENDENCY CHECK")
print("-"*70)

position_to_experts = defaultdict(list)
for pos, experts in zip(all_positions, all_second_token_experts):
    position_to_experts[pos].append(tuple(experts))

print(f"Positions where second token was generated: {set(all_positions)}")

for pos in sorted(position_to_experts.keys()):
    patterns = position_to_experts[pos]
    unique_at_pos = len(set(patterns))
    print(f"\nPosition {pos}:")
    print(f"  Number of different prompts at this position: {len(patterns)}")
    print(f"  Number of unique expert patterns: {unique_at_pos}")
    if unique_at_pos == 1:
        print(f"  ⚠️ All prompts at position {pos} use the same experts: {list(patterns[0])}")
    else:
        print(f"  ✓ Different patterns found at position {pos}")

# Additional test: Generate more tokens to see when diversity appears
print("\n" + "="*70)
print("EXTENDED GENERATION TEST")
print("-"*70)
print("Generating 5 tokens for one prompt to see routing evolution:\n")

test_prompt = "Explain how computers work"
messages = [{"role": "user", "content": test_prompt}]
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

current_ids = tokenizer(formatted_text, return_tensors="pt", truncation=True).to(model.device).input_ids
generated_tokens = []
token_experts = []

print(f"Prompt: '{test_prompt}'")
print(f"Generating tokens...\n")

for step in range(5):
    with torch.no_grad():
        outputs = model(
            input_ids=current_ids,
            output_router_logits=True,
            return_dict=True
        )
        
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        next_token = tokenizer.decode(next_token_id[0].item())
        generated_tokens.append(next_token)
        
        position = current_ids.shape[1] - 1
        
        if outputs.router_logits and outputs.router_logits[0] is not None:
            layer_logits = outputs.router_logits[0]
            if position < layer_logits.shape[0]:
                pos_logits = layer_logits[position]
            else:
                pos_logits = layer_logits[-1]
            
            probs = torch.softmax(pos_logits, dim=-1)
            top_experts = torch.topk(probs, top_k).indices.tolist()
            token_experts.append(sorted(top_experts))
            
            print(f"Token {step+1}: '{repr(next_token)}' at position {position}")
            print(f"         Experts: {sorted(top_experts)}")
        
        current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

print(f"\nGenerated text: '{''.join(generated_tokens)}'")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)