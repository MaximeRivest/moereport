import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings

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

def analyze_generation_routing(prompts, task_names, max_new_tokens=20):
    """
    Analyze expert routing during multi-token generation for different tasks.
    """
    all_task_experts = {}
    
    for prompt, task_name in zip(prompts, task_names):
        print(f"\n{'='*70}")
        print(f"TASK: {task_name}")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        # Track experts used across all generated tokens
        task_experts_by_layer = defaultdict(set)
        position_experts = defaultdict(lambda: defaultdict(set))
        
        # Generate tokens one by one to track routing
        print(f"\nGenerating {max_new_tokens} tokens...")
        generated_text = ""
        current_input_ids = inputs.input_ids
        
        for token_idx in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    output_router_logits=True,
                    return_dict=True
                )
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                next_token_text = tokenizer.decode(next_token_id[0].item())
                generated_text += next_token_text
                
                # Track routing for this generation step
                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    # We're interested in the routing for the LAST position (the one generating the new token)
                    seq_len = current_input_ids.shape[1]
                    
                    # Analyze first few layers
                    for layer_idx in range(min(5, len(outputs.router_logits))):
                        if outputs.router_logits[layer_idx] is not None:
                            layer_logits = outputs.router_logits[layer_idx]
                            
                            # Get the last position's routing (this determines the next token)
                            if len(layer_logits.shape) >= 2:
                                # The position that generates the new token
                                generating_position = seq_len - 1
                                
                                if len(layer_logits.shape) == 2:  # [seq_len, num_experts]
                                    if generating_position < layer_logits.shape[0]:
                                        pos_logits = layer_logits[generating_position]
                                    else:
                                        pos_logits = layer_logits[-1]
                                else:  # [batch, seq_len, num_experts]
                                    if generating_position < layer_logits.shape[1]:
                                        pos_logits = layer_logits[0, generating_position]
                                    else:
                                        pos_logits = layer_logits[0, -1]
                                
                                # Get top experts for this position
                                probs = torch.softmax(pos_logits, dim=-1)
                                top_experts = torch.topk(probs, top_k).indices.tolist()
                                
                                # Track which experts are used
                                task_experts_by_layer[layer_idx].update(top_experts)
                                position_experts[generating_position][layer_idx].update(top_experts)
                
                # Append generated token to input for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # Stop if we hit end token
                if next_token_id[0].item() == tokenizer.eos_token_id:
                    break
        
        print(f"Generated: '{generated_text[:100]}...'")
        
        # Store results for this task
        all_task_experts[task_name] = {
            'by_layer': dict(task_experts_by_layer),
            'by_position': dict(position_experts),
            'generated_text': generated_text
        }
        
        # Print statistics for this task
        print(f"\nExpert Usage Statistics for {task_name}:")
        print("-" * 40)
        
        for layer_idx in range(min(5, num_hidden_layers)):
            if layer_idx in task_experts_by_layer:
                num_experts = len(task_experts_by_layer[layer_idx])
                expert_list = sorted(list(task_experts_by_layer[layer_idx]))[:20]
                print(f"Layer {layer_idx}: {num_experts} unique experts used")
                print(f"  First 20: {expert_list}")
    
    return all_task_experts

def compare_tasks(task_results):
    """
    Compare expert usage across different tasks.
    """
    print("\n" + "="*70)
    print("CROSS-TASK COMPARISON")
    print("="*70)
    
    task_names = list(task_results.keys())
    
    # Compare layer by layer
    for layer_idx in range(min(5, num_hidden_layers)):
        print(f"\nLayer {layer_idx} Comparison:")
        print("-" * 40)
        
        layer_experts_by_task = {}
        for task_name in task_names:
            if 'by_layer' in task_results[task_name] and layer_idx in task_results[task_name]['by_layer']:
                experts = task_results[task_name]['by_layer'][layer_idx]
                layer_experts_by_task[task_name] = experts
                print(f"{task_name:30s}: {len(experts):3d} unique experts")
        
        # Find overlap and unique experts
        if len(layer_experts_by_task) >= 2:
            all_experts = set()
            for experts in layer_experts_by_task.values():
                all_experts.update(experts)
            
            # Find common experts (used by all tasks)
            common_experts = set.intersection(*layer_experts_by_task.values()) if layer_experts_by_task else set()
            
            print(f"\nTotal unique experts across all tasks: {len(all_experts)}")
            print(f"Experts used by ALL tasks: {len(common_experts)}")
            
            # Find task-specific experts
            for task_name in task_names:
                if task_name in layer_experts_by_task:
                    task_experts = layer_experts_by_task[task_name]
                    other_experts = set()
                    for other_task, other_task_experts in layer_experts_by_task.items():
                        if other_task != task_name:
                            other_experts.update(other_task_experts)
                    
                    unique_to_task = task_experts - other_experts
                    if unique_to_task:
                        print(f"Unique to {task_name}: {len(unique_to_task)} experts - {sorted(list(unique_to_task))[:10]}")
    
    # Calculate overall statistics
    print("\n" + "="*70)
    print("OVERALL EXPERT UTILIZATION")
    print("="*70)
    
    all_experts_used = set()
    for task_name, results in task_results.items():
        for layer_idx, experts in results['by_layer'].items():
            for expert in experts:
                all_experts_used.add((layer_idx, expert))
    
    total_possible = num_hidden_layers * num_experts_per_layer
    utilization_pct = (len(all_experts_used) / total_possible) * 100
    
    print(f"Total unique (layer, expert) pairs used: {len(all_experts_used):,}")
    print(f"Total possible experts: {total_possible:,}")
    print(f"Utilization: {utilization_pct:.2f}%")
    print(f"Unused experts: {total_possible - len(all_experts_used):,} ({100-utilization_pct:.2f}%)")

# Define test prompts for different task types
translation_prompts = [
    "Translate to French: 'The cat is sleeping on the couch'",
    "Translate to French: 'I love eating pizza with my friends'",
    "Translate to French: 'The weather is beautiful today'",
]

python_prompts = [
    "How do I read a CSV file in Python? Show me the code.",
    "How do I create a list comprehension in Python? Give an example.",
    "How do I handle exceptions in Python? Show a try-except block.",
]

r_prompts = [
    "How do I read a CSV file in R? Show me the code.",
    "How do I create a vector in R? Give an example.",
    "How do I handle errors in R? Show a tryCatch example.",
]

# Combine all prompts
all_prompts = translation_prompts + python_prompts + r_prompts
task_names = (
    ["Translation-1", "Translation-2", "Translation-3"] +
    ["Python-1", "Python-2", "Python-3"] +
    ["R-1", "R-2", "R-3"]
)

print("="*70)
print("MULTI-TOKEN GENERATION EXPERT ROUTING ANALYSIS")
print("="*70)
print(f"Analyzing {len(all_prompts)} prompts across 3 task types")
print(f"Generating up to 20 tokens per prompt")

# Analyze routing for all tasks
task_results = analyze_generation_routing(all_prompts, task_names, max_new_tokens=20)

# Compare results across tasks
compare_tasks(task_results)

# Additional analysis: Check if position affects routing more than content
print("\n" + "="*70)
print("POSITION vs CONTENT ANALYSIS")
print("="*70)

# Group results by task type
translation_results = {k: v for k, v in task_results.items() if k.startswith("Translation")}
python_results = {k: v for k, v in task_results.items() if k.startswith("Python")}
r_results = {k: v for k, v in task_results.items() if k.startswith("R")}

print("\nChecking consistency within task types:")
print("-" * 40)

for task_type, type_results in [("Translation", translation_results), 
                                  ("Python", python_results), 
                                  ("R", r_results)]:
    if type_results:
        print(f"\n{task_type} Tasks:")
        
        # Check if same positions use same experts across similar tasks
        for layer_idx in range(min(3, num_hidden_layers)):
            layer_experts_by_task = {}
            
            for task_name, results in type_results.items():
                if 'by_layer' in results and layer_idx in results['by_layer']:
                    layer_experts_by_task[task_name] = results['by_layer'][layer_idx]
            
            if len(layer_experts_by_task) > 1:
                # Check overlap
                common = set.intersection(*layer_experts_by_task.values())
                all_experts = set.union(*layer_experts_by_task.values())
                overlap_pct = (len(common) / len(all_experts) * 100) if all_experts else 0
                
                print(f"  Layer {layer_idx}: {overlap_pct:.1f}% expert overlap between {task_type} tasks")
                print(f"    Common: {len(common)}, Total: {len(all_experts)}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)