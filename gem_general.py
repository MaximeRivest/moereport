# general_moe_analyzer.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# 🚀 CONFIGURATION - MODIFY THIS SECTION
# ==============================================================================

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_NEW_TOKENS = 50  # Max tokens to generate for each prompt

# Define the two sets of prompts you want to compare.
# Give each a descriptive label.

PROMPT_SET_A = {
    "label": "🐍 Coding Questions",
    "prompts": [
        "Write a Python function to find the factorial of a number.",
        "In Python, what is the difference between a list and a tuple?",
        "Generate a simple Python script to read a CSV file and print its content.",
        "What is a dictionary comprehension in Python? Provide a short example.",
        "How do you handle exceptions in Python? Show a try-except block."
    ]
}

PROMPT_SET_B = {
    "label": "📜 History Questions",
    "prompts": [
        "Who was the first emperor of Rome and when did he rule?",
        "What were the main causes of the French Revolution?",
        "Describe the significance of the Silk Road in ancient times.",
        "Who was Charlemagne and what was he known for?",
        "What was the Treaty of Westphalia and why is it important?"
    ]
}

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================


def load_model(model_name):
    """Loads the tokenizer and model from Hugging Face."""
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("✅ Model loaded successfully.")
    
    config = {
        "num_experts": model.config.num_routed_experts if hasattr(model.config, 'num_routed_experts') else 128,
        "top_k": model.config.num_experts_per_tok if hasattr(model.config, 'num_experts_per_tok') else 8,
        "num_hidden_layers": model.config.num_hidden_layers
    }
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    return model, tokenizer, config

def process_prompts(model, tokenizer, config, prompts, label):
    """
    Generates responses for a set of prompts and tracks expert usage for every token.
    """
    expert_counts = defaultdict(lambda: defaultdict(int)) # Layer -> Expert -> Count
    generated_texts = []
    
    print("\n" + "="*50)
    print(f"Processing Prompt Set: '{label}' ({len(prompts)} prompts)")
    print("="*50)

    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}...")
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(formatted_text, return_tensors="pt").input_ids.to(model.device)
        
        current_generated_ids = []
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                outputs = model(input_ids=input_ids, output_router_logits=True)
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        if layer_logits is None: continue
                        
                        # --- 💡 FIX IS HERE ---
                        # Handle both 2D (seq_len, num_experts) and 3D (batch, seq_len, num_experts) shapes
                        if layer_logits.dim() == 3:
                            # Shape is [batch, seq_len, num_experts]
                            token_router_logits = layer_logits[0, -1, :]
                        else:
                            # Shape is [seq_len, num_experts]
                            token_router_logits = layer_logits[-1, :]
                        # --- END OF FIX ---
                        
                        top_experts = torch.topk(token_router_logits, config['top_k']).indices.tolist()
                        
                        for expert_id in top_experts:
                            expert_counts[layer_idx][expert_id] += 1
                
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                current_generated_ids.append(next_token_id.item())

        full_response = tokenizer.decode(current_generated_ids, skip_special_tokens=True)
        generated_texts.append(full_response)
        print(f"    Response: {full_response[:100].replace(chr(10), ' ')}...")
        
    return expert_counts, generated_texts

def analyze_expert_specialization(data_A, label_A, data_B, label_B):
    """
    Analyzes and compares expert usage between two prompt sets.
    """
    print("\n\n" + "="*70)
    print("TASK SPECIALIZATION EXPERT ANALYSIS")
    print("="*70)

    # Aggregate expert counts across all layers
    total_A = defaultdict(int)
    for layer_counts in data_A.values():
        for expert, count in layer_counts.items():
            total_A[expert] += count
            
    total_B = defaultdict(int)
    for layer_counts in data_B.values():
        for expert, count in layer_counts.items():
            total_B[expert] += count

    # --- Overall Statistics ---
    experts_A = set(total_A.keys())
    experts_B = set(total_B.keys())
    all_activated_experts = experts_A | experts_B
    
    print("\nOverall Expert Usage:")
    print(f"  • Total unique experts activated across both tasks: {len(all_activated_experts)}")
    print(f"  • Experts used by BOTH tasks: {len(experts_A & experts_B)}")
    print(f"  • Experts used ONLY for '{label_A}': {len(experts_A - experts_B)}")
    print(f"  • Experts used ONLY for '{label_B}': {len(experts_B - experts_A)}")

    # --- Top Experts per Task ---
    print("\n" + "="*50)
    print("Top 10 Experts by Task")
    print("="*50)

    sum_A = sum(total_A.values())
    sorted_A = sorted(total_A.items(), key=lambda x: x[1], reverse=True)
    print(f"\n- Top Experts for '{label_A}':")
    for i, (expert, count) in enumerate(sorted_A[:10], 1):
        percentage = (count / sum_A * 100) if sum_A > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:5d} activations ({percentage:5.2f}%)")

    sum_B = sum(total_B.values())
    sorted_B = sorted(total_B.items(), key=lambda x: x[1], reverse=True)
    print(f"\n- Top Experts for '{label_B}':")
    for i, (expert, count) in enumerate(sorted_B[:10], 1):
        percentage = (count / sum_B * 100) if sum_B > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:5d} activations ({percentage:5.2f}%)")

    # --- Specialization Score Analysis ---
    print("\n" + "="*50)
    print("Expert Specialization Scores")
    print("="*50)
    
    specialization_scores = {}
    for expert in all_activated_experts:
        count_A = total_A.get(expert, 0)
        count_B = total_B.get(expert, 0)
        total_count = count_A + count_B
        
        if total_count > 20:  # Threshold for meaningful analysis
            # Score: +1.0 = purely for A, -1.0 = purely for B
            score = (count_A - count_B) / total_count
            specialization_scores[expert] = {
                'score': score, 'count_A': count_A, 'count_B': count_B
            }

    sorted_specialists = sorted(specialization_scores.items(), key=lambda x: abs(x[1]['score']), reverse=True)
    
    print(f"(+1.0 = Purely '{label_A}', -1.0 = Purely '{label_B}')")
    print("-" * 60)
    for expert, data in sorted_specialists[:15]:
        label_tag = f"[{label_A.split(' ')[0]}]" if data['score'] > 0 else f"[{label_B.split(' ')[0]}]"
        print(f"  Expert {expert:3d}: {data['score']:+6.3f} {label_tag} (A:{data['count_A']:4d}, B:{data['count_B']:4d})")

    return all_activated_experts

def main():
    """Main execution function."""
    model, tokenizer, config = load_model(MODEL_NAME)
    
    # Process both sets of prompts
    expert_data_A, _ = process_prompts(model, tokenizer, config, PROMPT_SET_A['prompts'], PROMPT_SET_A['label'])
    expert_data_B, _ = process_prompts(model, tokenizer, config, PROMPT_SET_B['prompts'], PROMPT_SET_B['label'])

    # Analyze the results
    all_used_experts_across_layers = set()
    if expert_data_A and expert_data_B:
        # The analysis function now returns the set of all activated experts (per layer)
        specialized_experts = analyze_expert_specialization(expert_data_A, PROMPT_SET_A['label'], expert_data_B, PROMPT_SET_B['label'])
        
        # We need to get the experts on a per-layer basis for accurate pruning calculation
        for layer in range(config['num_hidden_layers']):
            experts_in_layer_A = set(expert_data_A.get(layer, {}).keys())
            experts_in_layer_B = set(expert_data_B.get(layer, {}).keys())
            all_used_experts_across_layers.update((layer, exp) for exp in experts_in_layer_A | experts_in_layer_B)
            
    else:
        print("\n⚠️ Insufficient data collected for analysis.")
        return

    # --- VRAM Pruning Potential Summary ---
    print("\n\n" + "=" * 70)
    print("SUMMARY: VRAM OPTIMIZATION POTENTIAL (FOR THESE SPECIFIC TASKS)")
    print("=" * 70)
    
    total_experts_in_model = config['num_hidden_layers'] * config['num_experts']
    total_experts_used = len(all_used_experts_across_layers)
    utilization = (total_experts_used / total_experts_in_model) * 100
    
    print(f"\n  • Total Experts in Full Model: {total_experts_in_model:,}")
    print(f"  • Unique Experts Used for '{PROMPT_SET_A['label']}' & '{PROMPT_SET_B['label']}': {total_experts_used:,}")
    print(f"  • Model Utilization Rate: {utilization:.2f}%")
    print(f"  • 💡 Potential VRAM Savings by Pruning: {100 - utilization:.2f}%")
    print("\n(This means you could theoretically drop the unused experts to run these tasks with a much smaller memory footprint.)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()