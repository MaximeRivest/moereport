import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Model loading section (unchanged)
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
total_experts_in_model = num_hidden_layers * num_experts_per_layer

print(f"\nModel Configuration:")
print(f"  - Hidden layers: {num_hidden_layers}")
print(f"  - Experts per layer: {num_experts_per_layer}")
print(f"  - Experts activated per token (Top-K): {top_k}")
print(f"  - TOTAL unique experts in model: {total_experts_in_model:,}")
print()

# Sports equipment review prompts
positive_reviews = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "List three programming languages.",
    "Translate 'Good morning' into Spanish.",
    "What is 12 multiplied by 8?",
    "Give me a synonym for 'happy'.",
    "Summarize the plot of Cinderella.",
    "Who wrote 'Pride and Prejudice'?",
    "Explain gravity like I’m five.",
    "List five countries in Africa.",
    "What’s the square root of 144?",
    "Name three types of clouds.",
    "Write a haiku about the ocean.",
    "Who was the first person on the moon?",
    "What’s the opposite of 'cold'?",
    "Give me a fun fact about cats.",
    "Explain what an atom is.",
    "List the planets in our solar system.",
    "What year did World War II end?",
    "Describe how rainbows form.",
    "Convert 10 kilometers to miles.",
    "What is the fastest land animal?",
    "Tell me a joke about computers.",
    "Explain the difference between a comet and an asteroid.",
    "What is the boiling point of water in Celsius?",
    "Name the largest mammal on Earth.",
    "Give me three examples of renewable energy.",
    "What is the Pythagorean theorem?",
    "Who painted the Mona Lisa?",
    "Explain the difference between a noun and a verb."
]

negative_reviews = [
    "Who invented the telephone?",
    "Define the word 'ecosystem'.",
    "Name the tallest mountain in the world.",
    "Explain what DNA does.",
    "List three primary colors.",
    "What is the capital city of Japan?",
    "Give me an example of a simile.",
    "How many continents are there?",
    "What is 7 to the power of 3?",
    "Tell me a riddle about animals.",
    "Who discovered penicillin?",
    "Explain what a black hole is.",
    "List three instruments in an orchestra.",
    "What is the chemical symbol for gold?",
    "Write a short tongue twister.",
    "Name a country where Portuguese is spoken.",
    "What does HTML stand for?",
    "What are three states of matter?",
    "Who is the author of '1984'?",
    "Explain what recycling is.",
    "What is the currency of the UK?",
    "Give me a proverb about patience.",
    "What’s the capital of Brazil?",
    "Explain what tides are.",
    "Name a famous landmark in Egypt.",
    "What is the largest organ in the human body?",
    "Tell me a fun fact about dolphins.",
    "What is the difference between fiction and nonfiction?",
    "How many degrees are in a circle?",
    "Name one invention by Thomas Edison."
]

def track_expert_activation_patterns(prompts, prompt_type="unknown"):
    """
    Track which specific experts are activated for each prompt.
    Returns detailed activation patterns per prompt and per layer.
    """
    # Structure: activation_patterns[prompt_idx][layer_idx] = list of expert indices
    activation_patterns = defaultdict(lambda: defaultdict(list))
    predictions = []
    
    print(f"  Processing {len(prompts)} {prompt_type} reviews...")
    
    for prompt_idx, prompt in enumerate(prompts):
        if (prompt_idx + 1) % 5 == 0:
            print(f"    Processed {prompt_idx + 1}/{len(prompts)} reviews...")
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

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
                predictions.append(predicted_text)
                
                # Track expert activations for this prompt
                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        if layer_logits is None:
                            continue
                        
                        # Get routing decision for last token
                        if len(layer_logits.shape) == 3:
                            last_token_logits = layer_logits[0, -1, :]
                        elif len(layer_logits.shape) == 2:
                            last_token_logits = layer_logits[0]
                        else:
                            continue
                        
                        # Get top-k experts for this layer
                        probs = torch.softmax(last_token_logits, dim=-1)
                        k = min(top_k, len(probs))
                        top_experts = torch.topk(probs, k).indices.tolist()
                        
                        # Store which experts were activated
                        activation_patterns[prompt_idx][layer_idx] = sorted(top_experts)
                        
            except Exception as e:
                if prompt_idx == 0:
                    print(f"    Note: Error during tracking - {str(e)[:50]}")
                predictions.append("Error")
    
    return activation_patterns, predictions

def analyze_activation_consistency(pos_patterns, neg_patterns, pos_predictions, neg_predictions):
    """
    Analyze which specific experts are activated and their consistency across prompts.
    """
    print("\n" + "=" * 70)
    print("DETAILED EXPERT ACTIVATION ANALYSIS")
    print("=" * 70)
    
    # 1. Accuracy Check
    print("\nPrediction Accuracy Check:")
    print("-" * 40)
    pos_correct = sum(1 for p in pos_predictions if 'yes' in p.lower())
    neg_correct = sum(1 for p in neg_predictions if 'no' in p.lower())
    total_predictions = len(pos_predictions) + len(neg_predictions)
    accuracy = ((pos_correct + neg_correct) / total_predictions * 100) if total_predictions > 0 else 0
    print(f"Positive reviews: {pos_correct}/{len(pos_predictions)} correct")
    print(f"Negative reviews: {neg_correct}/{len(neg_predictions)} correct")
    print(f"Overall accuracy: {accuracy:.1f}%")
    
    # 2. Layer-by-Layer Expert Activation Analysis
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER EXPERT ACTIVATION PATTERNS")
    print("=" * 70)
    
    # Combine all patterns
    all_patterns = {}
    for prompt_idx, layers in pos_patterns.items():
        all_patterns[f"pos_{prompt_idx}"] = layers
    for prompt_idx, layers in neg_patterns.items():
        all_patterns[f"neg_{prompt_idx}"] = layers
    
    # Analyze each layer
    layer_stats = {}
    
    for layer_idx in range(num_hidden_layers):
        # Collect all experts activated at this layer across all prompts
        layer_activations = defaultdict(int)  # expert_idx -> count
        prompt_expert_sets = []  # List of sets, one per prompt
        
        for prompt_key, layers in all_patterns.items():
            if layer_idx in layers:
                experts = layers[layer_idx]
                prompt_expert_sets.append(set(experts))
                for expert_idx in experts:
                    layer_activations[expert_idx] += 1
        
        # Calculate statistics for this layer
        all_activated = set(layer_activations.keys())
        num_unique_activated = len(all_activated)
        
        # Find consistently activated experts (appear in ALL prompts)
        if prompt_expert_sets:
            always_activated = set.intersection(*prompt_expert_sets) if prompt_expert_sets else set()
            sometimes_activated = all_activated - always_activated
        else:
            always_activated = set()
            sometimes_activated = set()
        
        # Store stats
        layer_stats[layer_idx] = {
            'total_unique': num_unique_activated,
            'always_activated': sorted(list(always_activated)),
            'sometimes_activated': sorted(list(sometimes_activated)),
            'never_activated': sorted([i for i in range(num_experts_per_layer) if i not in all_activated]),
            'activation_counts': dict(layer_activations)
        }
    
    # 3. Print Detailed Layer Statistics
    print("\nDetailed Layer-by-Layer Statistics:")
    print("=" * 70)
    
    for layer_idx in range(min(5, num_hidden_layers)):  # Show first 5 layers in detail
        stats = layer_stats[layer_idx]
        print(f"\nLayer {layer_idx:02d}:")
        print(f"  Total unique experts used: {stats['total_unique']}/{num_experts_per_layer}")
        
        if stats['always_activated']:
            print(f"  ALWAYS activated (in ALL prompts): {stats['always_activated'][:10]}")
            if len(stats['always_activated']) > 10:
                print(f"    ... and {len(stats['always_activated'])-10} more")
        else:
            print(f"  ALWAYS activated: None (no expert used in ALL prompts)")
        
        if stats['sometimes_activated']:
            # Show activation frequency for sometimes-activated experts
            sometimes_with_counts = [(idx, stats['activation_counts'][idx]) 
                                     for idx in stats['sometimes_activated'][:10]]
            print(f"  SOMETIMES activated (with counts):")
            for idx, count in sometimes_with_counts:
                pct = (count / len(all_patterns)) * 100
                print(f"    Expert {idx}: {count}/{len(all_patterns)} prompts ({pct:.1f}%)")
        
        print(f"  NEVER activated: {len(stats['never_activated'])} experts")
        if stats['never_activated']:
            print(f"    Indices: {stats['never_activated'][:20]}")
            if len(stats['never_activated']) > 20:
                print(f"    ... and {len(stats['never_activated'])-20} more")
    
    if num_hidden_layers > 5:
        print(f"\n... (Showing first 5 layers only, {num_hidden_layers-5} more layers omitted)")
    
    # 4. Global Statistics Across All Layers
    print("\n" + "=" * 70)
    print("GLOBAL EXPERT UTILIZATION SUMMARY")
    print("=" * 70)
    
    # Count globally unused experts
    globally_unused = set()
    globally_always_used = set()
    
    for layer_idx, stats in layer_stats.items():
        for expert_idx in stats['never_activated']:
            globally_unused.add((layer_idx, expert_idx))
        for expert_idx in stats['always_activated']:
            globally_always_used.add((layer_idx, expert_idx))
    
    total_utilized = total_experts_in_model - len(globally_unused)
    utilization_pct = (total_utilized / total_experts_in_model) * 100
    
    print(f"Total experts in model: {total_experts_in_model:,}")
    print(f"Experts NEVER used (across all prompts): {len(globally_unused):,} ({100-utilization_pct:.1f}%)")
    print(f"Experts SOMETIMES used: {total_utilized - len(globally_always_used):,}")
    print(f"Experts ALWAYS used (in every prompt): {len(globally_always_used):,}")
    print(f"\nMemory optimization potential: {100-utilization_pct:.1f}% of expert weights can be unloaded")
    
    # 5. Cross-Prompt Consistency Analysis
    print("\n" + "=" * 70)
    print("CROSS-PROMPT CONSISTENCY ANALYSIS")
    print("=" * 70)
    
    # Check if positive and negative prompts activate different experts
    pos_only_experts = set()
    neg_only_experts = set()
    
    for layer_idx in range(num_hidden_layers):
        pos_experts = set()
        neg_experts = set()
        
        for prompt_idx, layers in pos_patterns.items():
            if layer_idx in layers:
                pos_experts.update(layers[layer_idx])
        
        for prompt_idx, layers in neg_patterns.items():
            if layer_idx in layers:
                neg_experts.update(layers[layer_idx])
        
        # Find experts unique to each sentiment
        pos_unique = pos_experts - neg_experts
        neg_unique = neg_experts - pos_experts
        
        for exp in pos_unique:
            pos_only_experts.add((layer_idx, exp))
        for exp in neg_unique:
            neg_only_experts.add((layer_idx, exp))
    
    print(f"Experts activated ONLY for positive reviews: {len(pos_only_experts)}")
    print(f"Experts activated ONLY for negative reviews: {len(neg_only_experts)}")
    print(f"Experts activated for BOTH sentiments: {total_utilized - len(pos_only_experts) - len(neg_only_experts)}")
    
    if pos_only_experts and len(pos_only_experts) <= 20:
        print(f"\nPositive-only experts (layer, expert): {sorted(list(pos_only_experts))[:20]}")
    if neg_only_experts and len(neg_only_experts) <= 20:
        print(f"Negative-only experts (layer, expert): {sorted(list(neg_only_experts))[:20]}")
    
    return layer_stats, globally_unused

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED MoE EXPERT ACTIVATION ANALYSIS")
    print("Task: Sentiment Classification")
    print("=" * 70)
    
    # Process positive reviews
    print("\nProcessing Positive Reviews...")
    pos_patterns, pos_predictions = track_expert_activation_patterns(
        positive_reviews, 
        prompt_type="positive"
    )
    
    # Process negative reviews  
    print("\nProcessing Negative Reviews...")
    neg_patterns, neg_predictions = track_expert_activation_patterns(
        negative_reviews,
        prompt_type="negative"
    )
    
    # Analyze patterns
    if pos_patterns or neg_patterns:
        layer_stats, unused = analyze_activation_consistency(
            pos_patterns, neg_patterns, 
            pos_predictions, neg_predictions
        )
    else:
        print("\n⚠️ No data collected. Router logits may not be available.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)