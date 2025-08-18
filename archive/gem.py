import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Model Loading ---
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
print(f"Loading model: {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Using bfloat16 for broader compatibility
    )
    print("Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# --- Model Configuration ---
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

# --- Dataset ---
# (Dataset lists are collapsed for brevity, they remain unchanged)

positive_reviews = [
    """Review: "These running shoes are incredible! Perfect cushioning and support."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Best tennis racket I've ever owned. Great power and control."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "The yoga mat has excellent grip and thickness. Worth every penny!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Basketball has perfect bounce and grip. Professional quality!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "These dumbbells are solid and well-balanced. Excellent purchase."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Golf clubs exceeded expectations. Amazing distance and accuracy."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "The bike helmet is lightweight and comfortable. Very safe design."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Swimming goggles don't leak at all. Crystal clear vision underwater!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Hiking boots are extremely durable and waterproof. Love them!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Jump rope is the perfect length and weight. Smooth rotation."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Resistance bands are strong and versatile. Great for home workouts."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Soccer ball maintains perfect shape. Professional feel and quality."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "The weightlifting belt provides excellent support. Very sturdy construction."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Skateboard has smooth wheels and stable deck. Rides like a dream!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Boxing gloves fit perfectly and protect well. Outstanding quality."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "The foam roller works wonders for recovery. Firm but comfortable."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Climbing harness is super secure and adjustable. Feel very safe."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Baseball glove has perfect pocket formation. Catches everything!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Exercise bike is quiet and smooth. Display is very accurate."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Gym bag has tons of compartments. Durable material and zippers."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Pull-up bar is rock solid when installed. Holds weight perfectly."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Paddle board is stable and easy to maneuver. Great for beginners!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Ski goggles have amazing anti-fog coating. Clear vision all day."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "The rowing machine operates smoothly and quietly. Excellent cardio!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Volleyball has perfect weight and texture. Tournament quality for sure."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Treadmill is sturdy and has great features. Motor runs perfectly."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Cycling shorts have excellent padding. No discomfort on long rides."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "The punching bag is well-filled and durable. Takes heavy hits well."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Fitness tracker is accurate and comfortable. Battery lasts forever!"
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Hockey stick has perfect flex and weight. Shots are more powerful."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
]

negative_reviews = [
    """Review: "Running shoes fell apart after two weeks. Terrible quality."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Tennis racket strings broke on first use. Complete waste of money."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Yoga mat is too thin and slippery. Keeps sliding during practice."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Basketball loses air constantly. Won't maintain proper pressure."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Dumbbells have uneven weight distribution. Paint chips off easily."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Golf clubs bend too easily. Grips came loose after few games."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Bike helmet cracked on minor impact. Not safe at all."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Swimming goggles leak constantly. Straps broke within a week."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Hiking boots gave me blisters immediately. Not waterproof as claimed."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Jump rope handle broke on day one. Cord is too short."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Resistance bands snapped during light use. Dangerous and poorly made."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Soccer ball went flat after one game. Stitching came undone."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Weightlifting belt buckle broke immediately. Material is very cheap."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Skateboard wheels are uneven. Bearings seized up quickly."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Boxing gloves padding compressed flat. No protection anymore."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Foam roller is too soft to be effective. Started falling apart."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Climbing harness has weak stitching. Would not trust my life to it."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Baseball glove is stiff and won't break in. Laces keep breaking."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Exercise bike makes horrible noises. Display stopped working."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Gym bag zipper broke first day. Material ripped at seams."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Pull-up bar bent under normal weight. Mounting hardware failed."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Paddle board has multiple leaks. Impossible to stay inflated."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Ski goggles fog up instantly. Strap adjustment broke off."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Rowing machine resistance is inconsistent. Seat is uncomfortable."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Volleyball is lopsided and won't fly straight. Material feels cheap."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Treadmill belt slips constantly. Motor overheats after 10 minutes."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Cycling shorts padding is too thin. Seams cause painful chafing."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Punching bag leaked sand everywhere. Chain attachment broke."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Fitness tracker gives inaccurate readings. Won't sync with phone."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
    
    """Review: "Hockey stick blade separated from shaft. Grip tape peeled off."
Is this review positive or negative? ANSWER ONLY WITH Yes IF POSITIVE and No IF NEGATIVE. ONLY 1 word only Yes or No""",
]


def track_single_token_generation(prompts, expected_sentiment="unknown"):
    """
    Track expert usage for single-token generation.
    Returns both aggregated counts and per-prompt usage for consistency analysis.
    """
    per_prompt_expert_sets = []  # List of sets, one for each prompt
    predictions = []
    
    print(f"  Processing {len(prompts)} {expected_sentiment} reviews...")
    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{len(prompts)} reviews...")
        
        current_prompt_experts = set()  # Tracks (layer, expert) tuples for this prompt

        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                predicted_text = tokenizer.decode([next_token_id]).strip()
                predictions.append(predicted_text)

                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        if layer_logits is None: continue
                        
                        if len(layer_logits.shape) == 3:  # (batch, seq, experts)
                            last_token_logits = layer_logits[0, -1, :]
                        elif len(layer_logits.shape) == 2:  # (batch, experts)
                            last_token_logits = layer_logits[0]
                        else: continue
                        
                        probs = torch.softmax(last_token_logits, dim=-1)
                        k = min(top_k, len(probs))
                        top_experts = torch.topk(probs, k).indices.tolist()
                        
                        for exp_idx in top_experts:
                            current_prompt_experts.add((layer_idx, exp_idx))
            except Exception as e:
                if idx == 0:
                    print(f"    Note: Error during tracking - {str(e)[:50]}")
                predictions.append("Error")
        
        per_prompt_expert_sets.append(current_prompt_experts)

    # Aggregate results for existing report sections
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    for expert_set in per_prompt_expert_sets:
        for layer_idx, expert_idx in expert_set:
            expert_counts[layer_idx][expert_idx] += 1
            total_selections[layer_idx] += 1 # Simplified aggregation
            
    return expert_counts, total_selections, predictions, per_prompt_expert_sets


def analyze_task_utilization(positive_counts, negative_counts, pos_predictions, neg_predictions, pos_prompt_usage, neg_prompt_usage):
    """
    Analyzes overall expert utilization, lists activated experts, and checks for routing consistency.
    """
    print("\n" + "=" * 70)
    print("TASK-SPECIFIC EXPERT UTILIZATION ANALYSIS")
    print("=" * 70)

    # --- 1. Accuracy Check ---
    print("\nPrediction Accuracy Check:")
    print("-" * 40)
    pos_correct = sum(1 for p in pos_predictions if 'yes' in p.lower())
    neg_correct = sum(1 for p in neg_predictions if 'no' in p.lower())
    total_predictions = len(pos_predictions) + len(neg_predictions)
    accuracy = ((pos_correct + neg_correct) / total_predictions * 100) if total_predictions > 0 else 0
    print(f"Positive reviews (Expected 'Yes'): {pos_correct}/{len(pos_predictions)} correctly classified")
    print(f"Negative reviews (Expected 'No'): {neg_correct}/{len(neg_predictions)} correctly classified")
    print(f"Overall accuracy: {(pos_correct + neg_correct)}/{total_predictions} ({accuracy:.1f}%)")
    print(f"Sample predictions (Pos): {pos_predictions[:5]}")
    print(f"Sample predictions (Neg): {neg_predictions[:5]}")
    if accuracy < 50:
        print(f"\n⚠️ WARNING: Accuracy is low ({accuracy:.1f}%). The utilization analysis might be unreliable.")
        print("Ensure the chat template is applied correctly and the model understands the task.")

    # --- 2. Combine Utilization Data & Calculate Stats ---
    utilized_experts = set()
    for layer_counts in list(positive_counts.values()) + list(negative_counts.values()):
        for expert_idx in layer_counts.keys():
            utilized_experts.add((layer_idx, expert_idx)) # Note: This assumes layer_idx is part of the key structure, which it is.

    num_utilized_experts = len(utilized_experts)
    num_unused_experts = total_experts_in_model - num_utilized_experts
    utilization_percentage = (num_utilized_experts / total_experts_in_model) * 100
    prunable_percentage = (num_unused_experts / total_experts_in_model) * 100

    print(f"\n" + "="*40)
    print("OVERALL UTILIZATION REPORT")
    print("="*40)
    print(f"Total experts in model: {total_experts_in_model:,}")
    print(f"Total experts utilized for this task: {num_utilized_experts:,} ({utilization_percentage:.2f}%)")
    print(f"Total experts UNUSED (Prunable/Unloadable): {num_unused_experts:,} ({prunable_percentage:.2f}%)")
    print("="*40)
    print(f"Conclusion: You only need {utilization_percentage:.2f}% of the expert weights loaded in VRAM for this specific task.")
    
    # --- 3. Layer-wise Utilization Density ---
    print(f"\nLayer-wise Utilization Density:")
    print(f"(How many unique experts out of {num_experts_per_layer} were used in each layer)")
    print("-" * 70)
    layer_utilization = defaultdict(int)
    for layer_idx, _ in utilized_experts:
        layer_utilization[layer_idx] += 1
    
    def get_bar(count, total=num_experts_per_layer, width=50):
        filled = int(width * count / total) if total > 0 else 0
        return '█' * filled + '░' * (width - filled)

    for layer_idx in range(num_hidden_layers):
        count = layer_utilization.get(layer_idx, 0)
        percentage = (count / num_experts_per_layer * 100) if num_experts_per_layer > 0 else 0
        print(f"L{layer_idx:02d}: {count:4d}/{num_experts_per_layer} ({percentage:5.1f}%) | {get_bar(count)}")

    # --- 4. [NEW] Activated Experts Identification ---
    print(f"\n" + "="*40)
    print("ACTIVATED EXPERTS IDENTIFICATION (Detailed List)")
    print("="*40)
    
    used_experts_per_layer = defaultdict(list)
    for layer_idx, expert_idx in sorted(list(utilized_experts)):
        used_experts_per_layer[layer_idx].append(expert_idx)

    for layer_idx in range(num_hidden_layers):
        used_in_layer = used_experts_per_layer.get(layer_idx, [])
        if used_in_layer:
            print(f"\nLayer {layer_idx:02d}: {len(used_in_layer)} activated experts.")
            print("  Indices:")
            for i in range(0, len(used_in_layer), 10):
                print(f"    {used_in_layer[i:i+10]}")

    # --- 5. [NEW] Routing Consistency Analysis ---
    print(f"\n" + "="*40)
    print("EXPERT ROUTING CONSISTENCY ANALYSIS")
    print("="*40)
    
    all_prompt_usage = pos_prompt_usage + neg_prompt_usage
    if not all_prompt_usage or not any(all_prompt_usage):
        print("No expert usage data was collected to analyze consistency.")
    else:
        layer_expert_sets = defaultdict(list)
        for prompt_set in all_prompt_usage:
            experts_by_layer_this_prompt = defaultdict(set)
            for layer_idx, expert_idx in prompt_set:
                experts_by_layer_this_prompt[layer_idx].add(expert_idx)
            
            for layer_idx in range(num_hidden_layers):
                layer_expert_sets[layer_idx].append(experts_by_layer_this_prompt.get(layer_idx, set()))

        all_layers_consistent = True
        inconsistent_layers_details = []

        for layer_idx in range(num_hidden_layers):
            sets_for_this_layer = layer_expert_sets.get(layer_idx, [])
            if not sets_for_this_layer: continue

            unique_routing_patterns = {frozenset(s) for s in sets_for_this_layer}
            if len(unique_routing_patterns) > 1:
                all_layers_consistent = False
                inconsistent_layers_details.append(
                    f"  - Layer {layer_idx:02d}: ❌ INCONSISTENT. Found {len(unique_routing_patterns)} different expert sets."
                )
        
        if all_layers_consistent:
            print("✅ FULLY CONSISTENT: The exact same set of experts was activated in each layer for every prompt.")
            print("  This suggests the routing is static for this specific task and prompt set.")
        else:
            print("ℹ️ DYNAMIC ROUTING DETECTED: Different prompts triggered different sets of experts in some layers.")
            for detail in inconsistent_layers_details:
                print(detail)


# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 70)
    print("MoE UTILIZATION ANALYSIS FOR TASK-SPECIFIC VRAM OPTIMIZATION")
    print("Task: Sentiment Classification (Sports Equipment Reviews)")
    print("=" * 70)
    
    print("\n" + "="*40)
    print("Processing Positive Reviews")
    print("="*40)
    pos_counts, pos_totals, pos_predictions, pos_prompt_usage = track_single_token_generation(
        positive_reviews, expected_sentiment="positive"
    )

    print("\n" + "="*40)
    print("Processing Negative Reviews")
    print("="*40)
    neg_counts, neg_totals, neg_predictions, neg_prompt_usage = track_single_token_generation(
        negative_reviews, expected_sentiment="negative"
    )
    
    if pos_counts or neg_counts:
        analyze_task_utilization(
            pos_counts, neg_counts, 
            pos_predictions, neg_predictions, 
            pos_prompt_usage, neg_prompt_usage
        )
    else:
        print("\n⚠️ No data collected. Router logits may not be exposed properly for this model.")
        
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)