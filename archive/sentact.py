import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Assuming model and tokenizer are already loaded
# If not, uncomment these lines:
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
    # Exit if the model cannot be loaded
    exit()

# Get model configuration
num_experts_per_layer = model.config.num_routed_experts if hasattr(model.config, 'num_routed_experts') else 128
top_k = model.config.num_experts_per_tok if hasattr(model.config, 'num_experts_per_tok') else 8
num_hidden_layers = model.config.num_hidden_layers

# Calculate total experts in the model
# Based on the configuration, we assume all hidden layers are MoE layers.
total_experts_in_model = num_hidden_layers * num_experts_per_layer


print(f"\nModel Configuration:")
print(f"  - Hidden layers: {num_hidden_layers}")
print(f"  - Experts per layer: {num_experts_per_layer}")
print(f"  - Experts activated per token (Top-K): {top_k}")
print(f"  - TOTAL unique experts in model: {total_experts_in_model:,}")
print()

# Sports equipment review prompts for sentiment classification
# (The full lists are included below)
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
    Track expert usage for single-token generation (sentiment classification).
    The structure (Layer -> Expert Index -> Count) is kept as it is suitable for detailed analysis.
    """
    # Structure: expert_counts[layer_idx][expert_idx] = count
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    predictions = []
    
    print(f"  Processing {len(prompts)} {expected_sentiment} reviews...")
    
    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{len(prompts)} reviews...")
        
        # Apply the chat template (Crucial for Instruct models)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # add_generation_prompt=True signals the assistant to start generating.
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the formatted input
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            try:
                # Single forward pass to generate one token
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_router_logits=True,
                    return_dict=True
                )
                
                # Get the predicted token
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                predicted_text = tokenizer.decode([next_token_id]).strip()
                predictions.append(predicted_text)
                
                # Process router logits for this single forward pass
                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    # Iterate through layers
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        if layer_logits is None:
                            continue
                        
                        # Get the routing decision for the last position (the token determining the output)
                        if len(layer_logits.shape) == 3:  # (batch, seq, experts)
                            # We want the last token position
                            last_token_logits = layer_logits[0, -1, :]
                        elif len(layer_logits.shape) == 2:  # (batch, experts)
                            # Assuming batch size 1 if sequence dimension is missing
                            last_token_logits = layer_logits[0]
                        else:
                            continue
                        
                        # Convert to probabilities
                        probs = torch.softmax(last_token_logits, dim=-1)
                        
                        # Get top-k experts
                        k = min(top_k, len(probs))
                        top_experts = torch.topk(probs, k).indices.tolist()
                        
                        # Track expert usage
                        for exp_idx in top_experts:
                            expert_counts[layer_idx][exp_idx] += 1
                        total_selections[layer_idx] += k
                        
            except Exception as e:
                if idx == 0:
                    print(f"    Note: Error during tracking - {str(e)[:50]}")
                predictions.append("Error")
    
    return expert_counts, total_selections, predictions

def analyze_task_utilization(positive_counts, negative_counts, pos_predictions, neg_predictions):
    """
    Analyzes overall expert utilization across the entire task to identify optimization opportunities.
    Treats (Layer, Expert Index) as a unique expert.
    """
    print("\n" + "=" * 70)
    print("TASK-SPECIFIC EXPERT UTILIZATION ANALYSIS")
    print("=" * 70)
    
    # 1. Accuracy Check (Sanity check that the task was performed correctly)
    print("\nPrediction Accuracy Check:")
    print("-" * 40)
    
    # Check based on expected words "Yes" or "No" as instructed in the prompt
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

    # 2. Combine Utilization Data
    # Identify all unique experts used across both positive and negative reviews.
    utilized_experts = set() # Set of (layer_idx, expert_idx) tuples

    # Process positive counts
    for layer_idx, layer_counts in positive_counts.items():
        for expert_idx in layer_counts.keys():
            utilized_experts.add((layer_idx, expert_idx))

    # Process negative counts
    for layer_idx, layer_counts in negative_counts.items():
        for expert_idx in layer_counts.keys():
            utilized_experts.add((layer_idx, expert_idx))
            
    # 3. Calculate Utilization Statistics
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

    
    # 4. Layer-wise Utilization Density
    print(f"\nLayer-wise Utilization Density:")
    print(f"(How many unique experts out of {num_experts_per_layer} were used in each layer)")
    print("-" * 70)
    
    layer_utilization = defaultdict(int)
    for layer_idx, expert_idx in utilized_experts:
        layer_utilization[layer_idx] += 1
        
    # Visualization helper
    def get_bar(count, total=num_experts_per_layer, width=50):
        if total == 0: return '░' * width
        filled = int(width * count / total)
        return '█' * filled + '░' * (width - filled)

    for layer_idx in range(num_hidden_layers):
        count = layer_utilization.get(layer_idx, 0)
        percentage = (count / num_experts_per_layer * 100) if num_experts_per_layer > 0 else 0
        print(f"L{layer_idx:02d}: {count:4d}/{num_experts_per_layer} ({percentage:5.1f}%) | {get_bar(count)}")

    # 5. Identifying Unused Experts (Detailed List)
    print(f"\n" + "="*40)
    print("UNUSED EXPERTS IDENTIFICATION (Detailed List)")
    print("="*40)
    
    print("Identifying experts that were NEVER activated during this task.")
    
    unused_experts_list = []
    
    # Iterate through all possible experts
    for layer_idx in range(num_hidden_layers):
        unused_in_layer = []
        for expert_idx in range(num_experts_per_layer):
            if (layer_idx, expert_idx) not in utilized_experts:
                unused_in_layer.append(expert_idx)
        
        if unused_in_layer:
            print(f"\nLayer {layer_idx:02d}: {len(unused_in_layer)} unused experts.")
            # Print the list of indices, formatted for easier reading
            print("  Indices:")
            # Print in rows of 10
            for i in range(0, len(unused_in_layer), 10):
                print(f"    {unused_in_layer[i:i+10]}")
            
            unused_experts_list.extend([(layer_idx, idx) for idx in unused_in_layer])

    if not unused_experts_list:
        print("\n✅ All experts were utilized at least once.")
    else:
        print(f"\nTotal unused experts identified: {len(unused_experts_list)}")


# Main execution
if __name__ == "__main__":
    # We remove the old analysis function (analyze_sentiment_experts) as it was 
    # aggregating experts across layers, which contradicts the goal.
    
    print("=" * 70)
    print("MoE UTILIZATION ANALYSIS FOR TASK-SPECIFIC VRAM OPTIMIZATION")
    print("Task: Sentiment Classification (Sports Equipment Reviews)")
    print("=" * 70)
    
    # Process positive reviews
    print("\n" + "="*40)
    print("Processing Positive Reviews")
    print("="*40)
    pos_counts, pos_totals, pos_predictions = track_single_token_generation(
        positive_reviews, 
        expected_sentiment="positive"
    )
    
    # Process negative reviews
    print("\n" + "="*40)
    print("Processing Negative Reviews")
    print("="*40)
    neg_counts, neg_totals, neg_predictions = track_single_token_generation(
        negative_reviews,
        expected_sentiment="negative"
    )
    
    # Analyze utilization and pruning opportunities
    if pos_counts or neg_counts:
        analyze_task_utilization(pos_counts, neg_counts, pos_predictions, neg_predictions)
    else:
        print("\n⚠️  No data collected. Router logits may not be exposed properly for this model.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)