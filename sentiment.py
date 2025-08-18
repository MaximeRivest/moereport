import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Assuming model and tokenizer are already loaded
# If not, uncomment these lines:
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float8_e4m3fn if "FP8" in model_name else torch.bfloat16,
)

# Get model configuration
num_experts = model.config.num_routed_experts if hasattr(model.config, 'num_routed_experts') else 128
top_k = model.config.num_experts_per_tok if hasattr(model.config, 'num_experts_per_tok') else 8
num_hidden_layers = model.config.num_hidden_layers

print(f"Model Configuration:")
print(f"  - Number of experts: {num_experts}")
print(f"  - Experts per token: {top_k}")
print(f"  - Hidden layers: {num_hidden_layers}")
print()

# Sports equipment review prompts for sentiment classification
# Each prompt is designed to elicit a single-token "Positive" or "Negative" response

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
    """
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    predictions = []
    
    print(f"  Processing {len(prompts)} {expected_sentiment} reviews...")
    
    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{len(prompts)} reviews...")
        
        # --- MODIFICATION START ---
        # Define the conversation structure
        messages = [
            # System prompt is often optional but good practice
            # {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}
        ]
        
        # Apply the chat template. 
        # add_generation_prompt=True is crucial as it adds the tokens 
        # that signal the assistant to start generating its response.
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the formatted input
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        # --- MODIFICATION END ---

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
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        if layer_logits is None:
                            continue
                        
                        # Get the routing decision for the last position
                        if len(layer_logits.shape) == 3:  # (batch, seq, experts)
                            # We want the last token position
                            last_token_logits = layer_logits[0, -1, :]
                        elif len(layer_logits.shape) == 2:  # (batch, experts)
                            last_token_logits = layer_logits[0]
                        else:
                            continue
                        
                        # Convert to probabilities
                        probs = torch.softmax(last_token_logits, dim=-1)
                        
                        # Get top-k experts
                        k = min(top_k, len(probs))
                        top_experts = torch.topk(probs, k).indices.tolist()
                        
                        # Track expert usage
                        for exp in top_experts:
                            expert_counts[layer_idx][exp] += 1
                        total_selections[layer_idx] += k
                        
            except Exception as e:
                if idx == 0:
                    print(f"    Note: Error during tracking - {str(e)[:50]}")
                predictions.append("Error")
    
    return expert_counts, total_selections, predictions

def analyze_sentiment_experts(positive_counts, negative_counts, pos_predictions, neg_predictions):
    """
    Analyze expert specialization for positive vs negative sentiment classification.
    """
    print("\n" + "=" * 70)
    print("SENTIMENT CLASSIFICATION EXPERT ANALYSIS")
    print("=" * 70)
    
    # Check prediction accuracy
    print("\nPrediction Accuracy Check:")
    print("-" * 40)
    
    # Count positive predictions
    pos_correct = sum(1 for p in pos_predictions if 'positive' in p.lower() or 'pos' in p.lower())
    neg_as_pos = sum(1 for p in neg_predictions if 'positive' in p.lower() or 'pos' in p.lower())
    
    # Count negative predictions  
    neg_correct = sum(1 for p in neg_predictions if 'negative' in p.lower() or 'neg' in p.lower())
    pos_as_neg = sum(1 for p in pos_predictions if 'negative' in p.lower() or 'neg' in p.lower())
    
    print(f"Positive reviews: {pos_correct}/{len(pos_predictions)} correctly classified")
    print(f"Negative reviews: {neg_correct}/{len(neg_predictions)} correctly classified")
    print(f"Overall accuracy: {(pos_correct + neg_correct)}/{len(pos_predictions) + len(neg_predictions)} "
          f"({(pos_correct + neg_correct)/(len(pos_predictions) + len(neg_predictions))*100:.1f}%)")
    
    # Show sample predictions
    print(f"\nSample predictions (first 5 each):")
    print(f"  Positive reviews → {pos_predictions[:5]}")
    print(f"  Negative reviews → {neg_predictions[:5]}")
    
    # Aggregate expert usage
    pos_total = defaultdict(int)
    neg_total = defaultdict(int)
    
    for layer_counts in positive_counts.values():
        for expert, count in layer_counts.items():
            pos_total[expert] += count
    
    for layer_counts in negative_counts.values():
        for expert, count in layer_counts.items():
            neg_total[expert] += count
    
    if not pos_total and not neg_total:
        print("\n⚠️  No expert routing data collected")
        return
    
    # Calculate statistics
    all_experts = set(pos_total.keys()) | set(neg_total.keys())
    pos_only = set(pos_total.keys()) - set(neg_total.keys())
    neg_only = set(neg_total.keys()) - set(pos_total.keys())
    shared = set(pos_total.keys()) & set(neg_total.keys())
    
    print(f"\n" + "="*40)
    print("EXPERT USAGE STATISTICS")
    print("="*40)
    
    print(f"\nOverall Statistics:")
    print(f"  • Total unique experts activated: {len(all_experts)}")
    print(f"  • Experts used by both sentiments: {len(shared)}")
    print(f"  • Positive-only experts: {len(pos_only)}")
    print(f"  • Negative-only experts: {len(neg_only)}")
    
    # Top experts for each sentiment
    print(f"\n{'='*40}")
    print("TOP EXPERTS BY SENTIMENT")
    print(f"{'='*40}")
    
    # Positive sentiment experts
    pos_sorted = sorted(pos_total.items(), key=lambda x: x[1], reverse=True)
    total_pos = sum(pos_total.values())
    
    print(f"\n😊 Positive Sentiment - Top 10 Experts:")
    print(f"(Total activations: {total_pos:,})")
    for i, (expert, count) in enumerate(pos_sorted[:10], 1):
        percentage = (count / total_pos * 100) if total_pos > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:4d} ({percentage:5.2f}%)")
    
    # Negative sentiment experts
    neg_sorted = sorted(neg_total.items(), key=lambda x: x[1], reverse=True)
    total_neg = sum(neg_total.values())
    
    print(f"\n😞 Negative Sentiment - Top 10 Experts:")
    print(f"(Total activations: {total_neg:,})")
    for i, (expert, count) in enumerate(neg_sorted[:10], 1):
        percentage = (count / total_neg * 100) if total_neg > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:4d} ({percentage:5.2f}%)")
    
    # Sentiment specialization analysis
    print(f"\n{'='*40}")
    print("SENTIMENT SPECIALIZATION ANALYSIS")
    print(f"{'='*40}")
    
    specialization_scores = {}
    for expert in shared:
        pos_count = pos_total[expert]
        neg_count = neg_total[expert]
        total = pos_count + neg_count
        
        if total > 5:  # Only consider experts with meaningful usage
            # Score from -1 (pure negative) to +1 (pure positive)
            score = (pos_count - neg_count) / total
            specialization_scores[expert] = {
                'score': score,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'total': total
            }
    
    # Sort by absolute specialization
    sorted_specialists = sorted(specialization_scores.items(),
                               key=lambda x: abs(x[1]['score']),
                               reverse=True)
    
    # Identify specialists
    positive_specialists = []
    negative_specialists = []
    neutral_experts = []
    
    for expert, data in sorted_specialists:
        if data['score'] > 0.2:
            positive_specialists.append((expert, data))
        elif data['score'] < -0.2:
            negative_specialists.append((expert, data))
        elif abs(data['score']) < 0.1:
            neutral_experts.append((expert, data))
    
    print(f"\nSentiment Specialization Scores:")
    print("(+1.0 = pure positive, -1.0 = pure negative, 0 = balanced)")
    print("-" * 60)
    
    if positive_specialists:
        print(f"\n😊 Positive Sentiment Specialists (score > 0.2):")
        for expert, data in positive_specialists[:8]:
            print(f"  Expert {expert:3d}: score={data['score']:+.3f} "
                  f"(pos:{data['pos_count']} vs neg:{data['neg_count']})")
    
    if negative_specialists:
        print(f"\n😞 Negative Sentiment Specialists (score < -0.2):")
        for expert, data in negative_specialists[:8]:
            print(f"  Expert {expert:3d}: score={data['score']:+.3f} "
                  f"(pos:{data['pos_count']} vs neg:{data['neg_count']})")
    
    if neutral_experts:
        print(f"\n😐 Neutral/Balanced Experts (|score| < 0.1):")
        for expert, data in neutral_experts[:5]:
            print(f"  Expert {expert:3d}: score={data['score']:+.3f} "
                  f"(pos:{data['pos_count']} vs neg:{data['neg_count']})")
    
    # Exclusive experts
    if pos_only and len(pos_only) <= 20:
        print(f"\n✅ Positive-Exclusive Experts ({len(pos_only)}): {sorted(pos_only)}")
    elif pos_only:
        print(f"\n✅ Positive-Exclusive Experts: {len(pos_only)} experts")
        print(f"   First 10: {sorted(pos_only)[:10]}")
    
    if neg_only and len(neg_only) <= 20:
        print(f"\n❌ Negative-Exclusive Experts ({len(neg_only)}): {sorted(neg_only)}")
    elif neg_only:
        print(f"\n❌ Negative-Exclusive Experts: {len(neg_only)} experts")
        print(f"   First 10: {sorted(neg_only)[:10]}")
    
    # Layer-wise analysis
    print(f"\n{'='*40}")
    print("LAYER-WISE SENTIMENT PATTERNS")
    print(f"{'='*40}")
    
    # Find layers with strongest sentiment differentiation
    layer_specialization = {}
    for layer_idx in set(positive_counts.keys()) & set(negative_counts.keys()):
        pos_layer = positive_counts[layer_idx]
        neg_layer = negative_counts[layer_idx]
        
        # Calculate overlap coefficient for this layer
        pos_experts = set(pos_layer.keys())
        neg_experts = set(neg_layer.keys())
        
        if pos_experts and neg_experts:
            overlap = len(pos_experts & neg_experts)
            total_unique = len(pos_experts | neg_experts)
            differentiation = 1 - (overlap / total_unique)
            layer_specialization[layer_idx] = {
                'differentiation': differentiation,
                'pos_unique': len(pos_experts - neg_experts),
                'neg_unique': len(neg_experts - pos_experts),
                'shared': overlap
            }
    
    # Show most differentiated layers
    sorted_layers = sorted(layer_specialization.items(),
                          key=lambda x: x[1]['differentiation'],
                          reverse=True)
    
    print(f"\nMost Sentiment-Differentiated Layers (top 5):")
    for layer_idx, data in sorted_layers[:5]:
        print(f"  Layer {layer_idx:2d}: {data['differentiation']:.1%} differentiation")
        print(f"           (Pos-unique: {data['pos_unique']}, "
              f"Neg-unique: {data['neg_unique']}, Shared: {data['shared']})")

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT CLASSIFICATION MoE ANALYSIS")
    print("Sports Equipment Reviews")
    print("=" * 70)
    print(f"\nDataset:")
    print(f"  • {len(positive_reviews)} positive reviews")
    print(f"  • {len(negative_reviews)} negative reviews")
    print(f"  • Single token generation (Positive/Negative)")
    
    # Process positive reviews
    print("\n" + "="*40)
    print("Processing Positive Reviews")
    print("="*40)
    pos_counts, pos_totals, pos_predictions = track_single_token_generation(
        positive_reviews, 
        expected_sentiment="positive"
    )
    
    pos_layers = len(pos_counts)
    pos_experts = set()
    for layer_counts in pos_counts.values():
        pos_experts.update(layer_counts.keys())
    
    print(f"\nPositive Review Results:")
    print(f"  • Layers with routing: {pos_layers}")
    print(f"  • Unique experts activated: {len(pos_experts)}")
    print(f"  • Total expert selections: {sum(pos_totals.values()):,}")
    
    # Process negative reviews
    print("\n" + "="*40)
    print("Processing Negative Reviews")
    print("="*40)
    neg_counts, neg_totals, neg_predictions = track_single_token_generation(
        negative_reviews,
        expected_sentiment="negative"
    )
    
    neg_layers = len(neg_counts)
    neg_experts = set()
    for layer_counts in neg_counts.values():
        neg_experts.update(layer_counts.keys())
    
    print(f"\nNegative Review Results:")
    print(f"  • Layers with routing: {neg_layers}")
    print(f"  • Unique experts activated: {len(neg_experts)}")
    print(f"  • Total expert selections: {sum(neg_totals.values()):,}")
    
    # Analyze sentiment specialization
    if pos_counts and neg_counts:
        analyze_sentiment_experts(pos_counts, neg_counts, pos_predictions, neg_predictions)
    else:
        print("\n⚠️  Insufficient data collected for sentiment analysis")
        print("Router logits may not be exposed properly for this model")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)