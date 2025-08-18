import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Model loading
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float8_e4m3fn if "FP8" in model_name else torch.bfloat16,
)
print("Model loaded.")

# Get model configuration
num_experts = model.config.num_routed_experts if hasattr(model.config, 'num_routed_experts') else 128
top_k = model.config.num_experts_per_tok if hasattr(model.config, 'num_experts_per_tok') else 8
num_hidden_layers = model.config.num_hidden_layers

print(f"\nModel Configuration:")
print(f"  - Number of experts per layer: {num_experts}")
print(f"  - Experts activated per token: {top_k}")
print(f"  - Hidden layers: {num_hidden_layers}")
print()

# Sports equipment review prompts for sentiment classification
positive_reviews = [
    """Review: "These running shoes are incredible! Perfect cushioning and support."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Best tennis racket I've ever owned. Great power and control."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "The yoga mat has excellent grip and thickness. Worth every penny!"
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Basketball has perfect bounce and grip. Professional quality!"
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "These dumbbells are solid and well-balanced. Excellent purchase."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Golf clubs exceeded expectations. Amazing distance and accuracy."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "The bike helmet is lightweight and comfortable. Very safe design."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Swimming goggles don't leak at all. Crystal clear vision underwater!"
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Hiking boots are extremely durable and waterproof. Love them!"
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Jump rope is the perfect length and weight. Smooth rotation."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
]

negative_reviews = [
    """Review: "Running shoes fell apart after two weeks. Terrible quality."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Tennis racket strings broke on first use. Complete waste of money."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Yoga mat is too thin and slippery. Keeps sliding during practice."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Basketball loses air constantly. Won't maintain proper pressure."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Dumbbells have uneven weight distribution. Paint chips off easily."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Golf clubs bend too easily. Grips came loose after few games."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Bike helmet cracked on minor impact. Not safe at all."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Swimming goggles leak constantly. Straps broke within a week."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Hiking boots gave me blisters immediately. Not waterproof as claimed."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
    
    """Review: "Jump rope handle broke on day one. Cord is too short."
Is this review positive or negative? Answer with 'Positive' or 'Negative' only.""",
]

def track_second_token_generation(prompts, expected_sentiment="unknown"):
    """
    Track expert usage for the SECOND token generation (after the newline).
    This captures the actual content token, not the predictable newline.
    """
    # Track experts for both tokens
    first_token_experts = defaultdict(lambda: defaultdict(int))  # For newline
    second_token_experts = defaultdict(lambda: defaultdict(int))  # For actual content
    
    first_predictions = []
    second_predictions = []
    positions_used = []
    
    print(f"  Processing {len(prompts)} {expected_sentiment} reviews...")
    print(f"  Generating TWO tokens per prompt (skipping newline)...")
    
    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 5 == 0:
            print(f"    Processed {idx + 1}/{len(prompts)} reviews...")
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        initial_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            try:
                # FIRST TOKEN GENERATION (usually newline)
                outputs1 = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_router_logits=True,
                    return_dict=True
                )
                
                # Get first token
                first_token_id = torch.argmax(outputs1.logits[:, -1, :], dim=-1)
                first_token_text = tokenizer.decode(first_token_id[0].item())
                first_predictions.append(first_token_text)
                
                # Track routing for first token (position = initial_length - 1)
                first_position = initial_length - 1
                if hasattr(outputs1, 'router_logits') and outputs1.router_logits:
                    for layer_idx, layer_logits in enumerate(outputs1.router_logits):
                        if layer_logits is None:
                            continue
                        
                        # Get routing for the generating position
                        if len(layer_logits.shape) == 2:  # [seq_len, num_experts]
                            if first_position < layer_logits.shape[0]:
                                pos_logits = layer_logits[first_position]
                            else:
                                pos_logits = layer_logits[-1]
                        else:  # [batch, seq_len, num_experts]
                            if first_position < layer_logits.shape[1]:
                                pos_logits = layer_logits[0, first_position]
                            else:
                                pos_logits = layer_logits[0, -1]
                        
                        # Get top-k experts
                        probs = torch.softmax(pos_logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        
                        # Track expert usage for first token
                        for exp in top_experts:
                            first_token_experts[layer_idx][exp] += 1
                
                # SECOND TOKEN GENERATION (actual content)
                new_input_ids = torch.cat([inputs.input_ids, first_token_id.unsqueeze(0)], dim=1)
                new_attention_mask = torch.cat([
                    inputs.attention_mask,
                    torch.ones((1, 1), device=inputs.attention_mask.device, dtype=inputs.attention_mask.dtype)
                ], dim=1)
                
                outputs2 = model(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    output_router_logits=True,
                    return_dict=True
                )
                
                # Get second token
                second_token_id = torch.argmax(outputs2.logits[:, -1, :], dim=-1)
                second_token_text = tokenizer.decode(second_token_id[0].item())
                second_predictions.append(second_token_text)
                
                # Track routing for second token
                second_position = new_input_ids.shape[1] - 1
                positions_used.append(second_position)
                
                if hasattr(outputs2, 'router_logits') and outputs2.router_logits:
                    for layer_idx, layer_logits in enumerate(outputs2.router_logits):
                        if layer_logits is None:
                            continue
                        
                        # Get routing for the generating position
                        if len(layer_logits.shape) == 2:  # [seq_len, num_experts]
                            if second_position < layer_logits.shape[0]:
                                pos_logits = layer_logits[second_position]
                            else:
                                pos_logits = layer_logits[-1]
                        else:  # [batch, seq_len, num_experts]
                            if second_position < layer_logits.shape[1]:
                                pos_logits = layer_logits[0, second_position]
                            else:
                                pos_logits = layer_logits[0, -1]
                        
                        # Get top-k experts
                        probs = torch.softmax(pos_logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        
                        # Track expert usage for second token
                        for exp in top_experts:
                            second_token_experts[layer_idx][exp] += 1
                        
            except Exception as e:
                if idx == 0:
                    print(f"    Error: {str(e)[:100]}")
                first_predictions.append("Error")
                second_predictions.append("Error")
    
    return first_token_experts, second_token_experts, first_predictions, second_predictions, positions_used

def analyze_sentiment_experts(pos_first_experts, pos_second_experts, neg_first_experts, neg_second_experts,
                             pos_first_preds, pos_second_preds, neg_first_preds, neg_second_preds,
                             pos_positions, neg_positions):
    """
    Analyze expert specialization for sentiment classification, focusing on the second token.
    """
    print("\n" + "=" * 70)
    print("SENTIMENT CLASSIFICATION EXPERT ANALYSIS")
    print("=" * 70)
    
    # Check prediction accuracy
    print("\nPrediction Accuracy Check:")
    print("-" * 40)
    
    # Check SECOND token predictions (the actual content)
    pos_correct = sum(1 for p in pos_second_preds if 'positive' in p.lower() or 'pos' in p.lower())
    neg_correct = sum(1 for p in neg_second_preds if 'negative' in p.lower() or 'neg' in p.lower())
    
    total_predictions = len(pos_second_preds) + len(neg_second_preds)
    accuracy = ((pos_correct + neg_correct) / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"Positive reviews: {pos_correct}/{len(pos_second_preds)} correctly classified")
    print(f"Negative reviews: {neg_correct}/{len(neg_second_preds)} correctly classified")
    print(f"Overall accuracy: {accuracy:.1f}%")
    
    # Show sample predictions
    print(f"\nFirst token predictions (should be newline):")
    print(f"  Positive: {[repr(p) for p in pos_first_preds[:5]]}")
    print(f"  Negative: {[repr(p) for p in neg_first_preds[:5]]}")
    
    print(f"\nSecond token predictions (actual classification):")
    print(f"  Positive: {pos_second_preds[:5]}")
    print(f"  Negative: {neg_second_preds[:5]}")
    
    # Show positions used
    print(f"\nPositions where second token was generated:")
    print(f"  Positive reviews: {set(pos_positions)}")
    print(f"  Negative reviews: {set(neg_positions)}")
    
    # Analyze SECOND token experts (the meaningful ones)
    print("\n" + "="*40)
    print("SECOND TOKEN EXPERT ANALYSIS")
    print("="*40)
    
    # Aggregate expert usage across all layers
    pos_total = defaultdict(int)
    neg_total = defaultdict(int)
    
    for layer_counts in pos_second_experts.values():
        for expert, count in layer_counts.items():
            pos_total[expert] += count
    
    for layer_counts in neg_second_experts.values():
        for expert, count in layer_counts.items():
            neg_total[expert] += count
    
    # Calculate statistics
    all_experts = set(pos_total.keys()) | set(neg_total.keys())
    pos_only = set(pos_total.keys()) - set(neg_total.keys())
    neg_only = set(neg_total.keys()) - set(pos_total.keys())
    shared = set(pos_total.keys()) & set(neg_total.keys())
    
    print(f"\nOverall Expert Usage (Second Token):")
    print(f"  • Total unique experts activated: {len(all_experts)}")
    print(f"  • Experts used by BOTH sentiments: {len(shared)}")
    print(f"  • Positive-ONLY experts: {len(pos_only)}")
    print(f"  • Negative-ONLY experts: {len(neg_only)}")
    
    # Top experts for each sentiment
    print(f"\n{'='*40}")
    print("TOP EXPERTS BY SENTIMENT (Second Token)")
    print(f"{'='*40}")
    
    # Positive sentiment experts
    pos_sorted = sorted(pos_total.items(), key=lambda x: x[1], reverse=True)
    total_pos = sum(pos_total.values())
    
    print(f"\n😊 Positive Sentiment - Top 10 Experts:")
    for i, (expert, count) in enumerate(pos_sorted[:10], 1):
        percentage = (count / total_pos * 100) if total_pos > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:4d} activations ({percentage:5.2f}%)")
    
    # Negative sentiment experts
    neg_sorted = sorted(neg_total.items(), key=lambda x: x[1], reverse=True)
    total_neg = sum(neg_total.values())
    
    print(f"\n😞 Negative Sentiment - Top 10 Experts:")
    for i, (expert, count) in enumerate(neg_sorted[:10], 1):
        percentage = (count / total_neg * 100) if total_neg > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:4d} activations ({percentage:5.2f}%)")
    
    # Sentiment specialization analysis
    print(f"\n{'='*40}")
    print("SENTIMENT SPECIALIZATION SCORES")
    print(f"{'='*40}")
    
    specialization_scores = {}
    for expert in shared:
        pos_count = pos_total[expert]
        neg_count = neg_total[expert]
        total = pos_count + neg_count
        
        if total > 10:  # Only consider experts with meaningful usage
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
    
    # Show top specialists
    print(f"\nMost Specialized Experts:")
    print("(+1.0 = pure positive, -1.0 = pure negative)")
    print("-" * 60)
    
    for expert, data in sorted_specialists[:10]:
        sentiment = "POS" if data['score'] > 0 else "NEG"
        print(f"  Expert {expert:3d}: {data['score']:+.3f} [{sentiment}] "
              f"(pos:{data['pos_count']:3d}, neg:{data['neg_count']:3d})")
    
    # Show exclusive experts
    if pos_only:
        print(f"\n✅ Positive-EXCLUSIVE Experts ({len(pos_only)}):")
        print(f"   {sorted(list(pos_only))[:20]}")
    
    if neg_only:
        print(f"\n❌ Negative-EXCLUSIVE Experts ({len(neg_only)}):")
        print(f"   {sorted(list(neg_only))[:20]}")
    
    # Layer-by-layer analysis for key layers
    print(f"\n{'='*40}")
    print("LAYER-SPECIFIC ANALYSIS (First 5 Layers)")
    print(f"{'='*40}")
    
    for layer_idx in range(min(5, num_hidden_layers)):
        if layer_idx in pos_second_experts and layer_idx in neg_second_experts:
            pos_layer = set(pos_second_experts[layer_idx].keys())
            neg_layer = set(neg_second_experts[layer_idx].keys())
            
            overlap = len(pos_layer & neg_layer)
            total_unique = len(pos_layer | neg_layer)
            
            print(f"\nLayer {layer_idx:2d}:")
            print(f"  Positive experts: {len(pos_layer)}")
            print(f"  Negative experts: {len(neg_layer)}")
            print(f"  Shared experts: {overlap}")
            print(f"  Differentiation: {(1 - overlap/total_unique)*100:.1f}%")
    
    # Compare first vs second token patterns
    print(f"\n{'='*40}")
    print("FIRST vs SECOND TOKEN COMPARISON")
    print(f"{'='*40}")
    
    # First token analysis (should be mostly identical)
    first_pos_total = defaultdict(int)
    first_neg_total = defaultdict(int)
    
    for layer_counts in pos_first_experts.values():
        for expert, count in layer_counts.items():
            first_pos_total[expert] += count
    
    for layer_counts in neg_first_experts.values():
        for expert, count in layer_counts.items():
            first_neg_total[expert] += count
    
    first_all = set(first_pos_total.keys()) | set(first_neg_total.keys())
    first_shared = set(first_pos_total.keys()) & set(first_neg_total.keys())
    
    print(f"\nFirst Token (newline):")
    print(f"  Total unique experts: {len(first_all)}")
    print(f"  Overlap between sentiments: {len(first_shared)/len(first_all)*100:.1f}%")
    
    print(f"\nSecond Token (content):")
    print(f"  Total unique experts: {len(all_experts)}")
    print(f"  Overlap between sentiments: {len(shared)/len(all_experts)*100:.1f}%")
    
    print(f"\nDiversity increase from first to second token:")
    print(f"  {len(all_experts) - len(first_all)} more experts activated")
    print(f"  {(len(all_experts)/len(first_all) - 1)*100:.1f}% increase in diversity")

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("MoE SENTIMENT ANALYSIS - SECOND TOKEN ROUTING")
    print("Sports Equipment Reviews")
    print("=" * 70)
    print(f"\nDataset:")
    print(f"  • {len(positive_reviews)} positive reviews")
    print(f"  • {len(negative_reviews)} negative reviews")
    print(f"  • Analyzing SECOND token (after newline)")
    
    # Process positive reviews
    print("\n" + "="*40)
    print("Processing Positive Reviews")
    print("="*40)
    pos_first, pos_second, pos_first_preds, pos_second_preds, pos_positions = track_second_token_generation(
        positive_reviews, 
        expected_sentiment="positive"
    )
    
    # Process negative reviews
    print("\n" + "="*40)
    print("Processing Negative Reviews")
    print("="*40)
    neg_first, neg_second, neg_first_preds, neg_second_preds, neg_positions = track_second_token_generation(
        negative_reviews,
        expected_sentiment="negative"
    )
    
    # Analyze sentiment specialization
    if pos_second and neg_second:
        analyze_sentiment_experts(
            pos_first, pos_first, neg_first, neg_first, # Use the FIRST token expert data
            pos_first_preds, pos_first_preds, neg_first_preds, neg_first_preds, # Use the FIRST token predictions
            pos_positions, neg_positions # Positions will need re-evaluation, but the expert data is key
        )
    else:
        print("\n⚠️  Insufficient data collected for analysis")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: VRAM OPTIMIZATION POTENTIAL")
    print("=" * 70)
    
    # Calculate total unique experts needed
    all_experts_used = set()
    for layer_idx in range(num_hidden_layers):
        if layer_idx in pos_second:
            all_experts_used.update((layer_idx, exp) for exp in pos_second[layer_idx].keys())
        if layer_idx in neg_second:
            all_experts_used.update((layer_idx, exp) for exp in neg_second[layer_idx].keys())
    
    total_possible = num_hidden_layers * num_experts
    utilization = len(all_experts_used) / total_possible * 100
    
    print(f"\nFor this sentiment classification task:")
    print(f"  • Total experts in model: {total_possible:,}")
    print(f"  • Experts actually used: {len(all_experts_used):,}")
    print(f"  • Utilization: {utilization:.2f}%")
    print(f"  • Potential VRAM savings: {100-utilization:.2f}%")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)