import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load model and tokenizer
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Adjust as needed
)
print("Model loaded.")

# Model config
num_experts_per_layer = getattr(model.config, 'num_routed_experts', 128)
top_k = getattr(model.config, 'num_experts_per_tok', 8)
num_hidden_layers = model.config.num_hidden_layers

def track_expert_activation_for_prompts(prompts, label=""):
    expert_sets_per_layer = [defaultdict(set) for _ in range(num_hidden_layers)]
    predictions = []
    print(f"Processing {len(prompts)} prompts for {label}...")
    for idx, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers only with 'Yes' or 'No' for sentiment, or appropriate response for other tasks."},
            {"role": "user", "content": prompt}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_router_logits=True,
                return_dict=True
            )
            # Predict next token
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            predicted_text = tokenizer.decode([next_token_id]).strip()
            predictions.append(predicted_text)
            # Process router logits
            if hasattr(outputs, 'router_logits'):
                for layer_idx, layer_logits in enumerate(outputs.router_logits):
                    if layer_logits is None:
                        continue
                    # For simplicity, aggregate over all positions in the sequence
                    seq_len = layer_logits.shape[1]
                    layer_experts = set()
                    for pos in range(seq_len):
                        pos_logits = layer_logits[0, pos, :]
                        probs = torch.softmax(pos_logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        layer_experts.update(top_experts)
                    expert_sets_per_layer[layer_idx] = layer_experts  # For per-prompt, but we aggregate
        # To check if same across prompts, we need per-prompt sets
        # But for this test, we'll collect common sets
    # Compute common experts per layer (intersection across all prompts)
    common_per_layer = []
    for layer_idx in range(num_hidden_layers):
        all_sets = [expert_sets_per_layer[layer_idx] for _ in prompts]  # Need to adjust for per-prompt
        # Wait, in the code above, I overwrote expert_sets_per_layer each time, mistake
    # Fix: need to collect per-prompt
    # Let's adjust the code
    # Reset and redo logic

# Better version with per-prompt collection
def get_common_activated_experts(prompts, label=""):
    per_prompt_expert_sets = []
    predictions = []
    print(f"Processing {len(prompts)} prompts for {label}...")
    for idx, prompt in enumerate(prompts):
        layer_sets = defaultdict(set)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers only with 'Yes' or 'No' for sentiment, or appropriate response for other tasks."},
            {"role": "user", "content": prompt}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_router_logits=True,
                return_dict=True
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            predicted_text = tokenizer.decode([next_token_id]).strip()
            predictions.append(predicted_text)
            if hasattr(outputs, 'router_logits'):
                for layer_idx, layer_logits in enumerate(outputs.router_logits):
                    if layer_logits is None:
                        continue
                    seq_len = layer_logits.shape[1]
                    for pos in range(seq_len):
                        pos_logits = layer_logits[0, pos, :]
                        probs = torch.softmax(pos_logits, dim=-1)
                        top_experts = torch.topk(probs, top_k).indices.tolist()
                        layer_sets[layer_idx].update(top_experts)
        per_prompt_expert_sets.append(layer_sets)
    # Now compute common set per layer (intersection across prompts)
    common_per_layer = {}
    for layer_idx in range(num_hidden_layers):
        sets = [p[layer_idx] for p in per_prompt_expert_sets if layer_idx in p]
        if sets:
            common = set.intersection(*sets) if len(sets) > 1 else sets[0]
            common_per_layer[layer_idx] = sorted(common)
        else:
            common_per_layer[layer_idx] = []
    return common_per_layer, predictions, per_prompt_expert_sets



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

# Define different sets of prompts
# 1. Original sports equipment sentiment
sports_prompts = positive_reviews + negative_reviews  # Assume you have these lists from previous code

# 2. Movie review sentiment (different field)
movie_positive = [
    'Review: "Amazing movie with great plot!" Is this positive? Answer Yes or No only.',
    # Add more, say 5 for test
    'Review: "Loved the acting and story." Is this positive? Answer Yes or No only.',
    'Review: "Best film of the year!" Is this positive? Answer Yes or No only.',
    'Review: "Thrilling and engaging." Is this positive? Answer Yes or No only.',
    'Review: "Wonderful cinematography." Is this positive? Answer Yes or No only.'
]
movie_negative = [
    'Review: "Boring and predictable plot." Is this positive? Answer Yes or No only.',
    'Review: "Poor acting." Is this positive? Answer Yes or No only.',
    'Review: "Waste of time." Is this positive? Answer Yes or No only.',
    'Review: "Bad script." Is this positive? Answer Yes or No only.',
    'Review: "Disappointing ending." Is this positive? Answer Yes or No only.'
]
movie_prompts = movie_positive + movie_negative

# 3. Completely different task: simple math problems
math_prompts = [
    'What is 2 + 2? Answer with number only.',
    'What is 5 * 3? Answer with number only.',
    'What is 10 - 4? Answer with number only.',
    'What is 8 / 2? Answer with number only.',
    'What is 7 + 1? Answer with number only.',
    'What is 9 * 9? Answer with number only.',
    'What is 100 - 50? Answer with number only.',
    'What is 6 / 3? Answer with number only.',
    'What is 4 + 5? Answer with number only.',
    'What is 2 * 10? Answer with number only.'
]

# Run for each set
sports_common, sports_preds, sports_sets = get_common_activated_experts(sports_prompts, "Sports Sentiment")
movie_common, movie_preds, movie_sets = get_common_activated_experts(movie_prompts, "Movie Sentiment")
math_common, math_preds, math_sets = get_common_activated_experts(math_prompts, "Math Problems")

# Compare
print("\nComparison of Common Activated Experts per Layer:")
for layer_idx in range(num_hidden_layers):
    sports_set = set(sports_common.get(layer_idx, []))
    movie_set = set(movie_common.get(layer_idx, []))
    math_set = set(math_common.get(layer_idx, []))
    print(f"Layer {layer_idx:02d}:")
    print(f"  Sports: {sorted(sports_set)}")
    print(f"  Movie: {sorted(movie_set)}")
    print(f"  Math: {sorted(math_set)}")
    overlap_sm = sports_set.intersection(movie_set)
    overlap_all = sports_set.intersection(movie_set, math_set)
    print(f"  Overlap Sports-Movie: {sorted(overlap_sm)} ({len(overlap_sm)} experts)")
    print(f"  Overlap All: {sorted(overlap_all)} ({len(overlap_all)} experts)")
    if sports_set == movie_set:
        print("  Sports and Movie have identical common sets - overlap not due to field.")
    else:
        print("  Sports and Movie differ - overlap may be due to field or task similarity.")
    if sports_set == math_set:
        print("  Sports and Math have identical common sets - likely task-independent.")
    print("")

# To check if within each set, it's perfect overlap (same across prompts)
def check_perfect_overlap(per_prompt_sets):
    for layer_idx in range(num_hidden_layers):
        sets = [p.get(layer_idx, set()) for p in per_prompt_sets]
        first = sets[0]
        if all(s == first for s in sets):
            print(f"Layer {layer_idx}: Perfect overlap across prompts.")
        else:
            print(f"Layer {layer_idx}: Not perfect overlap.")

print("\nCheck perfect overlap within Sports:")
check_perfect_overlap(sports_sets)
print("\nCheck perfect overlap within Movie:")
check_perfect_overlap(movie_sets)
print("\nCheck perfect overlap within Math:")
check_perfect_overlap(math_sets)