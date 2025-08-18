from main import comprehensive_moe_analysis
  
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


prompts = positive_reviews + negative_reviews


# Run comprehensive analysis
logger, analyzer, pruner = comprehensive_moe_analysis(
    #model_name="Qwen/Qwen2.5-3B-Instruct",  # Use smaller model for testing
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",  # Large model
    prompts=prompts,
    task_name="python_coding",
    max_new_tokens=50,
    temperature=0.0,
    coverage=1.0,  # Keep all used experts
    save_pruned_model=True  # Save the pruned model
)

# Example 2: Analyze unused experts
unused = analyzer.get_unused_experts()
print(f"\nUnused experts per layer:")
for layer_idx in sorted(unused.keys()):
    print(f"  Layer {layer_idx}: {len(unused[layer_idx])} unused experts")
