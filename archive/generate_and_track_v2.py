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

# Domain-specific prompts
math_prompts = [
    "What is 2 + 2?",
    "Calculate 15 × 7",
    "What is 144 divided by 12?",
    "Solve for x: 2x + 5 = 13",
    "What is the square root of 64?",
    "Calculate 23% of 450",
    "What is 3 to the power of 4?",
    "Find the derivative of x^2",
    "What is the integral of 2x?",
    "Calculate the sine of 30 degrees",
    "What is the factorial of 6?",
    "Solve: 3x - 7 = 14",
    "What is the area of a circle with radius 5?",
    "Calculate the hypotenuse of a right triangle with sides 3 and 4",
    "What is 1/3 + 1/4?",
    "Convert 0.75 to a fraction",
    "What is the greatest common divisor of 24 and 36?",
    "Calculate the compound interest on $1000 at 5% for 3 years",
    "What is the cosine of 45 degrees?",
    "Find the limit as x approaches 0 of sin(x)/x",
    "What is the logarithm base 10 of 100?",
    "Calculate the mean of: 12, 15, 18, 21, 24",
    "What is the standard deviation of: 2, 4, 6, 8, 10?",
    "Solve the quadratic equation: x^2 - 5x + 6 = 0",
    "What is the volume of a sphere with radius 3?",
    "Calculate the binomial coefficient C(10,3)",
    "What is the sum of the first 10 natural numbers?",
    "Find the derivative of sin(x)",
    "What is e to the power of 0?",
    "Calculate the determinant of [[1,2],[3,4]]"
]

recipe_translation_prompts = [
    "Translate from French to English: Préchauffez le four à 180°C",
    "Translate this French cooking instruction: Battez les œufs avec le sucre jusqu'à ce que le mélange blanchisse",
    "Translate from French: Faites fondre le beurre dans une poêle",
    "Translate this recipe step: Pétrissez la pâte pendant 10 minutes",
    "Translate from French to English: Laissez reposer la pâte au réfrigérateur pendant une heure",
    "Translate: Émincez finement les oignons",
    "Translate this French instruction: Faites revenir l'ail dans l'huile d'olive",
    "Translate from French: Ajoutez une pincée de sel et de poivre",
    "Translate: Mélangez délicatement avec une spatule en bois",
    "Translate this cooking term: Faites cuire à feu doux pendant 20 minutes",
    "Translate from French to English: Incorporez la farine petit à petit",
    "Translate: Montez les blancs en neige ferme",
    "Translate this instruction: Versez la préparation dans un moule beurré",
    "Translate from French: Faites mariner la viande pendant 2 heures",
    "Translate: Découpez les légumes en julienne",
    "Translate this French recipe: Faites réduire la sauce de moitié",
    "Translate from French to English: Saupoudrez de fromage râpé",
    "Translate: Laissez mijoter à couvert pendant 45 minutes",
    "Translate this instruction: Fouettez la crème jusqu'à obtenir une chantilly",
    "Translate from French: Déglacez avec du vin blanc",
    "Translate: Enfournez pour 25 minutes jusqu'à ce que ce soit doré",
    "Translate this French term: Faites blanchir les légumes 3 minutes",
    "Translate from French to English: Assaisonnez selon votre goût",
    "Translate: Remuez constamment pour éviter que ça n'attache",
    "Translate this recipe step: Laissez refroidir complètement avant de démouler",
    "Translate from French: Parsemez d'herbes fraîches ciselées",
    "Translate: Faites caraméliser le sucre jusqu'à obtenir une couleur ambrée",
    "Translate this instruction: Égouttez et réservez l'eau de cuisson",
    "Translate from French to English: Nappez de sauce et servez immédiatement",
    "Translate: Rectifiez l'assaisonnement si nécessaire"
]

def generate_and_track_v2(prompts, max_new_tokens=20):
    """
    Forward pass tracking method that was working.
    """
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_selections = defaultdict(int)
    
    for prompt_idx, prompt in enumerate(prompts):
        if (prompt_idx + 1) % 10 == 0:
            print(f"    Processed {prompt_idx + 1}/{len(prompts)} prompts...")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        # Generate tokens one by one to track router decisions
        generated_ids = inputs.input_ids.clone()
        
        for step in range(min(max_new_tokens, 10)):  # Limit tokens for speed
            with torch.no_grad():
                try:
                    # Forward pass with router logits
                    outputs = model(
                        input_ids=generated_ids,
                        output_router_logits=True,
                        return_dict=True
                    )
                    
                    # Get next token
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Process router logits if available
                    if hasattr(outputs, 'router_logits') and outputs.router_logits:
                        for layer_idx, layer_logits in enumerate(outputs.router_logits):
                            if layer_logits is None:
                                continue
                            
                            # Handle different tensor shapes
                            if len(layer_logits.shape) == 3:  # (batch, seq, experts)
                                last_token_logits = layer_logits[:, -1, :]
                            elif len(layer_logits.shape) == 2:  # (batch, experts)
                                last_token_logits = layer_logits
                            else:
                                continue
                            
                            # Convert to probabilities
                            probs = torch.softmax(last_token_logits[0], dim=-1)
                            
                            # Get top-k experts
                            k = min(top_k, len(probs))
                            top_experts = torch.topk(probs, k).indices.tolist()
                            
                            # Track expert usage
                            for exp in top_experts:
                                expert_counts[layer_idx][exp] += 1
                            total_selections[layer_idx] += k
                    
                    # Stop if we hit EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                        
                except Exception as e:
                    if prompt_idx == 0 and step == 0:
                        print(f"    Note: Error during tracking - {str(e)[:50]}")
                    break
    
    return expert_counts, total_selections

def analyze_domain_comparison(math_counts, recipe_counts):
    """
    Compare expert usage between math and recipe translation domains.
    """
    print("\n" + "=" * 70)
    print("DOMAIN COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Aggregate expert usage across all layers for each domain
    math_total = defaultdict(int)
    recipe_total = defaultdict(int)
    
    for layer_counts in math_counts.values():
        for expert, count in layer_counts.items():
            math_total[expert] += count
    
    for layer_counts in recipe_counts.values():
        for expert, count in layer_counts.items():
            recipe_total[expert] += count
    
    # Get all experts used
    all_experts = set(math_total.keys()) | set(recipe_total.keys())
    
    if not all_experts:
        print("No expert data collected for comparison")
        return
    
    # Calculate statistics
    math_only = set(math_total.keys()) - set(recipe_total.keys())
    recipe_only = set(recipe_total.keys()) - set(math_total.keys())
    shared = set(math_total.keys()) & set(recipe_total.keys())
    
    print(f"\nExpert Usage Overview:")
    print(f"  • Total unique experts used: {len(all_experts)}")
    print(f"  • Experts used by both domains: {len(shared)}")
    print(f"  • Math-only experts: {len(math_only)}")
    print(f"  • Recipe-only experts: {len(recipe_only)}")
    
    # Show top experts for each domain
    print(f"\n{'='*35}")
    print("TOP EXPERTS BY DOMAIN")
    print(f"{'='*35}")
    
    # Math domain top experts
    math_sorted = sorted(math_total.items(), key=lambda x: x[1], reverse=True)
    total_math = sum(math_total.values())
    
    print(f"\nMath Domain - Top 10 Experts:")
    print(f"(Total activations: {total_math:,})")
    for i, (expert, count) in enumerate(math_sorted[:10], 1):
        percentage = (count / total_math * 100) if total_math > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:5d} ({percentage:5.2f}%)")
    
    # Recipe domain top experts
    recipe_sorted = sorted(recipe_total.items(), key=lambda x: x[1], reverse=True)
    total_recipe = sum(recipe_total.values())
    
    print(f"\nRecipe Translation - Top 10 Experts:")
    print(f"(Total activations: {total_recipe:,})")
    for i, (expert, count) in enumerate(recipe_sorted[:10], 1):
        percentage = (count / total_recipe * 100) if total_recipe > 0 else 0
        print(f"  {i:2d}. Expert {expert:3d}: {count:5d} ({percentage:5.2f}%)")
    
    # Calculate specialization scores
    print(f"\n{'='*35}")
    print("EXPERT SPECIALIZATION ANALYSIS")
    print(f"{'='*35}")
    
    specialization_scores = {}
    for expert in shared:
        math_count = math_total[expert]
        recipe_count = recipe_total[expert]
        total = math_count + recipe_count
        
        if total > 10:  # Only consider experts with significant usage
            # Score from -1 (pure recipe) to +1 (pure math)
            score = (math_count - recipe_count) / total
            specialization_scores[expert] = {
                'score': score,
                'math_count': math_count,
                'recipe_count': recipe_count,
                'total': total
            }
    
    # Sort by absolute specialization
    sorted_specialists = sorted(specialization_scores.items(), 
                               key=lambda x: abs(x[1]['score']), 
                               reverse=True)
    
    print(f"\nMost Specialized Experts:")
    print("(Score: +1.0 = pure math, -1.0 = pure recipe, 0 = balanced)")
    print("-" * 60)
    
    # Show strong specialists
    strong_math = []
    strong_recipe = []
    balanced = []
    
    for expert, data in sorted_specialists:
        if data['score'] > 0.3:
            strong_math.append((expert, data))
        elif data['score'] < -0.3:
            strong_recipe.append((expert, data))
        elif abs(data['score']) < 0.1 and data['total'] > 50:
            balanced.append((expert, data))
    
    if strong_math:
        print(f"\n🧮 Strong Math Specialists (score > 0.3):")
        for expert, data in strong_math[:5]:
            print(f"  Expert {expert:3d}: score={data['score']:+.3f} "
                  f"(math:{data['math_count']} vs recipe:{data['recipe_count']})")
    
    if strong_recipe:
        print(f"\n🍳 Strong Recipe Specialists (score < -0.3):")
        for expert, data in strong_recipe[:5]:
            print(f"  Expert {expert:3d}: score={data['score']:+.3f} "
                  f"(math:{data['math_count']} vs recipe:{data['recipe_count']})")
    
    if balanced:
        print(f"\n⚖️  Balanced Generalists (|score| < 0.1, usage > 50):")
        for expert, data in balanced[:5]:
            print(f"  Expert {expert:3d}: score={data['score']:+.3f} "
                  f"(math:{data['math_count']} vs recipe:{data['recipe_count']})")
    
    # Domain-exclusive experts
    if math_only and len(math_only) <= 15:
        print(f"\n📊 Math-Exclusive Experts: {sorted(math_only)}")
    elif math_only:
        print(f"\n📊 Math-Exclusive Experts: {len(math_only)} experts (too many to list)")
    
    if recipe_only and len(recipe_only) <= 15:
        print(f"\n📝 Recipe-Exclusive Experts: {sorted(recipe_only)}")
    elif recipe_only:
        print(f"\n📝 Recipe-Exclusive Experts: {len(recipe_only)} experts (too many to list)")

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("MoE DOMAIN SPECIALIZATION ANALYSIS")
    print("=" * 70)
    print(f"\nAnalyzing {len(math_prompts)} math prompts")
    print(f"Analyzing {len(recipe_translation_prompts)} recipe translation prompts")
    
    # Process Math domain
    print("\n" + "="*35)
    print("Processing Math Domain")
    print("="*35)
    math_counts, math_totals = generate_and_track_v2(math_prompts, max_new_tokens=15)
    
    math_layers = len(math_counts)
    math_experts = set()
    for layer_counts in math_counts.values():
        math_experts.update(layer_counts.keys())
    
    print(f"\nMath Results:")
    print(f"  • Layers with routing: {math_layers}")
    print(f"  • Unique experts activated: {len(math_experts)}")
    print(f"  • Total expert selections: {sum(math_totals.values()):,}")
    
    # Process Recipe Translation domain
    print("\n" + "="*35)
    print("Processing Recipe Translation Domain")
    print("="*35)
    recipe_counts, recipe_totals = generate_and_track_v2(recipe_translation_prompts, max_new_tokens=15)
    
    recipe_layers = len(recipe_counts)
    recipe_experts = set()
    for layer_counts in recipe_counts.values():
        recipe_experts.update(layer_counts.keys())
    
    print(f"\nRecipe Translation Results:")
    print(f"  • Layers with routing: {recipe_layers}")
    print(f"  • Unique experts activated: {len(recipe_experts)}")
    print(f"  • Total expert selections: {sum(recipe_totals.values()):,}")
    
    # Compare domains
    if math_counts and recipe_counts:
        analyze_domain_comparison(math_counts, recipe_counts)
    else:
        print("\n⚠️  Insufficient data collected for comparison")
        print("This might indicate that router_logits are not being exposed properly")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)