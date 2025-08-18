import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import numpy as np
import matplotlib.pyplot as plt


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




# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Domain-specific prompts for testing MoE expert specialization

# 30 Math prompts - varying difficulty and types
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

# 30 French to English cooking recipe translation prompts
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

def compare_expert_usage(model, tokenizer, math_prompts, recipe_prompts, max_new_tokens=30):
    """
    Compare expert usage between math and recipe translation domains.
    """
    print("=" * 70)
    print("DOMAIN COMPARISON: Math vs Recipe Translation")
    print("=" * 70)
    
    # Track experts for each domain
    domains = {
        "Math": math_prompts,
        "Recipe Translation": recipe_prompts
    }
    
    domain_results = {}
    
    for domain_name, prompts in domains.items():
        print(f"\n{'='*30}")
        print(f"Processing {domain_name} Domain")
        print(f"{'='*30}")
        
        expert_counts = defaultdict(lambda: defaultdict(int))
        total_selections = defaultdict(int)
        
        for idx, prompt in enumerate(prompts, 1):
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(prompts)} prompts...")
            
            try:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                
                with torch.no_grad():
                    # Try to get router logits
                    outputs = model(
                        **inputs,
                        output_router_logits=True,
                        return_dict=True
                    )
                    
                    # Check if router_logits exist
                    if hasattr(outputs, 'router_logits') and outputs.router_logits:
                        for layer_idx, layer_logits in enumerate(outputs.router_logits):
                            if layer_logits is None:
                                continue
                            
                            # Process router logits
                            if len(layer_logits.shape) == 3:
                                # Shape: (batch, seq, num_experts)
                                probs = torch.softmax(layer_logits[0, -1, :], dim=-1)
                            elif len(layer_logits.shape) == 2:
                                # Shape: (batch, num_experts)
                                probs = torch.softmax(layer_logits[0], dim=-1)
                            else:
                                continue
                            
                            # Get top-k experts
                            k = min(8, len(probs))  # top_k
                            top_experts = torch.topk(probs, k).indices.tolist()
                            
                            for exp in top_experts:
                                expert_counts[layer_idx][exp] += 1
                            total_selections[layer_idx] += k
                            
            except Exception as e:
                if idx == 1:  # Only print error once per domain
                    print(f"  Note: Router logits not available - {str(e)[:50]}")
                continue
        
        domain_results[domain_name] = {
            'counts': expert_counts,
            'totals': total_selections
        }
        
        # Print summary for this domain
        print(f"\n{domain_name} Summary:")
        if expert_counts:
            total_layers = len(expert_counts)
            all_experts = set()
            for layer_counts in expert_counts.values():
                all_experts.update(layer_counts.keys())
            print(f"  - Layers with routing: {total_layers}")
            print(f"  - Unique experts used: {len(all_experts)}")
            print(f"  - Total expert activations: {sum(sum(c.values()) for c in expert_counts.values()):,}")
        else:
            print("  - No routing data captured")
    
    return domain_results

def analyze_domain_differences(domain_results):
    """
    Analyze and visualize differences in expert usage between domains.
    """
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN ANALYSIS")
    print("=" * 70)
    
    if not all(domain_results.values()):
        print("Insufficient data for comparison")
        return
    
    # Calculate expert usage distribution for each domain
    domain_expert_distributions = {}
    
    for domain_name, results in domain_results.items():
        expert_counts = results['counts']
        
        # Aggregate expert usage across all layers
        total_expert_usage = defaultdict(int)
        for layer_counts in expert_counts.values():
            for expert, count in layer_counts.items():
                total_expert_usage[expert] += count
        
        domain_expert_distributions[domain_name] = total_expert_usage
    
    # Find domain-specific experts
    print("\nDomain-Specific Expert Analysis:")
    print("-" * 40)
    
    for domain_name, expert_usage in domain_expert_distributions.items():
        if not expert_usage:
            continue
            
        # Sort experts by usage
        sorted_experts = sorted(expert_usage.items(), key=lambda x: x[1], reverse=True)
        total_usage = sum(expert_usage.values())
        
        print(f"\n{domain_name} - Top 10 Most Used Experts:")
        for i, (expert, count) in enumerate(sorted_experts[:10], 1):
            percentage = (count / total_usage) * 100 if total_usage > 0 else 0
            print(f"  {i:2d}. Expert {expert:3d}: {count:5d} uses ({percentage:5.2f}%)")
    
    # Compare overlap between domains
    if len(domain_expert_distributions) == 2:
        domains = list(domain_expert_distributions.keys())
        experts_1 = set(domain_expert_distributions[domains[0]].keys())
        experts_2 = set(domain_expert_distributions[domains[1]].keys())
        
        overlap = experts_1 & experts_2
        unique_1 = experts_1 - experts_2
        unique_2 = experts_2 - experts_1
        
        print(f"\nExpert Overlap Analysis:")
        print(f"  - Experts used by both domains: {len(overlap)}")
        print(f"  - Experts unique to {domains[0]}: {len(unique_1)}")
        print(f"  - Experts unique to {domains[1]}: {len(unique_2)}")
        
        if unique_1 and len(unique_1) <= 10:
            print(f"\n  {domains[0]}-specific experts: {sorted(unique_1)}")
        if unique_2 and len(unique_2) <= 10:
            print(f"  {domains[1]}-specific experts: {sorted(unique_2)}")
        
        # Calculate specialization score for each expert
        print(f"\nExpert Specialization Scores:")
        print("(Positive = Math specialist, Negative = Recipe specialist)")
        print("-" * 40)
        
        specialization_scores = {}
        for expert in overlap:
            math_usage = domain_expert_distributions[domains[0]].get(expert, 0)
            recipe_usage = domain_expert_distributions[domains[1]].get(expert, 0)
            total = math_usage + recipe_usage
            if total > 0:
                # Score from -1 (pure recipe) to +1 (pure math)
                score = (math_usage - recipe_usage) / total
                specialization_scores[expert] = score
        
        # Show most specialized experts
        sorted_specialists = sorted(specialization_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\nMost Specialized Experts (|score| > 0.3):")
        for expert, score in sorted_specialists[:15]:
            if abs(score) > 0.3:
                domain = domains[0] if score > 0 else domains[1]
                print(f"  Expert {expert:3d}: {score:+.3f} ({domain} specialist)")

def visualize_expert_distribution(domain_results):
    """
    Create visualization of expert usage patterns.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (domain_name, results) in enumerate(domain_results.items()):
            expert_counts = results['counts']
            
            # Aggregate expert usage
            total_usage = defaultdict(int)
            for layer_counts in expert_counts.values():
                for expert, count in layer_counts.items():
                    total_usage[expert] += count
            
            if total_usage:
                # Prepare data for plotting
                experts = sorted(total_usage.keys())
                counts = [total_usage[e] for e in experts]
                
                # Create bar plot
                axes[idx].bar(range(len(experts)), counts, alpha=0.7)
                axes[idx].set_title(f'{domain_name} Expert Usage Distribution')
                axes[idx].set_xlabel('Expert ID')
                axes[idx].set_ylabel('Usage Count')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('expert_distribution_comparison.png', dpi=150)
        print("\nVisualization saved as 'expert_distribution_comparison.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available for visualization")

# Main execution
if __name__ == "__main__":
    print("Starting Domain Comparison Analysis")
    print(f"Math prompts: {len(math_prompts)}")
    print(f"Recipe translation prompts: {len(recipe_translation_prompts)}")
    
    # Assuming model and tokenizer are already loaded
    # Run the comparison
    domain_results = compare_expert_usage(
        model, 
        tokenizer, 
        math_prompts, 
        recipe_translation_prompts,
        max_new_tokens=20  # Reduced for faster testing
    )
    
    # Analyze differences
    analyze_domain_differences(domain_results)
    
    # Optional: Create visualization
    # visualize_expert_distribution(domain_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)