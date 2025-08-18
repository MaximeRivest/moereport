import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import warnings
import json
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MoETaskProfiler:
    """
    Universal profiler for analyzing MoE expert usage on any task.
    Can handle single-token classification or multi-token generation.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507", 
                 device_map: str = "auto", 
                 dtype: torch.dtype = torch.bfloat16):
        """Initialize the model and tokenizer."""
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
        print("Model loaded.")
        
        # Get model configuration
        self.num_experts = self.model.config.num_routed_experts if hasattr(self.model.config, 'num_routed_experts') else 128
        self.top_k = self.model.config.num_experts_per_tok if hasattr(self.model.config, 'num_experts_per_tok') else 8
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.total_experts = self.num_hidden_layers * self.num_experts
        
        print(f"\nModel Configuration:")
        print(f"  • Experts per layer: {self.num_experts}")
        print(f"  • Experts per token: {self.top_k}")
        print(f"  • Hidden layers: {self.num_hidden_layers}")
        print(f"  • Total experts: {self.total_experts:,}")
    
    def profile_task(self, 
                     prompts: List[str], 
                     task_name: str = "Task",
                     max_new_tokens: int = 1,
                     skip_initial_newline: bool = True,
                     temperature: float = 0.0,
                     analyze_categories: Optional[Dict[str, List[int]]] = None) -> Dict:
        """
        Profile expert usage for a given task.
        
        Args:
            prompts: List of prompts to analyze
            task_name: Name of the task for reporting
            max_new_tokens: Number of tokens to generate (1 for classification, more for generation)
            skip_initial_newline: Whether to skip initial newline if present
            temperature: Sampling temperature (0 for deterministic)
            analyze_categories: Optional dict mapping category names to prompt indices
                               e.g., {"positive": [0,1,2], "negative": [3,4,5]}
        
        Returns:
            Dictionary containing profiling results
        """
        print(f"\n{'='*70}")
        print(f"PROFILING: {task_name}")
        print(f"{'='*70}")
        print(f"  • Prompts: {len(prompts)}")
        print(f"  • Max tokens: {max_new_tokens}")
        print(f"  • Skip newline: {skip_initial_newline}")
        
        # Track expert usage
        all_expert_usage = defaultdict(lambda: defaultdict(int))  # layer -> expert -> count
        position_expert_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # position -> layer -> expert -> count
        prompt_results = []
        
        # Category tracking if provided
        category_expert_usage = {}
        if analyze_categories:
            for category in analyze_categories:
                category_expert_usage[category] = defaultdict(lambda: defaultdict(int))
        
        print(f"\nProcessing prompts...")
        
        for prompt_idx, prompt in enumerate(prompts):
            if (prompt_idx + 1) % max(1, len(prompts) // 10) == 0:
                print(f"  Progress: {prompt_idx + 1}/{len(prompts)}")
            
            # Determine category if applicable
            current_category = None
            if analyze_categories:
                for category, indices in analyze_categories.items():
                    if prompt_idx in indices:
                        current_category = category
                        break
            
            # Process prompt
            result = self._process_single_prompt(
                prompt, 
                max_new_tokens, 
                skip_initial_newline,
                temperature
            )
            
            prompt_results.append(result)
            
            # Aggregate expert usage
            for token_data in result['tokens']:
                position = token_data['position']
                for layer_idx, experts in token_data['experts'].items():
                    for expert in experts:
                        all_expert_usage[layer_idx][expert] += 1
                        position_expert_usage[position][layer_idx][expert] += 1
                        
                        if current_category:
                            category_expert_usage[current_category][layer_idx][expert] += 1
        
        # Analyze results
        analysis = self._analyze_results(
            all_expert_usage,
            position_expert_usage,
            category_expert_usage,
            prompt_results,
            task_name,
            analyze_categories
        )
        
        return analysis
    
    def _process_single_prompt(self, 
                               prompt: str, 
                               max_new_tokens: int,
                               skip_initial_newline: bool,
                               temperature: float) -> Dict:
        """Process a single prompt and track expert usage."""
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        current_ids = inputs.input_ids
        current_attention = inputs.attention_mask
        
        tokens_data = []
        generated_text = ""
        tokens_generated = 0
        skipped_newline = False
        
        while tokens_generated < max_new_tokens:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=current_attention,
                    output_router_logits=True,
                    return_dict=True
                )
                
                # Get next token
                if temperature == 0:
                    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                else:
                    probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
                
                next_token_text = self.tokenizer.decode(next_token_id.item())
                
                # Check if we should skip this token
                if skip_initial_newline and not skipped_newline and next_token_text.strip() == "":
                    skipped_newline = True
                    # Still need to update input for next iteration
                    # Fix: Ensure next_token_id has correct dimensions
                    if next_token_id.dim() == 0:
                        next_token_id = next_token_id.unsqueeze(0)
                    if next_token_id.dim() == 1:
                        next_token_id = next_token_id.unsqueeze(0)
                    current_ids = torch.cat([current_ids, next_token_id], dim=1)
                    current_attention = torch.cat([
                        current_attention,
                        torch.ones((1, 1), device=current_attention.device, dtype=current_attention.dtype)
                    ], dim=1)
                    continue
                
                generated_text += next_token_text
                position = current_ids.shape[1] - 1
                
                # Track expert usage for this token
                token_experts = {}
                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    for layer_idx, layer_logits in enumerate(outputs.router_logits):
                        if layer_logits is None:
                            continue
                        
                        # Get routing for the generating position
                        if len(layer_logits.shape) == 2:  # [seq_len, num_experts]
                            if position < layer_logits.shape[0]:
                                pos_logits = layer_logits[position]
                            else:
                                pos_logits = layer_logits[-1]
                        else:  # [batch, seq_len, num_experts]
                            if position < layer_logits.shape[1]:
                                pos_logits = layer_logits[0, position]
                            else:
                                pos_logits = layer_logits[0, -1]
                        
                        # Get top-k experts
                        probs = torch.softmax(pos_logits, dim=-1)
                        top_experts = torch.topk(probs, self.top_k).indices.tolist()
                        token_experts[layer_idx] = top_experts
                
                tokens_data.append({
                    'token': next_token_text,
                    'position': position,
                    'experts': token_experts
                })
                
                tokens_generated += 1
                
                # Check for end token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # Update input for next iteration
                # Fix: Ensure next_token_id has correct dimensions
                if next_token_id.dim() == 0:
                    next_token_id = next_token_id.unsqueeze(0)
                if next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(0)
                current_ids = torch.cat([current_ids, next_token_id], dim=1)
                current_attention = torch.cat([
                    current_attention,
                    torch.ones((1, 1), device=current_attention.device, dtype=current_attention.dtype)
                ], dim=1)
        
        return {
            'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
            'generated': generated_text,
            'tokens': tokens_data
        }
    
    def _analyze_results(self,
                        all_expert_usage: Dict,
                        position_expert_usage: Dict,
                        category_expert_usage: Dict,
                        prompt_results: List,
                        task_name: str,
                        analyze_categories: Optional[Dict]) -> Dict:
        """Analyze the collected expert usage data."""
        
        # Calculate unique experts used
        unique_experts = set()
        for layer_idx, experts in all_expert_usage.items():
            for expert_idx in experts.keys():
                unique_experts.add((layer_idx, expert_idx))
        
        utilization = len(unique_experts) / self.total_experts * 100
        
        # Find most used experts globally
        expert_counts = defaultdict(int)
        for layer_idx, experts in all_expert_usage.items():
            for expert_idx, count in experts.items():
                expert_counts[(layer_idx, expert_idx)] = count
        
        top_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Category analysis if applicable
        category_analysis = {}
        if analyze_categories and category_expert_usage:
            category_analysis = self._analyze_categories(category_expert_usage, analyze_categories)
        
        # Position diversity analysis
        position_diversity = {}
        for pos, layers in position_expert_usage.items():
            unique_at_pos = set()
            for layer_idx, experts in layers.items():
                for expert_idx in experts.keys():
                    unique_at_pos.add((layer_idx, expert_idx))
            position_diversity[pos] = len(unique_at_pos)
        
        # Build analysis report
        analysis = {
            'task_name': task_name,
            'num_prompts': len(prompt_results),
            'total_experts': self.total_experts,
            'unique_experts_used': len(unique_experts),
            'utilization_percent': utilization,
            'potential_savings_percent': 100 - utilization,
            'top_experts': top_experts,
            'position_diversity': position_diversity,
            'category_analysis': category_analysis,
            'sample_outputs': [r['generated'][:100] for r in prompt_results[:5]],
            'expert_usage_by_layer': {k: len(v) for k, v in all_expert_usage.items()},
            'raw_expert_usage': dict(all_expert_usage)  # For advanced analysis
        }
        
        return analysis
    
    def _analyze_categories(self, 
                           category_expert_usage: Dict,
                           analyze_categories: Dict) -> Dict:
        """Analyze expert usage by category."""
        category_analysis = {}
        
        # Get unique experts per category
        for category, expert_usage in category_expert_usage.items():
            unique_experts = set()
            for layer_idx, experts in expert_usage.items():
                for expert_idx in experts.keys():
                    unique_experts.add((layer_idx, expert_idx))
            
            category_analysis[category] = {
                'unique_experts': len(unique_experts),
                'num_prompts': len(analyze_categories[category])
            }
        
        # Find category-exclusive experts
        if len(category_expert_usage) == 2:  # Binary classification
            categories = list(category_expert_usage.keys())
            cat1_experts = set()
            cat2_experts = set()
            
            for layer_idx, experts in category_expert_usage[categories[0]].items():
                for expert_idx in experts.keys():
                    cat1_experts.add((layer_idx, expert_idx))
            
            for layer_idx, experts in category_expert_usage[categories[1]].items():
                for expert_idx in experts.keys():
                    cat2_experts.add((layer_idx, expert_idx))
            
            exclusive_cat1 = cat1_experts - cat2_experts
            exclusive_cat2 = cat2_experts - cat1_experts
            shared = cat1_experts & cat2_experts
            
            category_analysis['comparison'] = {
                f'{categories[0]}_exclusive': len(exclusive_cat1),
                f'{categories[1]}_exclusive': len(exclusive_cat2),
                'shared': len(shared),
                'overlap_percent': len(shared) / len(cat1_experts | cat2_experts) * 100 if (cat1_experts | cat2_experts) else 0
            }
        
        return category_analysis
    
    def generate_pruning_report(self, analysis: Dict) -> str:
        """Generate a detailed pruning/optimization report."""
        report = []
        report.append("\n" + "="*70)
        report.append(f"MoE PRUNING/OPTIMIZATION REPORT")
        report.append("="*70)
        report.append(f"\nTask: {analysis['task_name']}")
        report.append(f"Prompts analyzed: {analysis['num_prompts']}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-"*40)
        report.append(f"• Total experts in model: {analysis['total_experts']:,}")
        report.append(f"• Experts used for task: {analysis['unique_experts_used']:,}")
        report.append(f"• Utilization: {analysis['utilization_percent']:.2f}%")
        report.append(f"• Potential VRAM savings: {analysis['potential_savings_percent']:.2f}%")
        
        # Memory estimates (assuming each expert is ~50MB for a 30B model)
        expert_size_gb = 0.05  # Rough estimate
        total_memory_gb = analysis['total_experts'] * expert_size_gb
        used_memory_gb = analysis['unique_experts_used'] * expert_size_gb
        saved_memory_gb = total_memory_gb - used_memory_gb
        
        report.append("")
        report.append("MEMORY IMPACT ESTIMATES")
        report.append("-"*40)
        report.append(f"• Estimated total expert memory: {total_memory_gb:.1f} GB")
        report.append(f"• Memory for used experts: {used_memory_gb:.1f} GB")
        report.append(f"• Potential memory savings: {saved_memory_gb:.1f} GB")
        
        # Pruning recommendations
        report.append("")
        report.append("PRUNING RECOMMENDATIONS")
        report.append("-"*40)
        
        if analysis['utilization_percent'] < 10:
            report.append("✅ HIGHLY SUITABLE for expert pruning")
            report.append("   - Very low expert utilization indicates task-specific routing")
            report.append("   - Consider creating a task-specific model variant")
            report.append("   - Could potentially run on much smaller hardware")
        elif analysis['utilization_percent'] < 25:
            report.append("⚠️ MODERATELY SUITABLE for expert pruning")
            report.append("   - Significant savings possible but with some risk")
            report.append("   - Consider keeping a buffer of rarely-used experts")
            report.append("   - Test thoroughly on edge cases")
        else:
            report.append("❌ LIMITED BENEFIT from expert pruning")
            report.append("   - Task uses diverse experts across the model")
            report.append("   - Consider other optimization strategies")
            report.append("   - May benefit more from quantization than pruning")
        
        # Category analysis if available
        if analysis['category_analysis'] and 'comparison' in analysis['category_analysis']:
            report.append("")
            report.append("CATEGORY SPECIALIZATION")
            report.append("-"*40)
            comp = analysis['category_analysis']['comparison']
            report.append(f"• Expert overlap between categories: {comp['overlap_percent']:.1f}%")
            
            for key, value in comp.items():
                if 'exclusive' in key:
                    category = key.replace('_exclusive', '')
                    report.append(f"• {category.capitalize()}-exclusive experts: {value}")
        
        # Top experts
        report.append("")
        report.append("CRITICAL EXPERTS (Top 10 by usage)")
        report.append("-"*40)
        for i, ((layer, expert), count) in enumerate(analysis['top_experts'][:10], 1):
            report.append(f"{i:2d}. Layer {layer:2d}, Expert {expert:3d}: {count:4d} activations")
        
        # Implementation strategy
        report.append("")
        report.append("IMPLEMENTATION STRATEGY")
        report.append("-"*40)
        report.append("1. Create expert whitelist from this analysis")
        report.append("2. Modify model loading to skip non-whitelisted experts")
        report.append("3. Implement fallback for unexpected expert requests")
        report.append("4. Test on validation set to ensure quality maintained")
        report.append("5. Consider gradual pruning (remove least-used 50% first)")
        
        # Sample outputs
        report.append("")
        report.append("SAMPLE OUTPUTS")
        report.append("-"*40)
        for i, output in enumerate(analysis['sample_outputs'], 1):
            report.append(f"{i}. {output}")
        
        return "\n".join(report)
    
    def export_expert_list(self, analysis: Dict, output_file: str = "expert_whitelist.json"):
        """Export the list of used experts for pruning implementation."""
        expert_list = []
        for layer_idx, experts in analysis['raw_expert_usage'].items():
            for expert_idx in experts.keys():
                expert_list.append({
                    'layer': layer_idx,
                    'expert': expert_idx,
                    'usage_count': experts[expert_idx]
                })
        
        # Sort by usage count
        expert_list.sort(key=lambda x: x['usage_count'], reverse=True)
        
        export_data = {
            'task_name': analysis['task_name'],
            'total_experts': analysis['total_experts'],
            'used_experts': analysis['unique_experts_used'],
            'utilization_percent': analysis['utilization_percent'],
            'expert_whitelist': expert_list
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Expert whitelist exported to {output_file}")
        return output_file


# Example usage functions
def example_sentiment_classification():
    """Example: Sentiment classification task."""
    profiler = MoETaskProfiler()
    
    # Define prompts
    positive_prompts = [
        'Review: "This product is amazing!" Sentiment:',
        'Review: "Best purchase ever!" Sentiment:',
        'Review: "Highly recommend!" Sentiment:',
        'Review: "Excellent quality!" Sentiment:',
        'Review: "Love it!" Sentiment:',
    ]
    
    negative_prompts = [
        'Review: "Terrible product!" Sentiment:',
        'Review: "Waste of money!" Sentiment:',
        'Review: "Very disappointed!" Sentiment:',
        'Review: "Poor quality!" Sentiment:',
        'Review: "Do not buy!" Sentiment:',
    ]
    
    all_prompts = positive_prompts + negative_prompts
    
    # Define categories for analysis
    categories = {
        'positive': list(range(len(positive_prompts))),
        'negative': list(range(len(positive_prompts), len(all_prompts)))
    }
    
    # Profile the task
    analysis = profiler.profile_task(
        prompts=all_prompts,
        task_name="Sentiment Classification",
        max_new_tokens=1,
        skip_initial_newline=True,
        analyze_categories=categories
    )
    
    # Generate report
    report = profiler.generate_pruning_report(analysis)
    print(report)
    
    # Export expert list
    profiler.export_expert_list(analysis, "sentiment_experts.json")
    
    return analysis


def example_translation():
    """Example: Translation task."""
    profiler = MoETaskProfiler()
    
    prompts = [
        'Translate to French: "Hello, how are you?"',
        'Translate to French: "Good morning!"',
        'Translate to French: "Thank you very much."',
        'Translate to French: "See you tomorrow."',
        'Translate to French: "I love programming."',
    ]
    
    # Profile the task (multi-token generation)
    analysis = profiler.profile_task(
        prompts=prompts,
        task_name="English to French Translation",
        max_new_tokens=20,  # Generate more tokens for translation
        skip_initial_newline=True
    )
    
    # Generate report
    report = profiler.generate_pruning_report(analysis)
    print(report)
    
    # Export expert list
    profiler.export_expert_list(analysis, "translation_experts.json")
    
    return analysis


def example_custom_task(prompts: List[str], 
                        task_name: str,
                        max_tokens: int = 10,
                        categories: Optional[Dict] = None):
    """Example: Custom task profiling."""
    profiler = MoETaskProfiler()
    
    # Profile the task
    analysis = profiler.profile_task(
        prompts=prompts,
        task_name=task_name,
        max_new_tokens=max_tokens,
        skip_initial_newline=True,
        analyze_categories=categories
    )
    
    # Generate report
    report = profiler.generate_pruning_report(analysis)
    print(report)
    
    # Export expert list
    filename = task_name.lower().replace(" ", "_") + "_experts.json"
    profiler.export_expert_list(analysis, filename)
    
    return analysis


if __name__ == "__main__":
    print("MoE Task Profiler - Choose an example:")
    print("1. Sentiment Classification")
    print("2. Translation")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        example_sentiment_classification()
    elif choice == "2":
        example_translation()
    else:
        print("Exiting...")
        
    print("\n✅ Profiling complete!")
    print("Check the generated JSON file for the expert whitelist.")
    print("Use this list to implement selective expert loading for VRAM optimization.")