# Analyze how many of the 14,848 total experts are actually used
def analyze_expert_usage(detailed_logs, total_experts=14848):
    # Set to store unique (layer, expert_id) pairs - each represents a unique expert
    used_experts = set()
    
    # Track statistics
    total_tokens = 0
    all_layers = set()
    
    # Process each detailed log
    for log_idx, detailed_log in enumerate(detailed_logs):
        print(f"Processing detailed log {log_idx}: '{detailed_log['prompt'][:50]}...'")
        
        # Process each token
        for token_idx, token_log in enumerate(detailed_log["token_logs"]):
            total_tokens += 1
            layer_experts = token_log["layer_experts"]
            
            # Process each layer
            for layer_num, expert_tuples in layer_experts.items():
                all_layers.add(layer_num)
                
                # Each (layer, expert_id) pair is a unique expert
                for expert_tuple in expert_tuples:
                    expert_id = expert_tuple[0]  # First element is expert ID
                    used_experts.add((layer_num, expert_id))
    
    # Calculate statistics
    num_layers = len(all_layers)
    num_used_experts = len(used_experts)
    usage_percentage = (num_used_experts / total_experts) * 100
    
    print(f"\n=== EXPERT USAGE ANALYSIS ===")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Number of layers: {num_layers}")
    print(f"Layer range: {min(all_layers)} to {max(all_layers)}")
    
    print(f"\nTotal experts in model: {total_experts:,}")
    print(f"Unique experts used: {num_used_experts:,}")
    print(f"Unused experts: {total_experts - num_used_experts:,}")
    print(f"Percentage of experts used: {usage_percentage:.2f}%")
    print(f"Percentage of experts unused: {100 - usage_percentage:.2f}%")
    
    # Calculate experts per layer (assuming uniform distribution)
    experts_per_layer = total_experts // num_layers
    print(f"\nAssuming {experts_per_layer} experts per layer:")
    
    # Analyze per-layer usage
    layer_usage = {}
    for layer_num, expert_id in used_experts:
        if layer_num not in layer_usage:
            layer_usage[layer_num] = set()
        layer_usage[layer_num].add(expert_id)
    
    layer_usage_stats = []
    for layer_num in sorted(all_layers):
        used_in_layer = len(layer_usage.get(layer_num, set()))
        percentage = (used_in_layer / experts_per_layer) * 100
        layer_usage_stats.append((layer_num, used_in_layer, percentage))
    
    # Show summary stats
    percentages = [stats[2] for stats in layer_usage_stats]
    avg_usage = sum(percentages) / len(percentages)
    min_usage = min(percentages)
    max_usage = max(percentages)
    
    print(f"\nPer-layer usage statistics:")
    print(f"Average experts used per layer: {avg_usage:.1f}%")
    print(f"Min experts used in a layer: {min_usage:.1f}%")
    print(f"Max experts used in a layer: {max_usage:.1f}%")
    
    # Show first and last few layers
    print(f"\nFirst 5 layers:")
    for layer_num, used_count, percentage in layer_usage_stats[:5]:
        print(f"  Layer {layer_num}: {used_count}/{experts_per_layer} experts ({percentage:.1f}%)")
    
    print(f"\nLast 5 layers:")
    for layer_num, used_count, percentage in layer_usage_stats[-5:]:
        print(f"  Layer {layer_num}: {used_count}/{experts_per_layer} experts ({percentage:.1f}%)")
    
    return {
        'total_experts': total_experts,
        'used_experts': used_experts,
        'num_used': num_used_experts,
        'usage_percentage': usage_percentage,
        'total_tokens': total_tokens,
        'num_layers': num_layers,
        'layer_usage': layer_usage
    }

# Analyze overlap between experts used by different prompts
def analyze_expert_overlap(detailed_logs, total_experts=14848):
    # Get unique experts used by each prompt
    prompt_experts = []
    
    for log_idx, detailed_log in enumerate(detailed_logs):
        used_experts = set()
        prompt = detailed_log['prompt']
        
        # Collect all unique experts for this prompt
        for token_log in detailed_log["token_logs"]:
            layer_experts = token_log["layer_experts"]
            
            for layer_num, expert_tuples in layer_experts.items():
                for expert_tuple in expert_tuples:
                    expert_id = expert_tuple[0]
                    used_experts.add((layer_num, expert_id))
        
        prompt_experts.append({
            'idx': log_idx,
            'prompt': prompt,
            'experts': used_experts,
            'count': len(used_experts)
        })
    
    print("=== EXPERT USAGE BY PROMPT ===")
    for i, data in enumerate(prompt_experts):
        usage_pct = (data['count'] / total_experts) * 100
        print(f"Prompt {i}: '{data['prompt'][:50]}...'")
        print(f"  Experts used: {data['count']:,} ({usage_pct:.2f}%)")
    
    # Calculate overlap between all pairs of prompts
    print(f"\n=== EXPERT OVERLAP ANALYSIS ===")
    
    for i in range(len(prompt_experts)):
        for j in range(i + 1, len(prompt_experts)):
            experts_i = prompt_experts[i]['experts']
            experts_j = prompt_experts[j]['experts']
            
            # Calculate overlap
            overlap = experts_i & experts_j  # intersection
            union = experts_i | experts_j    # union
            
            overlap_count = len(overlap)
            union_count = len(union)
            
            # Calculate different overlap metrics
            jaccard_similarity = overlap_count / union_count if union_count > 0 else 0
            overlap_pct_i = (overlap_count / len(experts_i)) * 100 if len(experts_i) > 0 else 0
            overlap_pct_j = (overlap_count / len(experts_j)) * 100 if len(experts_j) > 0 else 0
            
            print(f"\nPrompt {i} vs Prompt {j}:")
            print(f"  '{prompt_experts[i]['prompt'][:30]}...' vs '{prompt_experts[j]['prompt'][:30]}...'")
            print(f"  Shared experts: {overlap_count:,}")
            print(f"  Total unique experts (union): {union_count:,}")
            print(f"  Jaccard similarity: {jaccard_similarity:.3f}")
            print(f"  Overlap as % of Prompt {i}: {overlap_pct_i:.1f}%")
            print(f"  Overlap as % of Prompt {j}: {overlap_pct_j:.1f}%")
            
            # Unique to each prompt
            unique_to_i = experts_i - experts_j
            unique_to_j = experts_j - experts_i
            print(f"  Unique to Prompt {i}: {len(unique_to_i):,}")
            print(f"  Unique to Prompt {j}: {len(unique_to_j):,}")
    
    # Overall statistics if more than 2 prompts
    if len(prompt_experts) > 2:
        print(f"\n=== OVERALL STATISTICS ===")
        all_experts_used = set()
        for data in prompt_experts:
            all_experts_used.update(data['experts'])
        
        total_unique_experts = len(all_experts_used)
        total_usage_pct = (total_unique_experts / total_experts) * 100
        
        print(f"Total unique experts across all prompts: {total_unique_experts:,}")
        print(f"Total model coverage: {total_usage_pct:.2f}%")
        
        # Find experts used by all prompts (core experts)
        core_experts = prompt_experts[0]['experts']
        for data in prompt_experts[1:]:
            core_experts = core_experts & data['experts']
        
        print(f"Experts used by ALL prompts: {len(core_experts):,}")
        if len(core_experts) > 0:
            core_pct = (len(core_experts) / total_unique_experts) * 100
            print(f"Core experts as % of total used: {core_pct:.1f}%")
    
    return {
        'prompt_experts': prompt_experts,
        'total_experts': total_experts
    }

# Optional: Analyze layer-wise overlap
def analyze_layer_wise_overlap(detailed_logs):
    print(f"\n=== LAYER-WISE OVERLAP ANALYSIS ===")
    
    # Get experts used per layer for each prompt
    prompt_layer_experts = []
    
    for log_idx, detailed_log in enumerate(detailed_logs):
        layer_experts_dict = {}
        
        for token_log in detailed_log["token_logs"]:
            layer_experts = token_log["layer_experts"]
            
            for layer_num, expert_tuples in layer_experts.items():
                if layer_num not in layer_experts_dict:
                    layer_experts_dict[layer_num] = set()
                
                for expert_tuple in expert_tuples:
                    expert_id = expert_tuple[0]
                    layer_experts_dict[layer_num].add(expert_id)
        
        prompt_layer_experts.append({
            'idx': log_idx,
            'prompt': detailed_log['prompt'],
            'layer_experts': layer_experts_dict
        })
    
    # Compare layer by layer
    if len(prompt_layer_experts) >= 2:
        prompt_0 = prompt_layer_experts[0]
        prompt_1 = prompt_layer_experts[1]
        
        all_layers = set(prompt_0['layer_experts'].keys()) | set(prompt_1['layer_experts'].keys())
        
        layer_overlaps = []
        for layer_num in sorted(all_layers):
            experts_0 = prompt_0['layer_experts'].get(layer_num, set())
            experts_1 = prompt_1['layer_experts'].get(layer_num, set())
            
            overlap = len(experts_0 & experts_1)
            total_0 = len(experts_0)
            total_1 = len(experts_1)
            union_size = len(experts_0 | experts_1)
            
            jaccard = overlap / union_size if union_size > 0 else 0
            layer_overlaps.append((layer_num, overlap, jaccard, total_0, total_1))
        
        print(f"Layer-by-layer overlap:")
        print(f"Layer | Shared | Jaccard | Prompt0 | Prompt1")
        print(f"------|--------|---------|---------|--------")
        for layer_num, overlap, jaccard, total_0, total_1 in layer_overlaps[:10]:  # Show first 10
            print(f"{layer_num:5d} | {overlap:6d} | {jaccard:7.3f} | {total_0:7d} | {total_1:7d}")
        
        if len(layer_overlaps) > 10:
            print("...")
            for layer_num, overlap, jaccard, total_0, total_1 in layer_overlaps[-5:]:  # Show last 5
                print(f"{layer_num:5d} | {overlap:6d} | {jaccard:7.3f} | {total_0:7d} | {total_1:7d}")
        
        # Summary stats
        avg_jaccard = sum(x[2] for x in layer_overlaps) / len(layer_overlaps)
        print(f"\nAverage Jaccard similarity across layers: {avg_jaccard:.3f}")

# Run layer-wise analysis
# Assuming you have detailed_logs[0], [1], [2], [3]
overlap_results = analyze_expert_overlap([
    detailed_logs[0],  # French poutine
    detailed_logs[1],  # Pandas 1
    detailed_logs[2],  # English poutine  
    detailed_logs[3]   # Pandas 2
], total_experts=9920)

# Run the analysis
results = analyze_expert_usage([detailed_logs[1]], total_experts=14848)

# Optional: Show some example used experts
# print(f"\nFirst 20 used experts (layer, expert_id):")
# sorted_experts = sorted(list(results['used_experts']))
# for i, (layer, expert_id) in enumerate(sorted_experts[:20]):
#     print(f"  Layer {layer}, Expert {expert_id}")

results = analyze_expert_usage(detailed_logs, total_experts=9920)


for i in detailed_logs:
    print("-" * 40)
    print(f"Prompt: {i['prompt']}")
    print(f"Generated: {i['generated_text']}")



