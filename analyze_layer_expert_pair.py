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

# Run the analysis
results = analyze_expert_usage([detailed_logs[1]], total_experts=14848)

# Optional: Show some example used experts
# print(f"\nFirst 20 used experts (layer, expert_id):")
# sorted_experts = sorted(list(results['used_experts']))
# for i, (layer, expert_id) in enumerate(sorted_experts[:20]):
#     print(f"  Layer {layer}, Expert {expert_id}")

results = analyze_expert_usage(detailed_logs, total_experts=14848)