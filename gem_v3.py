import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LogNorm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_LENGTH = 512 # Limit prompt length for efficiency

# Define tasks and representative prompts
# Increase the diversity and number of prompts for more robust results
TASK_PROMPTS = {
    "Coding": [
        "Write a Python function to calculate the factorial of a number.",
        "Explain the difference between lists and tuples in Python.",
        "How do I center a div in CSS?",
        "What is a segmentation fault in C++ and how do I debug it?",
        "Implement a binary search algorithm in Java.",
        "What is the purpose of a Dockerfile?",
        "Explain asynchronous programming in JavaScript."
    ],
    "Cooking": [
        "How do I make spaghetti carbonara from scratch?",
        "What is the best temperature to bake sourdough bread?",
        "Give me a recipe for a vegan lasagna.",
        "How long should I boil potatoes for mashing?",
        "What are the key ingredients in traditional Thai green curry?",
        "Tips for perfectly searing a steak?",
        "How to properly knead dough for pizza."
    ],
    "Translation_EN_FR": [
        "Translate 'Hello world' into French.",
        "What is the French word for 'computer'?",
        "How do you say 'Thank you very much' in French?",
        "Translate: 'The quick brown fox jumps over the lazy dog.'",
        "How do you ask for directions in French?",
        "Translate 'I love learning new languages' to French.",
        "Provide the French translation for 'Artificial Intelligence'."
    ],
    "General_Knowledge": [
        "What is the capital of Japan?",
        "Explain the theory of relativity in simple terms.",
        "Who wrote 'To Kill a Mockingbird'?",
        "What are the main causes of climate change?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
        "Describe the history of the Roman Empire."
    ]
}

# --- Model Loading ---
print(f"Loading model: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Determine the appropriate dtype
    try:
        dtype = torch.float8_e4m3fn if "FP8" in MODEL_NAME else torch.bfloat16
    except AttributeError:
        # Fallback if float8 types are not supported by the torch version
        dtype = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=dtype,
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}. Attempting fallback to bfloat16.")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded with fallback dtype (bfloat16).")
    except Exception as e2:
        print(f"Failed to load model: {e2}")
        exit()

# Get model configuration
num_experts_per_layer = getattr(model.config, 'num_routed_experts', 128)
top_k = getattr(model.config, 'num_experts_per_tok', 8)
num_hidden_layers = model.config.num_hidden_layers

print(f"\nModel Configuration:")
print(f"  - Hidden layers: {num_hidden_layers}")
print(f"  - Experts per layer: {num_experts_per_layer}")
print(f"  - Experts activated per token (Top-K): {top_k}\n")

# --- Analysis Functions ---

def analyze_prompt_routing(prompt, tokenizer, model, top_k, num_experts_per_layer):
    """
    Analyzes routing for a single prompt and returns expert usage statistics efficiently.
    """
    # Apply chat template and tokenize
    messages = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
    
    # Dictionary to store layer-wise counts
    expert_usage = {}
    
    with torch.no_grad():
        # Perform forward pass to get router logits during prompt processing
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_router_logits=True,
            return_dict=True
        )
        
        if hasattr(outputs, 'router_logits') and outputs.router_logits:
            for layer_idx, layer_logits in enumerate(outputs.router_logits):
                if layer_logits is None:
                    continue
                    
                # Handle dimension squeezing (if batch=1, shape might be [seq_len, num_experts])
                if len(layer_logits.shape) == 3:
                     # Shape: [batch, seq_len, num_experts]. Flatten to [batch*seq_len, num_experts]
                    logits_to_process = layer_logits.reshape(-1, num_experts_per_layer)
                elif len(layer_logits.shape) == 2:
                    # Shape: [seq_len, num_experts]
                    logits_to_process = layer_logits
                else:
                    continue

                # Get indices of the top-k experts. 
                # Softmax is monotonic, so topk(logits) == topk(softmax(logits)). We skip softmax for efficiency.
                top_experts = torch.topk(logits_to_process, top_k, dim=-1).indices # Shape: [seq_len, top_k]
                
                # Efficiently count expert activations on GPU using PyTorch
                # We use torch.bincount for efficient counting
                flat_indices = top_experts.reshape(-1)
                layer_counts = torch.bincount(flat_indices, minlength=num_experts_per_layer)
                
                expert_usage[layer_idx] = layer_counts.cpu().numpy()

    return expert_usage

def aggregate_task_usage(task_name, prompts, tokenizer, model, top_k, num_hidden_layers, num_experts_per_layer):
    """
    Aggregates expert usage across all prompts for a specific task.
    """
    print(f"\nAnalyzing Task: {task_name} ({len(prompts)} prompts)...")
    # Initialize structure
    task_usage = {}
    for layer_idx in range(num_hidden_layers):
        # Use int64 to prevent overflow during summation
        task_usage[layer_idx] = np.zeros(num_experts_per_layer, dtype=np.int64)
    
    total_tokens_processed = 0
        
    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}...")
        prompt_usage = analyze_prompt_routing(prompt, tokenizer, model, top_k, num_experts_per_layer)
        
        # Calculate tokens processed for this prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        total_tokens_processed += inputs.input_ids.shape[1]
        
        # Aggregate results
        for layer_idx, counts in prompt_usage.items():
             if layer_idx in task_usage:
                task_usage[layer_idx] += counts
            
    return task_usage, total_tokens_processed

# --- Execution ---
print("=" * 70)
print("TASK-BASED ROUTING ANALYSIS STARTING")
print("=" * 70)

all_tasks_usage = {}
total_tokens_per_task = {}

for task_name, prompts in TASK_PROMPTS.items():
    usage, tokens = aggregate_task_usage(task_name, prompts, tokenizer, model, top_k, num_hidden_layers, num_experts_per_layer)
    all_tasks_usage[task_name] = usage
    total_tokens_per_task[task_name] = tokens

print("\nAnalysis complete. Processing results...")

# --- Results Analysis ---

# 1. Convert aggregated data into a structured Numpy array
# Array shape: [num_tasks, num_layers, num_experts]
task_names = list(all_tasks_usage.keys())
num_tasks = len(task_names)
activation_matrix = np.zeros((num_tasks, num_hidden_layers, num_experts_per_layer), dtype=np.int64)

for task_idx, task_name in enumerate(task_names):
    task_usage = all_tasks_usage[task_name]
    for layer_idx in range(num_hidden_layers):
        if layer_idx in task_usage:
            activation_matrix[task_idx, layer_idx, :] = task_usage[layer_idx]

# 2. Analyze utilization per task (Your primary goal for pruning)
print("\n" + "=" * 70)
print("PER-TASK UTILIZATION ANALYSIS (Pruning Opportunities)")
print("=" * 70)

# This analysis checks if an expert is completely unused across ALL layers for a specific task.
for task_idx, task_name in enumerate(task_names):
    print(f"\nTask: {task_name}")
    task_activations = activation_matrix[task_idx] # Shape: [num_layers, num_experts]
    
    # Sum activations over the layer axis (axis=0)
    total_expert_usage = task_activations.sum(axis=0) # Shape: [num_experts]
    
    unused_experts = np.where(total_expert_usage == 0)[0]
    used_experts_count = num_experts_per_layer - len(unused_experts)
    
    print(f"  Total experts used: {used_experts_count}/{num_experts_per_layer} ({(used_experts_count/num_experts_per_layer)*100:.2f}%)")
    
    # Key finding for pruning:
    if len(unused_experts) > 0:
        print(f"  Potential for pruning: {len(unused_experts)} experts are completely unused for this task.")
        print(f"  Unused expert IDs: {unused_experts.tolist()}")
    else:
        print("  All experts were used at least once for this task.")
        
# 3. Compare utilization across tasks
print("\n" + "=" * 70)
print("CROSS-TASK COMPARISON (Specialization vs Core)")
print("=" * 70)

# Create a usage mask (1 if used by the task in ANY layer, 0 otherwise)
# Summing over layers (axis=1)
task_usage_mask = (activation_matrix.sum(axis=1) > 0) # Shape: [num_tasks, num_experts]

# Core experts (used by ALL tasks)
core_experts_mask = task_usage_mask.all(axis=0)
core_experts = np.where(core_experts_mask)[0]
print(f"Core experts (used by all {num_tasks} tasks): {len(core_experts)}/{num_experts_per_layer}")

# Specialized experts (used by ONLY ONE task)
# Summing over tasks (axis=0). If sum == 1, only one task used it.
specialized_experts_mask = (task_usage_mask.sum(axis=0) == 1)
specialized_experts = np.where(specialized_experts_mask)[0]
print(f"Specialized experts (used by only one task): {len(specialized_experts)}/{num_experts_per_layer}")

# Details on specialized experts
if len(specialized_experts) > 0:
    print("\nSpecialization details:")
    for task_idx, task_name in enumerate(task_names):
        # Experts used by this task AND marked as specialized
        task_specialists = np.where(task_usage_mask[task_idx] & specialized_experts_mask)[0]
        if len(task_specialists) > 0:
            print(f"  {task_name} exclusive experts: {len(task_specialists)}")
            # print(f"    IDs: {task_specialists.tolist()}")

# 4. Task Correlation Analysis
print("\n" + "=" * 70)
print("TASK CORRELATION ANALYSIS")
print("=" * 70)
# How similar are the activation patterns between tasks?

# Flatten the utilization patterns (all layers, all experts) for each task
correlation_data = {}
for task_idx, task_name in enumerate(task_names):
    # Flatten the [num_layers, num_experts] matrix into a 1D vector
    correlation_data[task_name] = activation_matrix[task_idx].flatten()

# Calculate the Pearson correlation matrix
correlation_df = pd.DataFrame(correlation_data).corr()
print("Correlation Matrix (Higher value = more similar expert usage):")
print(correlation_df.to_markdown(floatfmt=".4f"))

# --- Visualization ---
print("\nGenerating visualizations...")
# Create directory for visualizations
VIS_DIR = "expert_analysis_visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# Visualization 1: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
plt.title("Correlation of Expert Utilization Patterns Between Tasks")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/task_correlation.png")
print(f"Saved visualization: {VIS_DIR}/task_correlation.png")
plt.close()

# Visualization 2: Overall Expert Usage Binary Map
plt.figure(figsize=(20, 5))
sns.heatmap(task_usage_mask.astype(int), cmap="viridis", cbar=False, linewidths=.5)
plt.title("Expert Usage Binary Map Across Tasks (Yellow = Used, Purple = Unused)", fontsize=16)
plt.xlabel(f"Expert ID (0-{num_experts_per_layer-1})", fontsize=14)
plt.ylabel("Task", fontsize=14)
plt.yticks(np.arange(num_tasks) + 0.5, task_names, rotation=0)
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/expert_usage_binary_map.png")
print(f"Saved visualization: {VIS_DIR}/expert_usage_binary_map.png")
plt.close()

# Visualization 3: Normalized Activation Frequency (Middle Layer)
LAYER_TO_VISUALIZE = num_hidden_layers // 2
plt.figure(figsize=(20, 5))

layer_activations = activation_matrix[:, LAYER_TO_VISUALIZE, :].astype(float)
normalized_activations = np.zeros_like(layer_activations)

for task_idx, task_name in enumerate(task_names):
    total_tokens = total_tokens_per_task.get(task_name, 0)
    if total_tokens > 0:
        # Calculate frequency: activations / total tokens processed
        normalized_activations[task_idx, :] = layer_activations[task_idx, :] / total_tokens

sns.heatmap(normalized_activations, cmap="plasma", cbar_kws={'label': 'Normalized Activation Frequency'})
plt.title(f"Expert Activation Frequency Heatmap (Layer {LAYER_TO_VISUALIZE})", fontsize=16)
plt.xlabel(f"Expert ID (0-{num_experts_per_layer-1})", fontsize=14)
plt.ylabel("Task", fontsize=14)
plt.yticks(np.arange(num_tasks) + 0.5, task_names, rotation=0)
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/expert_activation_frequency_middle_layer.png")
print(f"Saved visualization: {VIS_DIR}/expert_activation_frequency_middle_layer.png")
plt.close()


# Visualization 4: Layer-wise analysis for each task
for task_idx, task_name in enumerate(task_names):
    # Data for the specific task: Shape [num_layers, num_experts]
    layerwise_data = activation_matrix[task_idx]

    plt.figure(figsize=(20, 15))
    
    # Use a logarithmic scale for color if the activation counts vary widely
    # Add a small epsilon to handle log(0)
    vmin = max(1, layerwise_data.min())
    vmax = layerwise_data.max() + 1e-6
    
    if vmax > vmin:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        
        sns.heatmap(layerwise_data + 1e-6, annot=False, cmap="magma", linewidths=.5, norm=norm, cbar_kws={'label': 'Activations (Log Scale)'})
        plt.title(f"Expert Activations per Layer for Task: {task_name}", fontsize=16)
        plt.ylabel("Layer ID", fontsize=14)
        plt.xlabel(f"Expert ID (0-{num_experts_per_layer-1})", fontsize=14)
        plt.tight_layout()
        
        filename = f"{VIS_DIR}/expert_usage_layerwise_{task_name}.png"
        plt.savefig(filename)
        print(f"Saved visualization: {filename}")
        plt.close()
    else:
        print(f"Skipping visualization for {task_name} due to lack of variation in data.")

print(f"\nANALYSIS COMPLETE. Visualizations saved in '{VIS_DIR}' directory.")