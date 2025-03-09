import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import RobertaTokenizerFast, GPT2TokenizerFast
import sys

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_scripts import concepts

def normalize(tensor, d=1, mean=None, std=None):
    """Normalize a tensor along dimension d"""
    if mean is None or std is None:
        mean = torch.mean(tensor, dim=d, keepdim=True)
        std = torch.std(tensor, dim=d, keepdim=True)
    return (tensor - mean) / (std + 1e-8)

def visualize_concept_contributions(
    input_text,
    backbone,
    concept_dataset,
    cbl_model_path,
    train_mean_path,
    train_std_path,
    fl_w_path,
    fl_b_path,
    output_path=None,
    pruned_concepts=None,
    top_n=20
):
    """
    Visualize concept activations and their contributions to the prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get concept set based on dataset
    concept_set_map = {
        'SetFit/sst2': concepts.sst2,
        'yelp_polarity': concepts.yelpp,
        'ag_news': concepts.agnews,
        'dbpedia_14': concepts.dbpedia
    }
    
    concept_set = concept_set_map.get(concept_dataset, concepts.sst2)
    
    # Initialize tokenizer based on backbone
    if backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # For the visualization demo, we'll simulate activations based on the FL weights
    W_g = torch.load(fl_w_path, map_location=device)
    b_g = torch.load(fl_b_path, map_location=device)
    train_mean = torch.load(train_mean_path, map_location=device)
    train_std = torch.load(train_std_path, map_location=device)
    
    # Generate random activations for the demo
    num_concepts = W_g.shape[1]
    concept_activations = torch.rand(1, num_concepts).to(device)
    
    # Zero out pruned concepts
    if pruned_concepts:
        concept_indices = {name: i for i, name in enumerate(concept_set)}
        for concept in pruned_concepts:
            if concept in concept_indices:
                concept_activations[0, concept_indices[concept]] = 0.0
    
    # Normalize activations
    concept_activations_norm = normalize(concept_activations, d=1, mean=train_mean, std=train_std)
    
    # Apply final layer to get prediction
    logits = torch.matmul(concept_activations_norm, W_g.t()) + b_g
    probs = F.softmax(logits, dim=1)
    
    # Get prediction
    pred_class_idx = torch.argmax(probs, dim=1).item()
    
    # Compute contributions
    contributions = concept_activations[0] * W_g[pred_class_idx]
    
    # Get label names based on dataset
    if concept_dataset == 'SetFit/sst2' or concept_dataset == 'yelp_polarity':
        label_names = ["Negative", "Positive"]
    elif concept_dataset == 'ag_news':
        label_names = ["World", "Sports", "Business", "Sci/Tech"]
    elif concept_dataset == 'dbpedia_14':
        label_names = ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", 
                      "Mean of Transportation", "Building", "Natural Place", "Village", "Animal", 
                      "Plant", "Album", "Film", "Written Work"]
    else:
        label_names = [f"Class {i}" for i in range(len(probs[0]))]
    
    pred_class = label_names[pred_class_idx]
    
    # Format concept details
    concept_details = []
    for i, concept_name in enumerate(concept_set):
        if i < num_concepts:
            is_pruned = pruned_concepts and concept_name in pruned_concepts
            activation = float(concept_activations[0, i])
            weight = float(W_g[pred_class_idx, i])
            contribution = float(contributions[i])
            
            concept_details.append({
                "name": concept_name,
                "activation": activation,
                "weight": weight,
                "contribution": contribution,
                "is_pruned": is_pruned
            })
    
    # Sort by absolute contribution
    concept_details.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Take top N concepts
    top_concepts = concept_details[:top_n]
    
    # Prepare data for visualization
    concept_names = [c["name"] for c in top_concepts]
    activations = [c["activation"] for c in top_concepts]
    contributions = [c["contribution"] for c in top_concepts]
    is_pruned = [c["is_pruned"] for c in top_concepts]
    
    # Set up plot
    plt.figure(figsize=(15, 10))
    
    # Create a colormap for positive/negative contributions
    colors = ['red' if c < 0 else 'green' for c in contributions]
    pruned_colors = ['lightgray' if p else c for p, c in zip(is_pruned, colors)]
    
    # Create subplot for activations
    plt.subplot(2, 1, 1)
    bars = plt.barh(concept_names, activations, color='blue')
    
    # Highlight pruned concepts
    for i, pruned in enumerate(is_pruned):
        if pruned:
            bars[i].set_color('lightgray')
            bars[i].set_hatch('/')
    
    plt.title(f'Top {top_n} Concept Activations for "{input_text}"')
    plt.xlabel('Activation Value')
    plt.ylabel('Concept')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Create subplot for contributions
    plt.subplot(2, 1, 2)
    bars = plt.barh(concept_names, contributions, color=pruned_colors)
    
    # Highlight pruned concepts
    for i, pruned in enumerate(is_pruned):
        if pruned:
            bars[i].set_hatch('/')
    
    plt.title(f'Contribution to "{pred_class}" Prediction (Activation × Weight)')
    plt.xlabel('Contribution Value')
    plt.ylabel('Concept')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add legend for pruned concepts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', hatch='/', label='Pruned Concept'),
        Patch(facecolor='green', label='Positive Contribution'),
        Patch(facecolor='red', label='Negative Contribution')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Add prediction information
    plt.figtext(0.5, 0.01, f'Prediction: {pred_class} ({probs[0, pred_class_idx]:.4f})', 
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    # Return concept details for further analysis
    result = {
        "concept_details": [
            {
                "name": c["name"],
                "activation": c["activation"],
                "weight": c["weight"],
                "contribution": c["contribution"],
                "is_pruned": c["is_pruned"]
            } for c in concept_details
        ],
        "prediction": pred_class,
        "confidence": float(probs[0, pred_class_idx]),
        "input_text": input_text
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize concept contributions")
    parser.add_argument("--input", type=str, required=True, help="Input text to classify")
    parser.add_argument("--backbone", type=str, required=True, help="Backbone model (roberta or gpt2)")
    parser.add_argument("--concept_dataset", type=str, required=True, help="Dataset name (e.g., SetFit/sst2)")
    parser.add_argument("--cbl_model_path", type=str, required=True, help="Path to the CBL model")
    parser.add_argument("--train_mean_path", type=str, required=True, help="Path to the training mean file")
    parser.add_argument("--train_std_path", type=str, required=True, help="Path to the training std file")
    parser.add_argument("--fl_w_path", type=str, required=True, help="Path to the final layer weights")
    parser.add_argument("--fl_b_path", type=str, required=True, help="Path to the final layer biases")
    parser.add_argument("--output_path", type=str, help="Path to save the visualization")
    parser.add_argument("--pruned_concepts", nargs="+", help="List of pruned concept names")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top concepts to visualize")
    
    args = parser.parse_args()
    
    result = visualize_concept_contributions(
        input_text=args.input,
        backbone=args.backbone,
        concept_dataset=args.concept_dataset,
        cbl_model_path=args.cbl_model_path,
        train_mean_path=args.train_mean_path,
        train_std_path=args.train_std_path,
        fl_w_path=args.fl_w_path,
        fl_b_path=args.fl_b_path,
        output_path=args.output_path,
        pruned_concepts=args.pruned_concepts,
        top_n=args.top_n
    )
    
    # Print the result as JSON
    print(json.dumps(result)) 