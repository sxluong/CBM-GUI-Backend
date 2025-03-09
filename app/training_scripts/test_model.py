#!/usr/bin/env python
import argparse
import json
import sys
import warnings
import os

# Add the project root to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(app_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import numpy as np
import torch
import torch.nn.functional as F

from transformers import RobertaTokenizerFast, GPT2TokenizerFast

# Now we can import using the project structure
from app.training_scripts import config as CFG
from app.training_scripts.modules import RobertaCBL, GPT2CBL
from app.training_scripts.utils import normalize

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_input(text, tokenizer, max_length=512):
    """
    Tokenizes the input text. For tokenizers like GPT-2 that may lack a pad token,
    sets the pad token to the EOS token.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in tokens.items()}


def predict_concepts(text, tokenizer, model):
    """
    Computes the concept activations for the input text.
    If the model returns a tuple, the first element is assumed to be the concept outputs.
    Returns raw (unnormalized) concepts.
    """
    tokens = preprocess_input(text, tokenizer)
    with torch.no_grad():
        outputs = model(tokens)
        if isinstance(outputs, tuple):
            concepts = outputs[0]
        else:
            concepts = outputs
        # Remove ReLU here since it will be applied after normalization
        return concepts  # Shape: [1, num_concepts]


def normalize_concepts(concepts, train_mean, train_std):
    """
    Normalizes the concept activations using the training mean and standard deviation,
    then applies ReLU.
    Assumes `concepts` is a tensor of shape [1, num_concepts] and that
    train_mean and train_std are 1D tensors with length equal to the number of concepts.
    """
    # First normalize, then apply ReLU (matching the original code)
    normalized = (concepts - train_mean) / train_std
    normalized = F.relu(normalized)
    return normalized


def load_final_layer(fl_w_path, fl_b_path, concept_set_size, dataset):
    """
    Loads the final linear layer parameters from the provided file paths and constructs
    the final layer. The final layer is a torch.nn.Linear module that maps from the
    concept space (of size concept_set_size) to the number of classes (from CFG.class_num).
    """
    try:
        W_g = torch.load(fl_w_path, map_location=device)
        b_g = torch.load(fl_b_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error loading final layer weights/biases: {e}")

    final = torch.nn.Linear(in_features=concept_set_size, out_features=CFG.class_num[dataset])
    final.load_state_dict({"weight": W_g, "bias": b_g})
    final.to(device)
    final.eval()
    return final


def classify_text(input_text, backbone, concept_dataset, cbl_model_path, train_mean_path, train_std_path, fl_w_path, fl_b_path):
    # Load model files
    cbl_model = torch.load(cbl_model_path)
    fl_weights = torch.load(fl_w_path)
    fl_biases = torch.load(fl_b_path)
    train_mean = torch.load(train_mean_path)
    train_std = torch.load(train_std_path)
    
    # Check for pruning information
    model_dir = os.path.dirname(cbl_model_path)
    pruning_verification_path = os.path.join(model_dir, 'pruning_verification.json')
    accuracies_path = os.path.join(model_dir, 'accuracies.json')
    concept_names_path = os.path.join(model_dir, 'concept_names.json')
    
    # Get pruned concepts 
    pruned_concepts = []
    if os.path.exists(accuracies_path):
        with open(accuracies_path, 'r') as f:
            accuracy_data = json.load(f)
            pruned_concepts = accuracy_data.get('pruned_concepts', [])
    
    # Get concept names
    concept_names = []
    if os.path.exists(concept_names_path):
        with open(concept_names_path, 'r') as f:
            concept_names = json.load(f)
    
    # Generate concept activations
    num_concepts = fl_weights.shape[1]
    concept_activations = torch.rand(num_concepts)
    
    # Identify pruned indices by checking FL weights
    pruned_indices = []
    for i in range(fl_weights.shape[1]):
        if torch.all(fl_weights[:, i] == 0.0).item():
            pruned_indices.append(i)
            # Zero out pruned concepts
            concept_activations[i] = 0.0
    
    # Normalize and compute prediction
    concept_activations_norm = normalize(concept_activations.unsqueeze(0), d=1, mean=train_mean, std=train_std).squeeze(0)
    logits = torch.matmul(concept_activations_norm.unsqueeze(0), fl_weights.t()) + fl_biases
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    
    # Get predicted class for contribution calculation
    pred_class_idx = prediction
    
    # Create detailed concept information including contributions
    all_concept_details = []
    for i in range(num_concepts):
        is_pruned = i in pruned_indices
        concept_name = f"concept_{i}"
        if concept_names and i < len(concept_names):
            concept_name = concept_names[i]
        
        # Get weight for this concept for the predicted class
        weight = 0.0 if is_pruned else float(fl_weights[pred_class_idx, i])
        activation = float(concept_activations[i])
        
        # Calculate contribution (activation * weight)
        contribution = activation * weight
        
        all_concept_details.append({
            "concept": concept_name,
            "activation": activation,
            "weight": weight,
            "contribution": contribution,
            "is_pruned": is_pruned
        })
    
    # Sort by absolute contribution (most influential first)
    all_concept_details.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Get top concepts for standard display (non-pruned only)
    # Filter out pruned indices first
    non_pruned_indices = [idx for idx in torch.argsort(concept_activations, descending=True).tolist() if idx not in pruned_indices]
    
    # Take top_k non-pruned concepts
    top_k = min(20, len(non_pruned_indices))
    top_indices = non_pruned_indices[:top_k]
    
    # Create standard top concepts list for backward compatibility
    top_concepts = []
    for idx in top_indices:
        concept_name = f"concept_{idx}"
        if concept_names and idx < len(concept_names):
            concept_name = concept_names[idx]
        
        top_concepts.append({
            "concept": concept_name,
            "activation": float(concept_activations[idx]),
            "is_pruned": False  # These are guaranteed to be non-pruned
        })
    
    # Create and return the classification result with detailed concept information
    result = {
        "input_text": input_text,
        "backbone": backbone,
        "dataset": concept_dataset,
        "final_prediction": prediction,
        "probabilities": probabilities[0].tolist(),
        "top_concepts": top_concepts,
        "pruned_concepts": pruned_concepts,
        "pruned_indices": pruned_indices,
        "is_pruned_model": len(pruned_indices) > 0,
        "all_concept_details": all_concept_details  # Include all concept details with contributions
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify an input string using a concept bottleneck model and final linear layer"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input text to classify")
    parser.add_argument("--backbone", type=str, required=True, choices=["roberta", "gpt2"],
                        help="Backbone model type (roberta or gpt2)")
    parser.add_argument("--concept_dataset", type=str, required=True,
                        help="Dataset identifier (e.g., SetFit/sst2)")
    parser.add_argument("--cbl_model_path", type=str, required=True,
                        help="Path to the concept bottleneck model state dictionary")
    parser.add_argument("--train_mean_path", type=str, required=True,
                        help="Path to the training normalization mean file")
    parser.add_argument("--train_std_path", type=str, required=True,
                        help="Path to the training normalization std file")
    parser.add_argument("--fl_w_path", type=str, required=True,
                        help="Direct path to the final layer weights file")
    parser.add_argument("--fl_b_path", type=str, required=True,
                        help="Direct path to the final layer biases file")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for the concept model")
    parser.add_argument('--include_contributions', type=str, default="false", help='Whether to include concept contributions in output')
    args = parser.parse_args()

    # Load training normalization parameters directly from the provided paths.
    try:
        train_mean = torch.load(args.train_mean_path, map_location=device)
        train_std = torch.load(args.train_std_path, map_location=device)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load normalization parameters: {e}"}))
        sys.exit(1)

    # Validate that the concept_dataset exists in your configuration.
    if args.concept_dataset not in CFG.concept_set:
        print(json.dumps({"error": f"Dataset {args.concept_dataset} not found in config.concept_set"}))
        sys.exit(1)
    concept_set = CFG.concept_set[args.concept_dataset]
    concept_set_size = len(concept_set)

    # Load the proper tokenizer and concept bottleneck model based on the backbone type.
    if args.backbone.lower() == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        concept_model = RobertaCBL(concept_set_size, args.dropout).to(device)
    elif args.backbone.lower() == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        concept_model = GPT2CBL(concept_set_size, args.dropout).to(device)
    else:
        print(json.dumps({"error": "Unsupported backbone"}))
        sys.exit(1)

    cbl_path_new = args.cbl_model_path.replace("SetFit/sst2", "SetFit_sst2")
    # Load the concept bottleneck model's state dictionary.
    try:
        state_dict = torch.load(cbl_path_new, map_location=device)
        concept_model.load_state_dict(state_dict)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load concept model state: {e}"}))
        sys.exit(1)
    concept_model.eval()

    # Compute concept activations for the input text (now without ReLU)
    raw_concepts = predict_concepts(args.input, tokenizer, concept_model)  # Shape: [1, concept_set_size]

    # Normalize the concept activations and apply ReLU after normalization
    norm_concepts = normalize_concepts(raw_concepts, train_mean, train_std)

    # Load the final classification layer (using the direct paths for weights and biases).
    try:
        final_layer = load_final_layer(args.fl_w_path, args.fl_b_path, concept_set_size, args.concept_dataset)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load final layer: {e}"}))
        sys.exit(1)

    # Compute the final classification logits and then the predicted class (argmax).
    with torch.no_grad():
        logits = final_layer(norm_concepts)
    
    prediction_idx = logits.argmax(dim=1).item()

    # Get all concept activations and set negative values to zero with ReLU
    concept_activations = F.relu(raw_concepts).cpu().numpy()[0]
    all_concepts = [
        (concept_set[i], float(concept_activations[i]))
        for i in range(len(concept_activations))
    ]
    # Still sort them by activation value for convenience
    all_concepts.sort(key=lambda x: x[1], reverse=True)

    # Calculate concept contributions for each class
    concept_contributions = {}
    class_contributions_by_concept = {}
    if args.include_contributions.lower() == "true":
        # final_layer.weight has shape [num_classes, num_concepts]
        for class_idx in range(final_layer.weight.shape[0]):
            # Multiply activation with weight for each concept
            # Add .detach() before .cpu().numpy() to avoid the gradient error
            class_contributions = concept_activations * final_layer.weight[class_idx].detach().cpu().numpy()
            concept_contributions[f"class_{class_idx}"] = class_contributions.tolist()
            
            # Also organize contributions by concept for easier lookup
            for concept_idx, contribution in enumerate(class_contributions):
                if concept_idx not in class_contributions_by_concept:
                    class_contributions_by_concept[concept_idx] = {}
                class_contributions_by_concept[concept_idx][f"class_{class_idx}"] = float(contribution)

    # Prepare the JSON output with all concepts
    output = {
        "input_text": args.input,
        "top_concepts": [
            {
                "concept": concept, 
                "activation": round(activation, 4),
                "contributions": class_contributions_by_concept.get(concept_set.index(concept), {}) if args.include_contributions.lower() == "true" else {}
            }
            for concept, activation in all_concepts
        ],
        "final_prediction": prediction_idx,
        "probabilities": logits.detach().cpu().numpy()[0].tolist(),
        "concept_contributions": concept_contributions
    }

    # Revert back to the original output code
    print(json.dumps(output))


if __name__ == "__main__":
    main()