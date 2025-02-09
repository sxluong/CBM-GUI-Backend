#!/usr/bin/env python
import argparse
import json
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from transformers import RobertaTokenizerFast, GPT2TokenizerFast

# Import your configuration file. Ensure that config.concept_set is a dictionary
# mapping dataset identifiers (e.g., "SetFit/sst2") to lists of concept names,
# and that config.class_num maps dataset identifiers to the number of output classes.
import config as CFG

# Import your custom concept model classes.
from modules import RobertaCBL, GPT2CBL

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
    Activations are passed through ReLU.
    """
    tokens = preprocess_input(text, tokenizer)
    with torch.no_grad():
        outputs = model(tokens)
        if isinstance(outputs, tuple):
            concepts = outputs[0]
        else:
            concepts = outputs
        # Apply ReLU to obtain non-negative activations
        raw_concepts = torch.relu(concepts)
    return raw_concepts  # Shape: [1, num_concepts]


def normalize_concepts(concepts, train_mean, train_std):
    """
    Normalizes the concept activations using the training mean and standard deviation,
    then applies ReLU.
    Assumes `concepts` is a tensor of shape [1, num_concepts] and that
    train_mean and train_std are 1D tensors with length equal to the number of concepts.
    """
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

    # Load the concept bottleneck model's state dictionary.
    try:
        state_dict = torch.load(args.cbl_model_path, map_location=device)
        concept_model.load_state_dict(state_dict)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load concept model state: {e}"}))
        sys.exit(1)
    concept_model.eval()

    # Compute concept activations for the input text.
    raw_concepts = predict_concepts(args.input, tokenizer, concept_model)  # Shape: [1, concept_set_size]

    # Normalize the concept activations.
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
    prediction_idx = int(torch.argmax(logits, dim=1).item())

    # Prepare the list of top activated concepts.
    concept_activations = raw_concepts.cpu().numpy()[0]
    active_concepts = [
        (concept_set[i], float(concept_activations[i]))
        for i in range(len(concept_activations))
        if concept_activations[i] > 0
    ]
    active_concepts.sort(key=lambda x: x[1], reverse=True)
    top_n = 10
    top_concepts = active_concepts[:top_n]

    # Prepare the JSON output.
    output = {
        "input_text": args.input,
        "top_concepts": [
            {"concept": concept, "activation": round(activation, 4)}
            for concept, activation in top_concepts
        ],
        "final_prediction": prediction_idx
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()