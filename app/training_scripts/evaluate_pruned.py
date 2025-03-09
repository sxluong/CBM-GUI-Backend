#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
import sys

def evaluate_model(dataset, cbl_model_path, fl_weights_path, fl_biases_path, output_path):
    """
    Properly evaluate a pruned model on a test dataset.
    """
    print(f"Evaluating model with dataset: {dataset}")
    print(f"CBL model path: {cbl_model_path}")
    print(f"FL weights path: {fl_weights_path}")
    
    # Check if files exist
    for path in [cbl_model_path, fl_weights_path, fl_biases_path]:
        if not os.path.exists(path):
            print(f"ERROR: File does not exist: {path}")
            sys.exit(1)
    
    # Load the CBL model
    cbl_model_data = torch.load(cbl_model_path, map_location=torch.device('cpu'))
    backbone_name = cbl_model_data.get('backbone_name', 'roberta-base')
    concepts = cbl_model_data.get('concepts', [])
    print(f"Loaded {len(concepts)} concepts from model")
    
    # Load the model state dict for the backbone
    backbone_state_dict = cbl_model_data.get('model_state_dict')
    if not backbone_state_dict:
        print("WARNING: No model_state_dict found in CBL model data")
    
    # Load the FL weights and biases
    fl_weights = torch.load(fl_weights_path, map_location=torch.device('cpu'))
    fl_biases = torch.load(fl_biases_path, map_location=torch.device('cpu'))
    
    # Print debugging info about the weights
    print(f"FL weights shape: {fl_weights.shape}")
    print(f"FL biases shape: {fl_biases.shape}")
    print(f"Number of concepts: {len(concepts)}")
    
    # Check which columns (concepts) have zero weights
    zero_columns = []
    for i in range(fl_weights.shape[1]):
        if torch.all(fl_weights[:, i] == 0):
            zero_columns.append(i)
    print(f"Found {len(zero_columns)} columns with all-zero weights")
    for idx in zero_columns:
        if idx < len(concepts):
            print(f"  Pruned concept: {concepts[idx]} (index {idx})")
    
    # Load the tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        model = AutoModel.from_pretrained(backbone_name)
        
        # If we have model_state_dict, load it into the model
        if backbone_state_dict:
            model.load_state_dict(backbone_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load the dataset
    try:
        if dataset == 'SetFit/sst2':
            test_dataset = load_dataset("sst2", split="validation")
        elif dataset == 'yelp_polarity':
            test_dataset = load_dataset("yelp_polarity", split="test")
        elif dataset == 'ag_news':
            test_dataset = load_dataset("ag_news", split="test")
        elif dataset == 'dbpedia_14':
            test_dataset = load_dataset("dbpedia_14", split="test")
        else:
            print(f"Unknown dataset: {dataset}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Limit test set size for faster evaluation
    test_dataset = test_dataset.select(range(min(100, len(test_dataset))))
    print(f"Evaluating on {len(test_dataset)} examples")
    
    # Dynamically check the structure of the CBL model data
    print("\nExploring the CBL model data structure:")
    for key in cbl_model_data.keys():
        if key != 'model_state_dict':  # Skip the large state dict
            print(f"Key: {key}, Type: {type(cbl_model_data[key])}")

    # Check if the model uses a different structure for concept classifiers
    if 'concept_classifiers' in cbl_model_data:
        print("Found concept_classifiers in model data")
        concept_classifiers = cbl_model_data['concept_classifiers']
        print(f"Type of concept_classifiers: {type(concept_classifiers)}")
        print(f"Number of concept classifiers: {len(concept_classifiers) if isinstance(concept_classifiers, list) else 'not a list'}")
    
    # Try different approaches to find concept classifiers
    concept_classifiers = []

    # Approach 1: Check if concept_classifiers exists directly
    if 'concept_classifiers' in cbl_model_data and isinstance(cbl_model_data['concept_classifiers'], list):
        print("Using concept_classifiers from model data")
        concept_classifiers = cbl_model_data['concept_classifiers']
    # Approach 2: Check for projection weight/bias which is often used for concept projection
    elif 'projection.weight' in cbl_model_data and 'projection.bias' in cbl_model_data:
        print("Using projection weights from model")
        weights = cbl_model_data['projection.weight']
        bias = cbl_model_data['projection.bias']
        num_concepts = weights.shape[0]
        
        # Create concept classifiers using projection weights
        for i in range(num_concepts):
            classifier = (weights[i:i+1], bias[i:i+1])
            concept_classifiers.append(classifier)
        
        print(f"Created {len(concept_classifiers)} classifiers from projection weights")
    # Approach 3: Look for individual concept weights
    else:
        print("Looking for individual concept classifiers")
        for i in range(len(concepts)):
            # Try different naming patterns
            patterns = [
                (f'concept_{i}_weights', f'concept_{i}_bias'),
                (f'concept_classifier_{i}.weight', f'concept_classifier_{i}.bias'),
                (f'c{i}_weights', f'c{i}_bias')
            ]
            
            found = False
            for weight_key, bias_key in patterns:
                if weight_key in cbl_model_data and bias_key in cbl_model_data:
                    weights = cbl_model_data[weight_key]
                    bias = cbl_model_data[bias_key]
                    classifier = (weights, bias)
                    concept_classifiers.append(classifier)
                    found = True
                    break
            
            if not found:
                print(f"WARNING: No classifier found for concept {i} ({concepts[i] if i < len(concepts) else 'unknown'})")
                # Create dummy weights and bias with proper shape
                dummy_weights = torch.zeros((1, 768), device=torch.device('cpu'))  # 768 is RoBERTa hidden size
                dummy_bias = torch.zeros(1, device=torch.device('cpu'))
                concept_classifiers.append((dummy_weights, dummy_bias))
    
    # Prepare for evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            try:
                # Print some debugging info about the first few examples
                if i < 3:
                    print(f"\nProcessing example {i}: {example.keys()}")
                    
                # Prepare input
                if 'text' in example:
                    text = example['text']
                elif 'sentence' in example:
                    text = example['sentence']
                else:
                    print(f"Example {i} has no text or sentence field. Available keys: {example.keys()}")
                    continue
                    
                if 'label' not in example:
                    print(f"Example {i} has no label field. Available keys: {example.keys()}")
                    continue
                    
                label = example['label']
                
                # Print the actual input text for the first few examples
                if i < 3:
                    print(f"Text: {text[:100]}...")
                    print(f"Label: {label}")
                
                # Tokenize and get embeddings
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
                outputs = model(**inputs)
                
                # Get the [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Calculate concept activations using the concept classifiers
                concept_activations = torch.zeros(len(concept_classifiers), device=device)
                for i, classifier in enumerate(concept_classifiers):
                    weights, bias = classifier
                    weights = weights.to(device)
                    bias = bias.to(device)
                    # Get activation
                    activation = torch.sigmoid(F.linear(embeddings, weights, bias)).squeeze()
                    concept_activations[i] = activation
                
                # Apply FL weights and biases - this is where pruning has effect
                logits = F.linear(concept_activations.unsqueeze(0), fl_weights, fl_biases)
                pred = torch.argmax(logits, dim=1).item()
                
                # For debugging, print some predictions
                if total < 5:
                    print(f"Example {total}: Label={label}, Prediction={pred}")
                    print(f"Top 5 activations: {concept_activations[:5]}")
                    print(f"Logits: {logits}")
                
                if pred == label:
                    correct += 1
                total += 1
                
                # At the beginning of the evaluation loop, add:
                # Print the actual weight values for the first 3 examples to ensure pruning is having effect
                if total < 3:
                    print(f"\nEXAMPLE {total} DETAILS:")
                    # For concepts that should be pruned
                    for idx in zero_columns:
                        if idx < len(concepts):
                            print(f"Pruned concept {concepts[idx]} (idx {idx}): activation={concept_activations[idx].item()}")
                            # Show its contribution to each class
                            for class_idx in range(fl_weights.shape[0]):
                                # Contribution should be 0 since weights are 0
                                contribution = concept_activations[idx].item() * fl_weights[class_idx, idx].item()
                                print(f"  Contribution to class {class_idx}: {contribution} (weight={fl_weights[class_idx, idx].item()})")
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
    
    print(f"Processed {total} examples successfully")
    
    # Add a guard clause to handle the zero case
    if total == 0:
        print("WARNING: No examples were successfully processed!")
        # Save a default result
        results = {
            "accuracy": 0.0,
            "total_examples": 0,
            "correct_predictions": 0,
            "random_baseline": 0.5,  # Default for binary classification
            "pruned_concept_count": len(zero_columns),
            "error": "No examples could be processed"
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f)
        
        # Don't try to calculate accuracy, just return 0
        return 0.0
    
    # Only calculate accuracies if total > 0
    accuracy = correct / total
    print(f"Final accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Add a guard clause for random accuracy calculation too
    random_correct = 0
    num_classes = len(np.unique([ex['label'] for ex in test_dataset]))
    print(f"Number of classes: {num_classes}")
    
    for example in test_dataset:
        random_pred = np.random.randint(0, num_classes)
        if random_pred == example['label']:
            random_correct += 1
    
    random_accuracy = random_correct / len(test_dataset) if len(test_dataset) > 0 else 0.0
    print(f"Random baseline accuracy: {random_accuracy:.4f} ({random_correct}/{len(test_dataset)})")
    
    # Save the results
    results = {
        "accuracy": accuracy,
        "total_examples": total,
        "correct_predictions": correct,
        "random_baseline": random_accuracy,
        "pruned_concept_count": len(zero_columns)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    # Check for concept label files in the model directory
    base_dir = os.path.dirname(cbl_model_path)
    concept_labels_val_path = os.path.join(base_dir, "concept_labels_val.npy")
    concept_labels_train_path = os.path.join(base_dir, "concept_labels_train.npy")

    print(f"Looking for concept label files:")
    print(f"  - Val labels: {concept_labels_val_path} (exists: {os.path.exists(concept_labels_val_path)})")
    print(f"  - Train labels: {concept_labels_train_path} (exists: {os.path.exists(concept_labels_train_path)})")

    # Try to load concept labels if they exist
    concept_labels = None
    if os.path.exists(concept_labels_val_path):
        try:
            concept_labels = np.load(concept_labels_val_path)
            print(f"Loaded concept labels from validation set, shape: {concept_labels.shape}")
        except Exception as e:
            print(f"Error loading concept labels (val): {e}")
    elif os.path.exists(concept_labels_train_path):
        try:
            concept_labels = np.load(concept_labels_train_path)
            print(f"Loaded concept labels from training set, shape: {concept_labels.shape}")
        except Exception as e:
            print(f"Error loading concept labels (train): {e}")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned model on test set")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--cbl_model_path", type=str, required=True, help="Path to CBL model")
    parser.add_argument("--fl_weights_path", type=str, required=True, help="Path to FL weights")
    parser.add_argument("--fl_biases_path", type=str, required=True, help="Path to FL biases")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    
    args = parser.parse_args()
    evaluate_model(args.dataset, args.cbl_model_path, args.fl_weights_path, args.fl_biases_path, args.output_path) 