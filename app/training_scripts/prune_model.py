import os
import torch
import json
from datetime import datetime
import numpy as np

def prune_model(
    model_id,
    pruned_concepts,
    concept_dataset="SetFit/sst2",
    backbone="roberta",
    output_dir=None
):
    """
    Prune specific concepts from a trained CBL model.
    
    Args:
        model_id (str): The ID of the model to prune
        pruned_concepts (list): List of concept names to prune
        concept_dataset (str): The dataset used for training 
        backbone (str): The backbone model (roberta or gpt2)
        output_dir (str, optional): Directory to save pruned model
    
    Returns:
        dict: Information about the pruning process, including paths to saved files
    """
    
    # Format the dataset name for directory structure
    formatted_dataset = concept_dataset.replace('/', '_')
    
    # Prepare directory paths
    base_dir = "mpnet_acs"
    model_dir = f"{base_dir}/{formatted_dataset}/model_{model_id}/{backbone}_cbm"
    
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pruned_model_id = f"{model_id}_pruned_{len(pruned_concepts)}_{timestamp}"
        output_dir = f"{base_dir}/{formatted_dataset}/model_{pruned_model_id}/{backbone}_cbm"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from: {model_dir}")
    print(f"Saving pruned model to: {output_dir}")
    
    # Try different approaches to load concept sets
    try:
        # First attempt - direct relative import
        from ..training_scripts import concepts
        concept_set = {'SetFit/sst2': concepts.sst2, 
                      'yelp_polarity': concepts.yelpp, 
                      'ag_news': concepts.agnews, 
                      'dbpedia_14': concepts.dbpedia}[concept_dataset]
        print(f"Loaded {len(concept_set)} concepts from relative import")
    except (ImportError, KeyError) as e:
        print(f"First import attempt failed: {e}")
        try:
            # Second attempt - absolute import
            from app.training_scripts import concepts
            concept_set = {'SetFit/sst2': concepts.sst2, 
                          'yelp_polarity': concepts.yelpp, 
                          'ag_news': concepts.agnews, 
                          'dbpedia_14': concepts.dbpedia}[concept_dataset]
            print(f"Loaded {len(concept_set)} concepts from absolute import")
        except (ImportError, KeyError) as e:
            print(f"Second import attempt failed: {e}")
            
            # Third attempt - load concepts from JSON file if available
            concept_names_path = f"{model_dir}/concept_names.json"
            if os.path.exists(concept_names_path):
                with open(concept_names_path, 'r') as f:
                    concept_set = json.load(f)
                print(f"Loaded {len(concept_set)} concepts from JSON file")
            else:
                print("Warning: Could not load concepts. Using concept name matching instead.")
                concept_set = []
    
    # Get indices of concepts to prune
    pruned_indices = []
    
    # If we have a concept set, use it to find indices
    if concept_set:
        for concept in pruned_concepts:
            if concept in concept_set:
                idx = concept_set.index(concept)
                pruned_indices.append(idx)
                print(f"Will prune concept '{concept}' at index {idx}")
            else:
                print(f"Warning: Concept '{concept}' not found in concept set, skipping.")
    else:
        # If no concept set available, we'll try exact match on column names
        print("Using index-based matching - will save concepts and indices mapping")
    
    # Load the original model files
    cbl_path = f"{model_dir}/cbl_acc_best.pt"
    fl_weights_path = f"{model_dir}/W_g_acc_best.pt"
    fl_biases_path = f"{model_dir}/b_g_acc_best.pt"
    train_std_path = f"{model_dir}/train_std_acc_best.pt"
    train_mean_path = f"{model_dir}/train_mean_acc_best.pt"
    accuracy_path = f"{model_dir}/accuracies.json"
    
    # Check if all required files exist
    required_files = [cbl_path, fl_weights_path, fl_biases_path, train_std_path, train_mean_path, accuracy_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load model components
    cbl_state_dict = torch.load(cbl_path)
    fl_weights = torch.load(fl_weights_path)
    fl_biases = torch.load(fl_biases_path)
    train_std = torch.load(train_std_path)
    train_mean = torch.load(train_mean_path)
    
    with open(accuracy_path, 'r') as f:
        accuracy_data = json.load(f)
    
    original_accuracy = accuracy_data.get('full_accuracy', 0.0)
    
    # Save weight statistics before pruning for verification
    fl_weights_before = fl_weights.clone()
    
    # Create a pruning mask for the CBL output
    if "fc2.weight" in cbl_state_dict:
        # Prune the CBL output layer
        fc2_shape = cbl_state_dict["fc2.weight"].shape[0]
        pruning_mask = torch.ones(fc2_shape)
        print(f"FC2 weight shape: {fc2_shape}")
        
        for idx in pruned_indices:
            if idx < len(pruning_mask):
                pruning_mask[idx] = 0
                print(f"Applied mask to index {idx} in CBL output layer")
            else:
                print(f"Warning: Index {idx} out of bounds for mask (size {len(pruning_mask)})")
        
        # Apply pruning mask to weights and update the model
        for layer_name in ["fc2.weight", "fc2.bias"]:
            if layer_name in cbl_state_dict:
                if "weight" in layer_name:
                    # For weight matrices, apply mask to rows (output features)
                    mask_reshaped = pruning_mask.view(-1, 1)
                    print(f"Shape before masking {layer_name}: {cbl_state_dict[layer_name].shape}")
                    cbl_state_dict[layer_name] = cbl_state_dict[layer_name] * mask_reshaped
                    print(f"Shape after masking {layer_name}: {cbl_state_dict[layer_name].shape}")
                else:
                    # For bias vectors, apply mask directly
                    print(f"Shape before masking {layer_name}: {cbl_state_dict[layer_name].shape}")
                    cbl_state_dict[layer_name] = cbl_state_dict[layer_name] * pruning_mask
                    print(f"Shape after masking {layer_name}: {cbl_state_dict[layer_name].shape}")
    else:
        print("Warning: Could not find fc2.weight in CBL state dict. Model structure differs from expected.")
        # Print keys to help debugging
        print(f"Available keys in state dict: {cbl_state_dict.keys()}")
    
    # Prune the final layer weights for pruned concepts
    # FL weights shape is typically [num_classes, num_concepts]
    print(f"FL weights shape: {fl_weights.shape}")
    successfully_pruned = []
    
    for idx in pruned_indices:
        if idx < fl_weights.shape[1]:  # Check if index is valid
            # Save weights before zeroing for verification
            col_before = fl_weights[:, idx].clone()
            
            # Zero out the column corresponding to the pruned concept
            fl_weights[:, idx] = 0.0
            
            # Verify zeroing was successful
            col_after = fl_weights[:, idx]
            pruned_success = torch.all(col_after == 0.0).item()
            
            if pruned_success:
                successfully_pruned.append(idx)
                print(f"Successfully pruned concept at index {idx}")
                print(f"  Before: {col_before[:5]}...")  # Print first few values
                print(f"  After: {col_after[:5]}...")    # Should all be zeros
            else:
                print(f"WARNING: Failed to prune concept at index {idx}")
                print(f"  Before: {col_before[:5]}")
                print(f"  After: {col_after[:5]}")
        else:
            print(f"Warning: Index {idx} out of bounds for FL weights (shape {fl_weights.shape})")
    
    # Create a mapping of pruned concepts to their indices
    concept_indices_map = {}
    for i, concept in enumerate(pruned_concepts):
        if i < len(pruned_indices):
            concept_indices_map[concept] = pruned_indices[i]
    
    # Save the pruned model
    torch.save(cbl_state_dict, f"{output_dir}/cbl_acc_best.pt")
    torch.save(fl_weights, f"{output_dir}/W_g_acc_best.pt")
    torch.save(fl_biases, f"{output_dir}/b_g_acc_best.pt")
    torch.save(train_std, f"{output_dir}/train_std_acc_best.pt")
    torch.save(train_mean, f"{output_dir}/train_mean_acc_best.pt")
    
    # Create a copy of the accuracy data with pruning info
    pruned_accuracy_data = accuracy_data.copy()
    pruned_accuracy_data["pruned_concepts"] = pruned_concepts
    pruned_accuracy_data["pruned_indices"] = pruned_indices
    pruned_accuracy_data["successfully_pruned_indices"] = successfully_pruned
    
    with open(f"{output_dir}/accuracies.json", "w") as f:
        json.dump(pruned_accuracy_data, f, indent=2)
    
    # Save pruning verification data for debugging
    verification_data = {
        "pruned_concepts": pruned_concepts,
        "pruned_indices": pruned_indices,
        "successfully_pruned": successfully_pruned,
        "concept_indices_map": concept_indices_map,
        "fl_weights_shape": list(fl_weights.shape),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{output_dir}/pruning_verification.json", "w") as f:
        json.dump(verification_data, f, indent=2)
    
    # Save the concept list if we have it
    if concept_set:
        with open(f"{output_dir}/concept_names.json", "w") as f:
            json.dump(concept_set, f)
    
    # Return information about the pruning process
    return {
        "original_model_id": model_id,
        "pruned_model_id": pruned_model_id,
        "pruned_concepts": pruned_concepts,
        "pruned_concept_indices": pruned_indices,
        "successfully_pruned_indices": successfully_pruned,
        "original_accuracy": original_accuracy,
        "model_path": f"{output_dir}/cbl_acc_best.pt",
        "fl_weights_path": f"{output_dir}/W_g_acc_best.pt",
        "fl_biases_path": f"{output_dir}/b_g_acc_best.pt",
        "fl_std_path": f"{output_dir}/train_std_acc_best.pt",
        "fl_mean_path": f"{output_dir}/train_mean_acc_best.pt",
    } 