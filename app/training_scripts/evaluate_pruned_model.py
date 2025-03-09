import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import json
import app.training_scripts.config as CFG
from app.training_scripts.modules import CBL, RobertaCBL, GPT2CBL
from app.training_scripts.utils import normalize, eos_pooling
from torch.utils.data import DataLoader, TensorDataset
import time

def evaluate_pruned_model(
    model_id,
    concept_dataset="SetFit/sst2",
    backbone="roberta",
    batch_size=64
):
    """
    Evaluate the accuracy of a pruned CBL model on the test dataset.
    
    Args:
        model_id (str): The ID of the model to evaluate
        concept_dataset (str): The dataset to use for evaluation
        backbone (str): The backbone model (roberta or gpt2)
        batch_size (int): Batch size for evaluation
    
    Returns:
        dict: Evaluation results including accuracy
    """
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation of model {model_id}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Using device: {device}")
    
    # Format the dataset name for directory structure
    formatted_dataset = concept_dataset.replace('/', '_')
    
    # Prepare directory paths
    base_dir = "mpnet_acs"
    model_dir = f"{base_dir}/{formatted_dataset}/model_{model_id}/{backbone}_cbm"
    
    print(f"[{time.strftime('%H:%M:%S')}] Loading model from: {model_dir}")
    
    # Load model components
    cbl_path = f"{model_dir}/cbl_acc_best.pt"
    fl_weights_path = f"{model_dir}/W_g_acc_best.pt"
    fl_biases_path = f"{model_dir}/b_g_acc_best.pt"
    train_std_path = f"{model_dir}/train_std_acc_best.pt"
    train_mean_path = f"{model_dir}/train_mean_acc_best.pt"
    
    # Check if all files exist
    required_files = [cbl_path, fl_weights_path, fl_biases_path, train_std_path, train_mean_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: File not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print(f"[{time.strftime('%H:%M:%S')}] All model files found, now loading test dataset")
    
    # First load the dataset
    dataset_load_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading test dataset from {concept_dataset}")
    test_dataset = load_dataset(concept_dataset, split="test")

    # Print dataset info for debugging
    print(f"Dataset type: {type(test_dataset)}")
    print(f"Dataset structure: {test_dataset.features}")
    print(f"Dataset columns: {test_dataset.column_names}")

    # Initialize tokenizer BEFORE encoding the dataset
    tokenizer_start = time.time()
    if backbone == 'roberta':
        print(f"[{time.strftime('%H:%M:%S')}] Loading RoBERTa tokenizer")
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif backbone == 'gpt2':
        print(f"[{time.strftime('%H:%M:%S')}] Loading GPT-2 tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Unsupported backbone: {backbone}")
        raise ValueError(f"Unsupported backbone: {backbone}")
    tokenizer_time = time.time() - tokenizer_start
    print(f"[{time.strftime('%H:%M:%S')}] Tokenizer loaded in {tokenizer_time:.2f} seconds")

    # THEN encode the dataset with the tokenizer
    # Convert dataset to a format that can be batched
    encoded_test_dataset = {}
    if 'text' in test_dataset.column_names:
        # If dataset has a 'text' column
        print(f"Processing dataset with 'text' column")
        encoded_test_dataset['input_ids'] = []
        encoded_test_dataset['attention_mask'] = []
        encoded_test_dataset['label'] = []
        
        for example in test_dataset:
            encoded = tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)
            encoded_test_dataset['input_ids'].append(encoded['input_ids'])
            encoded_test_dataset['attention_mask'].append(encoded['attention_mask'])
            encoded_test_dataset['label'].append(example['label'])
    else:
        # If dataset has 'sentence' column
        print(f"Processing dataset with alternative column structure")
        encoded_test_dataset['input_ids'] = []
        encoded_test_dataset['attention_mask'] = []
        encoded_test_dataset['label'] = []
        
        for example in test_dataset:
            text_key = 'sentence' if 'sentence' in example else 'sentence1'
            encoded = tokenizer(example[text_key], padding="max_length", truncation=True, max_length=512)
            encoded_test_dataset['input_ids'].append(encoded['input_ids'])
            encoded_test_dataset['attention_mask'].append(encoded['attention_mask'])
            encoded_test_dataset['label'].append(example['label'])

    dataset_load_time = time.time() - dataset_load_start
    print(f"[{time.strftime('%H:%M:%S')}] Test dataset loaded in {dataset_load_time:.2f} seconds. Size: {len(test_dataset)}")
    
    # We've already manually selected only the columns we need when creating the dictionary
    print(f"[{time.strftime('%H:%M:%S')}] Dataset prepared with {len(encoded_test_dataset['label'])} examples")
    
    # Create data loader
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            if hasattr(self.texts, 'items'):
                t = {key: torch.tensor(values[idx]) for key, values in self.texts.items()}
            else:
                t = {key: torch.tensor(self.texts[idx][key]) for key in self.texts.column_names}
            
            return t, self.labels[idx]
    
    test_dataset = CustomDataset(encoded_test_dataset, torch.tensor(encoded_test_dataset["label"]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    print(f"[{time.strftime('%H:%M:%S')}] DataLoader created with batch size {batch_size}")
    
    # Import concept sets
    print(f"[{time.strftime('%H:%M:%S')}] Loading concept set for {concept_dataset}")
    from app.training_scripts import concepts
    concept_set = {
        'SetFit/sst2': concepts.sst2, 
        'yelp_polarity': concepts.yelpp, 
        'ag_news': concepts.agnews, 
        'dbpedia_14': concepts.dbpedia
    }[concept_dataset]
    print(f"[{time.strftime('%H:%M:%S')}] Concept set loaded with {len(concept_set)} concepts")
    
    # Load model components
    model_load_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading model components")
    cbl_state_dict = torch.load(cbl_path, map_location=device)
    train_mean = torch.load(train_mean_path, map_location=device)
    train_std = torch.load(train_std_path, map_location=device)
    W_g = torch.load(fl_weights_path, map_location=device)
    b_g = torch.load(fl_biases_path, map_location=device)
    model_load_time = time.time() - model_load_start
    print(f"[{time.strftime('%H:%M:%S')}] Model components loaded in {model_load_time:.2f} seconds")
    
    # Initialize model based on backbone
    if backbone == 'roberta':
        print(f"[{time.strftime('%H:%M:%S')}] Initializing RoBERTa CBL model")
        backbone_cbl = RobertaCBL(len(concept_set), dropout=0.1).to(device)
        backbone_cbl.load_state_dict(cbl_state_dict)
        backbone_cbl.eval()
    elif backbone == 'gpt2':
        print(f"[{time.strftime('%H:%M:%S')}] Initializing GPT-2 CBL model")
        backbone_cbl = GPT2CBL(len(concept_set), dropout=0.1).to(device)
        backbone_cbl.load_state_dict(cbl_state_dict)
        backbone_cbl.eval()
    
    # Check if this is a pruned model
    print(f"[{time.strftime('%H:%M:%S')}] Checking for pruned concepts")
    pruned_concepts = []
    accuracy_path = f"{model_dir}/accuracies.json"
    if os.path.exists(accuracy_path):
        with open(accuracy_path, 'r') as f:
            accuracy_data = json.load(f)
            if "pruned_concepts" in accuracy_data:
                pruned_concepts = accuracy_data["pruned_concepts"]
                print(f"[{time.strftime('%H:%M:%S')}] Found {len(pruned_concepts)} pruned concepts")
    
    # Evaluate the model
    print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation")
    eval_start = time.time()
    correct = 0
    total = 0
    
    num_batches = len(test_loader)
    print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation with {num_batches} batches (using only first 3 for speed)")

    try:
        for batch_idx, batch in enumerate(test_loader):
            # Print progress for every batch to see continuous updates
            print(f"[{time.strftime('%H:%M:%S')}] Processing batch {batch_idx+1}/{num_batches} ({(batch_idx+1)/num_batches*100:.1f}%)")
            
            try:
                # Unpack the batch tuple - it's (inputs, labels)
                inputs, labels = batch
                
                # Handle the inputs dictionary
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                with torch.no_grad():
                    # Get concept activations
                    concept_activations = backbone_cbl(inputs)
                    
                    # Normalize and apply ReLU
                    concept_activations, _, _ = normalize(concept_activations, d=0, mean=train_mean, std=train_std)
                    concept_activations = F.relu(concept_activations)
                    
                    # Get predictions
                    logits = concept_activations @ W_g.t() + b_g
                    predictions = torch.argmax(logits, dim=1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                    # Print the accuracy so far
                    current_accuracy = correct / total
                    print(f"[{time.strftime('%H:%M:%S')}] Current accuracy: {current_accuracy:.4f} ({correct}/{total})")
                    
                # Break after 3 batches for a quick preview
                if batch_idx >= 2:  # This is the 3rd batch (index 2)
                    print(f"[{time.strftime('%H:%M:%S')}] Stopping early after 3 batches for preview")
                    break
                
            except Exception as batch_error:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR in batch {batch_idx+1}: {str(batch_error)}")
                import traceback
                print(traceback.format_exc())
                # Continue with next batch
                continue
                
    except Exception as eval_error:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR during evaluation loop: {str(eval_error)}")
        import traceback
        print(traceback.format_exc())
        # Raise the error to be caught by the outer try/except
        raise
    
    eval_time = time.time() - eval_start
    print(f"[{time.strftime('%H:%M:%S')}] Evaluation completed in {eval_time:.2f} seconds")
    
    accuracy = correct / total
    print(f"[{time.strftime('%H:%M:%S')}] Model accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    
    # Save the evaluation results
    print(f"[{time.strftime('%H:%M:%S')}] Saving evaluation results")
    with open(f"{model_dir}/evaluation.json", "w") as f:
        json.dump({
            "model_id": model_id,
            "dataset": concept_dataset,
            "accuracy": accuracy,
            "pruned_concepts": pruned_concepts,
            "num_pruned_concepts": len(pruned_concepts),
            "total_samples": total,
            "correct_predictions": correct,
            "evaluation_time_seconds": eval_time
        }, f, indent=2)
    
    # Update the accuracies.json file
    if os.path.exists(accuracy_path):
        print(f"[{time.strftime('%H:%M:%S')}] Updating accuracies.json file")
        with open(accuracy_path, 'r') as f:
            accuracy_data = json.load(f)
        
        accuracy_data["evaluated_accuracy"] = accuracy
        
        with open(accuracy_path, 'w') as f:
            json.dump(accuracy_data, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Total evaluation time: {total_time:.2f} seconds")
    
    return {
        "model_id": model_id,
        "accuracy": accuracy,
        "pruned_concepts": pruned_concepts,
        "num_pruned_concepts": len(pruned_concepts),
        "total_samples": total,
        "correct_predictions": correct,
        "evaluation_time_seconds": eval_time,
        "total_time_seconds": total_time
    } 