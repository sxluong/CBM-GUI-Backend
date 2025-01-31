import argparse
import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset, concatenate_datasets
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from training_utils import cos_sim_cubed, get_labels, eos_pooling
import time

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--backbone", type=str, default="roberta", help="roberta or gpt2")
parser.add_argument("--model_id", type=str, required=True, help="Unique model ID to load saved concept files")  # Added model_id
parser.add_argument('--tune_cbl_only', action=argparse.BooleanOptionalAction)
parser.add_argument('--automatic_concept_correction', action=argparse.BooleanOptionalAction)
parser.add_argument("--labeling", type=str, default="mpnet", help="mpnet, angle, simcse, llm")
parser.add_argument("--cbl_only_batch_size", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    if args.dataset == 'SetFit/sst2':
        val_dataset = load_dataset(args.dataset, split='validation')

    print(f"Training data length: {len(train_dataset)}")
    if args.dataset == 'SetFit/sst2':
        print(f"Validation data length: {len(val_dataset)}")

    print("Tokenizing datasets...")
    if args.backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif args.backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("Backbone should be either roberta or gpt2")

    # Construct the correct prefix path to load concept similarity matrices
    d_name = args.dataset.replace('/', '_')
    prefix = f"./{args.labeling}_acs/{d_name}/model_{args.model_id}/"

    # Load the precomputed concept similarity matrices
    print(f"Loading concept similarity matrices from {prefix}...")
    train_similarity_path = os.path.join(prefix, "concept_labels_train.npy")
    val_similarity_path = os.path.join(prefix, "concept_labels_val.npy")

    if not os.path.exists(train_similarity_path):
        raise FileNotFoundError(f"Concept labels file not found: {train_similarity_path}")

    train_similarity = np.load(train_similarity_path)
    val_similarity = np.load(val_similarity_path) if args.dataset == 'SetFit/sst2' else None

    print(f"Loaded training similarity matrix: {train_similarity.shape}")
    if val_similarity is not None:
        print(f"Loaded validation similarity matrix: {val_similarity.shape}")

    # Apply automatic concept correction if specified
    if args.automatic_concept_correction:
        print("Applying automatic concept correction...")
        for i in range(train_similarity.shape[0]):
            for j in range(len(CFG.concept_set[args.dataset])):
                if get_labels(j, args.dataset) != train_dataset["label"][i] or train_similarity[i][j] < 0.0:
                    train_similarity[i][j] = 0.0

        if args.dataset == 'SetFit/sst2':
            for i in range(val_similarity.shape[0]):
                for j in range(len(CFG.concept_set[args.dataset])):
                    if get_labels(j, args.dataset) != val_dataset["label"][i] or val_similarity[i][j] < 0.0:
                        val_similarity[i][j] = 0.0

    # Tokenize datasets
    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length),
        batched=True
    )

    encoded_val_dataset = None
    if args.dataset == 'SetFit/sst2':
        encoded_val_dataset = val_dataset.map(
            lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length),
            batched=True
        )

    # Model Training Directory
    model_save_prefix = os.path.join(prefix, "trained_model")
    if not os.path.exists(model_save_prefix):
        os.makedirs(model_save_prefix)

    # Initialize model
    print("Initializing model...")
    concept_set_size = len(CFG.concept_set[args.dataset])
    if args.tune_cbl_only:
        cbl = CBL(concept_set_size, args.dropout).to(device)
        preLM = RobertaModel.from_pretrained('roberta-base').to(device) if args.backbone == 'roberta' else GPT2Model.from_pretrained('gpt2').to(device)
        preLM.eval()
        optimizer = torch.optim.Adam(cbl.parameters(), lr=1e-4)
    else:
        backbone_cbl = RobertaCBL(concept_set_size, args.dropout).to(device) if args.backbone == 'roberta' else GPT2CBL(concept_set_size, args.dropout).to(device)
        optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=5e-6)

    # Start Training
    print("Starting training...")
    best_loss = float('inf')
    epochs = CFG.cbl_epochs[args.dataset]

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")

        # Training phase
        model = cbl if args.tune_cbl_only else backbone_cbl
        model.train()

        training_loss = []
        for batch_text, batch_sim in encoded_train_dataset:
            batch_text = {k: torch.tensor(v).to(device) for k, v in batch_text.items()}
            batch_sim = torch.tensor(batch_sim).to(device)

            if args.tune_cbl_only:
                with torch.no_grad():
                    LM_features = preLM(batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                    if args.backbone == 'roberta':
                        LM_features = LM_features[:, 0, :]
                    elif args.backbone == 'gpt2':
                        LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                cbl_features = cbl(LM_features)
            else:
                cbl_features = backbone_cbl(batch_text)

            loss = -cos_sim_cubed(cbl_features, batch_sim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())

        avg_training_loss = np.mean(training_loss)
        print(f"Training Loss: {avg_training_loss}")

        # Save best model
        if avg_training_loss < best_loss:
            best_loss = avg_training_loss
            save_path = os.path.join(model_save_prefix, f"best_model_epoch_{e+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model: {save_path}")

    print("Training completed!")