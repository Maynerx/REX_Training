
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import argparse
from trainer import Trainer
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
# ignore warnings
import warnings
from omegaconf import OmegaConf
import tiktoken
warnings.filterwarnings('ignore')

import os
import sys

# TODO: File another way to import the model
# Same problem as in the trainer.py file
CURRENT_DIR = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.model import Transformer


tokenizer = tiktoken.get_encoding("r50k_base") # Using tiktoken for GPT-4 encoding



def chunkify(token_list, block_size, stride=None):
    if stride is None:
        stride = block_size
    blocks = []
    for i in range(0, len(token_list) - block_size, stride):
        x = token_list[i : i + block_size]
        y = token_list[i + 1 : i + block_size + 1]
        blocks.append((x, y))
    return blocks

class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        x, y = self.blocks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# === Loading Dataset ===
def load_dataset(file_path: str, train_ratio: int, val_ratio: int, max_length: int) -> tuple[Dataset, Dataset]:
    """
    Load the dataset from a text file and split it into training and validation sets.
    """
    with open(f"{file_path}", 'r', encoding='utf-8') as f:
        text = f.read()
    #tokenizer = tiktoken.get_encoding("o200k_base") # Using tiktoken for GPT-4 encoding
    #tokens = tokenizer.encode(text)
    tokens = tokenizer.encode(text)
    N = len(tokens)
    train_end = int(train_ratio * N)
    val_end = train_end + int(val_ratio * N)
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    train_blocks = chunkify(train_tokens, max_length)
    val_blocks = chunkify(val_tokens,   max_length)
    print(f"Train blocks: {len(train_blocks)}, Val blocks: {len(val_blocks)}")

    train_ds = BlockDataset(train_blocks)
    val_ds   = BlockDataset(val_blocks)
    return train_ds, val_ds


# === Main Function ===
def main():
    cfg = OmegaConf.load("../config/config.yaml")
    run_name = cfg.training.run_name



    # === Model ===
    model = Transformer(
        vocab_size=tokenizer.n_vocab,
        n_embd=cfg.model.n_embd,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_len=cfg.model.max_length,
        num_dense_layers=cfg.model.num_dense_layers,
        num_expert=cfg.model.num_experts,
        score_fn=cfg.model.score_fn,
        top_k=cfg.model.top_k,
        inter_size=cfg.model.inner_dim
    )

    # === Load Dataset ===
    train_ds, val_ds = load_dataset(
        file_path="input.txt",
        train_ratio=cfg.training.train_ratio,
        val_ratio=cfg.training.val_ratio,
        max_length=cfg.model.max_length
    )


    # === Training ===
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        learning_rate=cfg.training.lr,
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        mixed_precision=cfg.training.mixed_precision,
        T_max=cfg.training.epochs,
        max_grad_norm=cfg.training.max_grad_norm,
    )

    trainer.run(
        num_epochs=cfg.training.epochs, 
        run_name=run_name, 
        curent_dir=CURRENT_DIR
    
    )
    # Should change this to the current directory
    trainer.save_model(
        run_name=run_name, 
    )

if __name__ == "__main__":
    main()


            
