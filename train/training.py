
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
warnings.filterwarnings('ignore')

import os
import sys

# TODO: File another way to import the model
# Same problem as in the trainer.py file
CURRENT_DIR = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.model import Transformer



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token



class BlockDataset(Dataset):
    def __init__(self, token_ids: torch.LongTensor, block_size: int):
        n_tokens = token_ids.size(0)
        n_blocks = (n_tokens - 1) // block_size
        usable = token_ids[: n_blocks * block_size + 1]
        self.inputs = usable[:-1].view(n_blocks, block_size)
        self.labels = usable[1:].view(n_blocks, block_size)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels":    self.labels[idx],
        }

# === Loading Dataset ===


# Should be changed too 
def load_wikitext_block(tokenizer, block_size=512):
    # 1) load and filter out empty lines
    ds = load_dataset("wikitext", "wikitext-2-v1")
    train_texts = [t for t in ds["train"]["text"]       if t.strip() != ""]
    val_texts   = [t for t in ds["validation"]["text"]  if t.strip() != ""]

    # 2) concatenate & tokenize once per split
    all_train = " ".join(train_texts)
    all_val   = " ".join(val_texts)

    train_ids = tokenizer(all_train, return_tensors="pt")["input_ids"].squeeze(0)
    val_ids   = tokenizer(all_val,   return_tensors="pt")["input_ids"].squeeze(0)

    # 3) build block datasets
    train_ds = BlockDataset(train_ids, block_size)
    val_ds   = BlockDataset(val_ids,   block_size)
    return train_ds, val_ds


# === Main Function ===
def main():
    cfg = OmegaConf.load("../config/config.yaml")
    run_name = cfg.training.run_name



    # === Model ===
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg.model.n_embd,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_len=cfg.model.max_length
    )

    model_state = torch.load(
        cfg.model.pretrained_model_path, 
        map_location=torch.device('cpu')
    )

    model.load_state_dict(model_state)
    

    # === Load Dataset ===
    train_ds, val_ds = load_wikitext_block(
        tokenizer=tokenizer,
        block_size=cfg.model.max_length
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


            
