import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import tqdm
from transformers import GPT2Tokenizer
from torch.amp import autocast, GradScaler
import math
import os
import sys
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.accelerator import get_accelerator
import gc
import psutil
# TODO: File another way to import the model
# This is a temporary soloution to import the model, since it is not really a good practice.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.model import Transformer


class Trainer:
    """
    A class to train and validate a Transformer model using PyTorch.
    """
    def __init__(self, 
                model: Transformer,
                train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset,
                tokenizer: GPT2Tokenizer,
                batch_size: int = 32, 
                learning_rate: float = 1e-4,
                mixed_precision: bool = False,
                T_max: int = 10, # Ensure T_max is a parameter
                max_grad_norm: float = 1.0,
                ):   
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate # Store for reference if needed
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.T_max = T_max # Store T_max for scheduler
        self.args = self.model.args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

        # Create optimizer instance before DeepSpeed initialization
        optimizer = DeepSpeedCPUAdam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        

        # Initialize DeepSpeed, passing the optimizer
        # DeepSpeed will manage the optimizer and can also return a scheduler
        self.model_engine, _, _, returned_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer, # Pass the externally created optimizer
            lr_scheduler=self._lr_scheduler_callable,  # Use a callable for the scheduler
            config="train.json"  # Path to your DeepSpeed configuration file
        )
        self.scheduler = returned_scheduler
        
            
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=False)
        self.scaler = GradScaler() if mixed_precision else None

    def _lr_scheduler_callable(self, optimizer):
        """
        A callable for the learning rate scheduler.
        """
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)

    def train_one_epoch(self) -> float:
        """
        Train the model for one epoch.
        """
        self.model_engine.train()
        total_train_loss = 0.0
        scaler = self.scaler if self.mixed_precision else None
        self.check_mem(1)
        get_accelerator().empty_cache()
        self.model_engine.zero_grad()
        torch.cuda.empty_cache()
        for input_ids, targets in self.train_loader: 
            input_ids, targets = input_ids.to(self.device), targets.to(self.device)
            if self.mixed_precision:
                with autocast(self.device.type):
                    logits, aux_loss = self.model_engine(input_ids)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                loss += aux_loss
                self.model_engine.backward(loss)
            else:
                logits, aux_loss = self.model_engine(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss += aux_loss
                self.model_engine.backward(loss)
            
            self.model_engine.step()  # This will handle the optimizer step
            self.model_engine.zero_grad() 
            total_train_loss += loss.item()
            del logits, aux_loss, loss
            del input_ids, targets
            torch.cuda.empty_cache()
            get_accelerator().empty_cache()
        self.check_mem(2)
        return total_train_loss / len(self.train_loader)

    def check_mem(self, id):
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        print(f'memory use: {memoryUse}, id : {id}')
    

    def validate(self) -> float:
        """
        Validate the model on the validation set.
        """
        self.model_engine.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            self.check_mem(3)
            for input_ids, targets in self.val_loader: 
                input_ids, targets = input_ids.to(self.device), targets.to(self.device)
                logits = self.model_engine(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                total_val_loss += loss.item()
                del input_ids, targets, loss
                torch.cuda.empty_cache()
                get_accelerator().empty_cache()
        avg_val = total_val_loss / len(self.val_loader)
        self.check_mem(4)
        return avg_val
    
    def save_model(self, run_name: str):
        """
        Save the model and log it to MLflow.
        """
        torch.save(self.model_engine.module.state_dict(), f"model_{run_name}.pth")
        self.model_engine.save_checkpoint(f"deepspeed_{run_name}")
        print(f"Model saved to {run_name}.pth")
        

    def log_metrics(self, epoch : int, train_loss : float):
        """
        Log training and validation metrics to MLflow.
        """
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("epoch", epoch)
        mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)
        mlflow.log_metric("train_perplexity", math.exp(train_loss), step=epoch)
        if self.device.type == "cuda":
            mlflow.log_metric("GPU Memory Allocated", torch.cuda.memory_allocated(self.device) / 1024 ** 2, step=epoch)
            mlflow.log_metric("GPU Memory Cached", torch.cuda.memory_reserved(self.device) / 1024 ** 2, step=epoch)


    def run(self, num_epochs: int = 1, run_name: str = "BasicLM", curent_dir: str = '') -> Transformer:
        """
        Run the training and validation process.
        """
        gc.collect()
        print(f"Training {self.model.__class__.__name__} for {num_epochs} epochs...")
        print(f"Run name: {run_name}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of Parameters: {self.num_parameters / 1e6:.2f}M")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Max Grad Norm: {self.max_grad_norm}")
        print(f"Device: {self.device}")
        eval_interval = max(1, num_epochs // 10)
        val_iterations = 0
        mlflow.set_tracking_uri(f"file:///{curent_dir}/mlruns") 
        mlflow.set_experiment(run_name)
        self.check_mem(5)
        with mlflow.start_run():
            get_accelerator().empty_cache()
            mlflow.run_name = run_name
            mlflow.log_param("num_epoch", num_epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("n_layers", self.model.n_layers)
            mlflow.log_param("n_heads", self.model.n_heads)
            mlflow.log_param("n_embd", self.model.n_embd)
            mlflow.log_param("max_length", self.model.max_length)
            for epoch in tqdm.tqdm(range(num_epochs)):   
                gc.collect()
                train_loss = self.train_one_epoch()
                self.check_mem(6)
                self.log_metrics(epoch, train_loss)
                if (epoch + 1) % eval_interval == 0:
                    self.check_mem(7)
                    gc.collect()
                    val_loss  = self.validate()
                    mlflow.log_metric("val_perplexity", math.exp(val_loss), step=val_iterations)
                    mlflow.log_metric("val_loss", val_loss, step=val_iterations)
                    val_iterations += 1
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    print(f"Memory reserved: {torch.cuda.memory_reserved(self.device) / 1024 ** 2:.2f} MB")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                get_accelerator().empty_cache()
            print("Training complete.")
        return self.model
