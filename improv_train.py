import math
import gzip
import random
import numpy as np
from contextlib import nullcontext
from datetime import datetime, timedelta
import time
import json
import os
from typing import Optional

import torch
from torch.optim import Adam
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.parametrize as parametrize

from nGPT_pytorch import nGPT
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# Training configuration
class TrainingConfig:
    def __init__(self):
        self.num_batches = int(1e5)
        self.batch_size = 4
        self.grad_accum_every = 4
        self.learning_rate = 1e-3
        self.validate_every = 100
        self.prime_length = 128
        self.generate_every = 500
        self.generate_length = 512
        self.seq_len = 512
        self.use_amp = True
        self.use_parametrize = True
        self.checkpoint_dir = "checkpoints"
        
    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
            
    @classmethod
    def load(cls, filename: str) -> 'TrainingConfig':
        config = cls()
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                config.__dict__.update(json.load(f))
        return config

# Helper functions
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.to(self.device)

class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.console = Console()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
    def setup_model(self):
        self.model = nGPT(
            num_tokens=256,
            dim=512,
            depth=8,
            tied_embedding=True,
            add_value_residual=True,
            attn_norm_qk=False,
            manual_norm_weights=not self.config.use_parametrize
        ).to(self.device)
        
        self.scaler = GradScaler(enabled=self.config.use_amp)
        
    def setup_data(self):
        with gzip.open("./data/enwik8.gz") as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            self.data_train = torch.from_numpy(np_train)
            self.data_val = torch.from_numpy(np_valid)

        self.train_dataset = TextSamplerDataset(self.data_train, self.config.seq_len, self.device)
        self.val_dataset = TextSamplerDataset(self.data_val, self.config.seq_len, self.device)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size
        )
        
    def setup_training(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        if not self.config.use_parametrize:
            self.model.register_step_post_hook(self.optim)
            
    def save_checkpoint(self, step: int, loss: float):
        checkpoint = {
            'step': step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optim.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optim.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['step'], checkpoint['loss']
        
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        batch = next(iter(self.val_loader))
        
        with torch.no_grad():
            loss = self.model(batch, return_loss=True)
            total_loss += loss.item()
            
        return total_loss
    
    def log(self, t, eps = 1e-20):
        return torch.log(t.clamp(min = eps))

    def gumbel_noise(self, t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))

    def gumbel_sample(self, t, temperature = 1., dim = -1, keepdim = True):
        return ((t / max(temperature, 1e-10)) + self.gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

    def min_p_filter(self, logits, min_p = 0.1):
        probs = logits.softmax(dim = -1)
        max_probs = probs.amax(dim = -1, keepdim = True)
        limit = min_p * max_probs
        return torch.where(probs < limit, float('-inf'), logits)

    def base_decoding(
        self,
        net,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.5,
        min_p = 1e-1,
        filter_thres = 0.9,
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        for _ in range(sample_num_times):
            logits = net(out)
            logits = logits[:, -1]

            logits = self.min_p_filter(logits, min_p = min_p)
            sample = self.gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    def generate_sample(self) -> str:
        self.model.eval()
        
        inp = random.choice(self.val_dataset)[:self.config.prime_length]
        prime = decode_tokens(inp)
        prompt = inp[None, ...]

        with torch.no_grad():
            sampled = self.base_decoding(
                self.model,
                prompt,
                self.config.generate_length,
                temperature=1.5,
                min_p=0.1,
                filter_thres=0.9
            )
                
        generated = decode_tokens(sampled[0])
        return f"{prime}\n\n{'=' * 40}\n\n{generated}"

    def train(self):
        start_time = time.time()
        best_loss = float('inf')
        running_loss = 0
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("[cyan]Training...", total=self.config.num_batches)
            
            for i in range(self.config.num_batches):
                self.model.train()
                
                # Gradient accumulation loop
                for _ in range(self.config.grad_accum_every):
                    batch = next(iter(self.train_loader))
                    
                    with torch.autocast(
                        device_type='cuda',
                        dtype=torch.float16,
                        enabled=self.config.use_amp
                    ):
                        loss = self.model(batch, return_loss=True)
                        
                    self.scaler.scale(loss / self.config.grad_accum_every).backward()
                    running_loss += loss.item() / self.config.grad_accum_every
                
                # Update weights
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
                
                # Progress update
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Training... Loss: {running_loss:.4f}"
                )
                
                # Validation
                if i % self.config.validate_every == 0:
                    val_loss = self.validate()
                    self.console.print(f"\nValidation loss: {val_loss:.4f}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self.save_checkpoint(i, val_loss)
                        self.console.print("[green]Saved new best checkpoint!")
                
                # Generate sample
                if i % self.config.generate_every == 0:
                    sample = self.generate_sample()
                    self.console.print(f"\nGenerated sample:\n{sample}\n")
                
                running_loss = 0
                
        training_time = time.time() - start_time
        self.console.print(f"\nTraining completed in {format_time(training_time)}")
        self.console.print(f"Best validation loss: {best_loss:.4f}")

def main():
    console = Console()
    
    # Load or create config
    config_file = "training_config.json"
    config = TrainingConfig.load(config_file) if os.path.exists(config_file) else TrainingConfig()
    
    # Show configuration
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config.__dict__.items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    if Confirm.ask("Start training with these parameters?"):
        # Save config
        config.save(config_file)
        
        # Start training
        trainer = TrainingManager(config)
        trainer.train()
    else:
        console.print("Training cancelled")

if __name__ == "__main__":
    main()