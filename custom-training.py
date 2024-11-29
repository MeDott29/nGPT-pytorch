import math
import random
import tqdm
import numpy as np
from pathlib import Path

import torch
from torch.optim import Adam
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.parametrize as parametrize

from nGPT_pytorch import nGPT

# Training configuration
class Config:
    NUM_BATCHES = int(1e5)
    BATCH_SIZE = 4
    GRAD_ACCUM_EVERY = 4
    LEARNING_RATE = 1e-3
    VALIDATE_EVERY = 100
    PRIME_LENGTH = 128
    GENERATE_EVERY = 500
    GENERATE_LENGTH = 512
    SEQ_LEN = 512
    VOCAB_SIZE = 256  # Modify based on your tokenization
    
    # Model configuration
    MODEL_DIM = 512
    MODEL_DEPTH = 8
    
    # Training settings
    USE_AMP = True
    USE_PARAMETRIZE = True
    
    # Data paths
    TRAIN_DATA_PATH = "path/to/train.txt"
    VAL_DATA_PATH = "path/to/val.txt"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TextDataset(Dataset):
    def __init__(self, data_path, seq_len, vocab_size):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Load and preprocess your data here
        path = Path(data_path)
        with path.open('r', encoding='utf-8') as f:
            text = f.read()
        
        # Example: basic character-level tokenization
        # Modify this based on your tokenization strategy
        self.data = torch.tensor([ord(c) % self.vocab_size for c in text], dtype=torch.long)
    
    def __len__(self):
        return max(0, self.data.size(0) - self.seq_len)
    
    def __getitem__(self, index):
        chunk = self.data[index:index + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        return chunk.to(device)

def create_dataloaders(config):
    train_dataset = TextDataset(config.TRAIN_DATA_PATH, config.SEQ_LEN, config.VOCAB_SIZE)
    val_dataset = TextDataset(config.VAL_DATA_PATH, config.SEQ_LEN, config.VOCAB_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    return train_loader, val_loader

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_AMP else 'cpu')
    
    model = nGPT(
        num_tokens=config.VOCAB_SIZE,
        dim=config.MODEL_DIM,
        depth=config.MODEL_DEPTH,
        tied_embedding=True,
        add_value_residual=True,
        attn_norm_qk=False,
        manual_norm_weights=not config.USE_PARAMETRIZE
    ).to(device)
    
    scaler = GradScaler(enabled=config.USE_AMP)
    optim = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_loader, val_loader = create_dataloaders(config)
    
    # Training loop
    for i in tqdm.tqdm(range(config.NUM_BATCHES), mininterval=10.0, desc="training"):
        model.train()
        
        for _ in range(config.GRAD_ACCUM_EVERY):
            data = next(iter(train_loader))
            data = data.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.USE_AMP):
                loss = model(data, return_loss=True)
            
            scaler.scale(loss / config.GRAD_ACCUM_EVERY).backward()
        
        print(f"training loss: {loss.item():.3f}")
        
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        
        if i % config.VALIDATE_EVERY == 0:
            validate(model, val_loader, device)
        
        if i % config.GENERATE_EVERY == 0:
            generate_sample(model, val_loader, config, device)

def validate(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        valid_data = next(iter(val_loader)).to(device)
        loss = model(valid_data, return_loss=True)
        print(f"validation loss: {loss.item():.3f}")

def generate_sample(model, val_loader, config, device):
    model.eval()
    sample_data = next(iter(val_loader))[0][:config.PRIME_LENGTH].to(device)
    
    # Modify this based on your tokenization/detokenization
    def decode_tokens(tokens):
        return "".join(chr(max(32, int(t))) for t in tokens)
    
    prime = decode_tokens(sample_data)
    print(f"{prime} \n\n {'*' * 100}")
    
    prompt = sample_data[None, ...]
    sampled = base_decoding(model, prompt, config.GENERATE_LENGTH)
    output = decode_tokens(sampled[0])
    print(f"\n\n{output}\n")

if __name__ == "__main__":
    # Example configuration
    config = Config(
        TRAIN_DATA_PATH="data/train.txt",
        VAL_DATA_PATH="data/val.txt",
        VOCAB_SIZE=256,  # Adjust based on your tokenization
        NUM_BATCHES=1000,  # Adjust based on your needs
    )
    
    train(config)