import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    """Configuration for the hyperspherical data generator"""
    dim: int = 512  # Embedding dimension
    num_tokens: int = 32000  # Vocabulary size
    sequence_length: int = 1024  # Length of generated sequences
    num_positions: int = 256  # Number of position tokens for TokenLinear
    hidden_dim: int = 64  # Hidden dimension for token processing
    num_heads: int = 4  # Number of attention heads
    dropout: float = 0.1
    temperature: float = 1.0
    norm_eps: float = 1e-5

class HypersphericalNorm(nn.Module):
    """Ensures outputs lie on the hypersphere"""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + self.eps)

class PositionalTokenGenerator(nn.Module):
    """Generates positional tokens using frequency mixing"""
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # Create frequency mixture encodings
        positions = torch.linspace(0, 1, config.num_positions).unsqueeze(-1)
        freqs = torch.linspace(1, 10, config.hidden_dim // 2).unsqueeze(0)
        angles = positions * freqs * 2 * math.pi
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        self.register_buffer('position_encodings', pos_enc)
        
        # Project to embedding space
        self.pos_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.dim),
            HypersphericalNorm(config.norm_eps)
        )
    
    def forward(self) -> torch.Tensor:
        return self.pos_proj(self.position_encodings)

class TokenMixer(nn.Module):
    """Mixes token embeddings using attention"""
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        head_dim = config.dim // config.num_heads
        
        self.to_qkv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim, config.dim),
                HypersphericalNorm(config.norm_eps)
            ) for _ in range(3)
        ])
        
        self.to_out = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            HypersphericalNorm(config.norm_eps)
        )
        
        self.scale = head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = [proj(x) for proj in self.to_qkv]
        
        # Split heads
        q, k, v = [rearrange_heads(t, self.config.num_heads) for t in (q, k, v)]
        
        # Attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask, -torch.finfo(dots.dtype).max)
        attn = F.softmax(dots, dim=-1)
        
        # Combine heads and project
        out = torch.matmul(attn, v)
        out = rearrange_heads_back(out)
        return self.to_out(out)

class HypersphericalDataGenerator(nn.Module):
    """Main data generation architecture"""
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # Token embedding generator
        self.token_embeddings = TokenLinearGL(
            in_features=config.dim,
            out_features=config.dim,
            num_tokens=config.num_tokens
        )
        
        # Positional embeddings
        self.pos_embeddings = PositionalTokenGenerator(config)
        
        # Token mixer
        self.mixer = TokenMixer(config)
        
        # Output projection
        self.to_logits = nn.Sequential(
            nn.Linear(config.dim, config.num_tokens),
            HypersphericalNorm(config.norm_eps)
        )
        
        self.temperature = config.temperature
    
    def generate_sequence(
        self, 
        prompt: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        max_length = max_length or self.config.sequence_length
        
        # Start with prompt or generate from scratch
        if prompt is None:
            sequence = torch.randint(0, self.config.num_tokens, (1, 1), device=device)
        else:
            sequence = prompt
            
        # Generate sequence auto-regressively
        for _ in range(max_length - sequence.size(1)):
            # Get embeddings for current sequence
            embeddings = self.get_embeddings(sequence)
            
            # Generate next token probabilities
            logits = self.get_next_token_logits(embeddings)
            probs = F.softmax(logits / self.temperature, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs[:, -1], 1)
            sequence = torch.cat([sequence, next_token], dim=1)
            
        return sequence
    
    def get_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        # Get token embeddings
        token_embeddings = self.token_embeddings(
            torch.zeros(ids.size(0), ids.size(1), self.config.dim, device=ids.device)
        )
        
        # Add positional embeddings
        pos_embeddings = self.pos_embeddings()[:ids.size(1)]
        embeddings = token_embeddings + pos_embeddings
        
        # Mix tokens
        return self.mixer(embeddings)
    
    def get_next_token_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.to_logits(embeddings)

def rearrange_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Rearrange tensor to separate attention heads"""
    b, n, d = x.shape
    return x.reshape(b, n, num_heads, -1).transpose(1, 2)

def rearrange_heads_back(x: torch.Tensor) -> torch.Tensor:
    """Merge attention heads back"""
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)

# Example usage:
if __name__ == "__main__":
    config = GeneratorConfig()
    generator = HypersphericalDataGenerator(config)
    
    # Generate a sequence
    sequence = generator.generate_sequence(max_length=64)
    print(f"Generated sequence shape: {sequence.shape}")
