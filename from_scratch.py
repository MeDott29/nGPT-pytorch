import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None

class Attention(nn.Module):
    """Multi-head self attention mechanism."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        # Linear projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x : torch.Tensor, shape (batch, seq_len, dim)
        mask : torch.Tensor, shape (batch, seq_len)
        """
        h = self.heads

        # Project input to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Masking for causal attention (optional)
        if self.causal:
            mask = torch.ones(dots.shape[-2:], device=x.device).triu(1).bool()
            dots.masked_fill_(mask, float('-inf'))

        # Softmax
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Multiply by values and combine heads
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class FeedForward(nn.Module):
    """Simple feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # Attention block with residual
        x = x + self.attention(self.attn_norm(x), mask=mask)
        
        # Feed-forward block with residual
        x = x + self.ff(self.ff_norm(x))
        
        return x

class SimpleTransformer(nn.Module):
    """Simple transformer architecture for language modeling."""
    
    def __init__(
        self,
        *,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Token embedding and positional encoding
        self.token_embed = nn.Embedding(num_tokens, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, dim))  # Fixed max length of 1024
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, ff_mult, dropout)
            for _ in range(depth)
        ])
        
        # Output layer
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # Layer normalization before output
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        b, n = x.shape
        device = x.device
        
        # Get embeddings and add positional encoding
        x = self.token_embed(x)
        x = x + self.pos_embed[:, :n]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
            
        # Apply final normalization
        x = self.norm(x)
        
        # Convert to logits
        return self.to_logits(x)

# Test the implementation
if __name__ == "__main__":
    # Create model
    model = SimpleTransformer(
        num_tokens=256,  # vocabulary size
        dim=512,         # embedding dimension
        depth=6,         # number of transformer blocks
        heads=8,         # number of attention heads
        dim_head=64,     # dimension per head
        ff_mult=4        # feed-forward expansion factor
    )
    
    # Create sample input
    x = torch.randint(0, 256, (2, 128))  # batch_size=2, seq_len=128
    
    # Forward pass
    logits = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")