from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from functools import partial

# Utility functions
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

from nGPT_pytorch.nGPT import (
    l2norm,
    NormLinear,
    Scale,
    Residual,
    FeedForward
)

class Word2VecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: list[str],
        vocab: dict[str, int],
        window_size: int = 5,
        mode: str = 'skipgram'
    ):
        """
        Args:
            texts: List of tokenized sentences (list of strings)
            vocab: Dictionary mapping words to indices
            window_size: Context window size
            mode: 'skipgram' or 'cbow'
        """
        self.texts = texts
        self.vocab = vocab
        self.window_size = window_size
        self.mode = mode
        self.pairs = self._create_pairs()
    
    def _create_pairs(self):
        pairs = []
        for text in self.texts:
            indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
            
            for i in range(len(indices)):
                # Get context window
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                context = indices[start:i] + indices[i+1:end]
                
                if self.mode == 'skipgram':
                    # For skip-gram, target word predicts context words
                    for ctx in context:
                        pairs.append((indices[i], ctx))
                else:
                    # For CBOW, context words predict target word
                    pairs.append((context, indices[i]))
                    
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

class nWord2Vec(Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        embedding_dim: int = 300,
        mode: str = 'skipgram',
        use_feedforward: bool = True,
        ff_expand_factor: float = 2.,
        manual_norm_weights: bool = False,
        s_logit_init: float = 1.,
        s_logit_scale: float | None = None,
        norm_eps: float = 0.,
        num_hyperspheres: int = 1,
        negative_samples: int = 5
    ):
        """
        Args:
            num_tokens: Vocabulary size
            embedding_dim: Dimension of word embeddings
            mode: 'skipgram' or 'cbow'
            use_feedforward: Whether to use a feedforward layer
            ff_expand_factor: Expansion factor for feedforward layer
            manual_norm_weights: Whether to manually normalize weights
            s_logit_init: Initial scale for logits
            s_logit_scale: Scale factor for logits
            norm_eps: Epsilon for normalization
            num_hyperspheres: Number of hyperspheres for normalization
            negative_samples: Number of negative samples for training
        """
        super().__init__()
        
        self.mode = mode
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        
        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres
        )
        
        self.l2norm = partial(l2norm, norm_eps=norm_eps, groups=num_hyperspheres)
        
        # Input embeddings
        self.input_embeddings = NormLinear_(embedding_dim, num_tokens)
        
        # Optional feedforward layer
        self.use_feedforward = use_feedforward
        if use_feedforward:
            self.ff = FeedForward(
                embedding_dim,
                expand_factor=ff_expand_factor,
                manual_norm_weights=manual_norm_weights,
                norm_eps=norm_eps,
                num_hyperspheres=num_hyperspheres
            )
        
        # Output embeddings (context for skip-gram, target for CBOW)
        self.output_embeddings = NormLinear_(embedding_dim, num_tokens)
        
        # Logit scaling
        self.logit_scale = Scale(
            num_tokens,
            s_logit_init,
            default(s_logit_scale, embedding_dim ** -0.5)
        )
        
        # Initialize embeddings
        self.init_embeddings()
    
    def init_embeddings(self):
        """Initialize embeddings using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)
    
    @torch.no_grad()
    def norm_weights_(self):
        """Normalize weights of embedding layers"""
        for module in self.modules():
            if isinstance(module, NormLinear):
                module.norm_weights_()
    
    def forward(
        self,
        input_ids,
        target_ids=None,
        return_loss=False
    ):
        """
        Forward pass
        
        Args:
            input_ids: Input word indices (for skip-gram) or context indices (for CBOW)
            target_ids: Target word indices (optional, for training)
            return_loss: Whether to return loss
        """
        if self.mode == 'skipgram':
            # Get input embeddings
            embeddings = self.input_embeddings.weight[input_ids]
        else:  # CBOW
            # Average context embeddings
            context_embeds = self.input_embeddings.weight[input_ids]
            embeddings = context_embeds.mean(dim=1)
        
        # Apply feedforward if enabled
        if self.use_feedforward:
            embeddings = self.ff(embeddings)
        
        # Compute logits
        logits = F.linear(
            embeddings,
            self.output_embeddings.weight
        ) * self.logit_scale()
        
        if not return_loss:
            return logits
        
        # Compute loss using negative sampling
        positive_logits = logits.gather(1, target_ids.unsqueeze(-1))
        
        # Generate negative samples
        num_samples = logits.size(0)
        negative_samples = torch.randint(
            0, self.num_tokens,
            (num_samples, self.negative_samples),
            device=logits.device
        )
        negative_logits = logits.gather(1, negative_samples)
        
        # Combined loss using negative sampling
        positive_loss = F.logsigmoid(positive_logits).mean()
        negative_loss = F.logsigmoid(-negative_logits).mean()
        
        loss = -(positive_loss + negative_loss)
        
        return loss
    
    def get_embeddings(self):
        """Return normalized word embeddings"""
        return self.l2norm(self.input_embeddings.weight.detach())

def train_word2vec(
    model: nWord2Vec,
    dataset: Word2VecDataset,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Train the Word2Vec model
    
    Args:
        model: nWord2Vec model
        dataset: Word2Vec dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            if model.mode == 'skipgram':
                input_ids, target_ids = batch
            else:  # CBOW
                context_ids, target_ids = batch
                input_ids = context_ids
            
            loss = model(input_ids, target_ids, return_loss=True)
            loss.backward()
            
            optimizer.step()
            model.norm_weights_()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Example usage:
if __name__ == "__main__":
    # Sample vocabulary and texts
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'quick': 3,
        'brown': 4,
        'fox': 5,
        'jumps': 6,
        'over': 7,
        'lazy': 8,
        'dog': 9
    }
    
    texts = [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    ]
    
    # Create dataset
    dataset = Word2VecDataset(
        texts=texts,
        vocab=vocab,
        window_size=2,
        mode='skipgram'
    )
    
    # Create and train model
    model = nWord2Vec(
        num_tokens=len(vocab),
        embedding_dim=64,
        mode='skipgram'
    )
    
    train_word2vec(
        model=model,
        dataset=dataset,
        num_epochs=5,
        batch_size=16
    )
    
    # Get trained embeddings
    embeddings = model.get_embeddings()
