import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)
logger = logging.getLogger()

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from pathlib import Path

class TokenGeneratorNet(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        hidden_dim=128,
        num_tokens=32,
        in_features=128,
        out_features=128,
        pos_freqs=32
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_features = in_features
        self.out_features = out_features
        
        # Position encoding
        positions = torch.linspace(0, 1, num_tokens).unsqueeze(-1)
        freqs = torch.linspace(1, 10, pos_freqs).unsqueeze(0)
        angles = positions * freqs * 2 * math.pi
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        self.register_buffer('pos_encodings', pos_enc)
        
        # Generator networks
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim + 2 * pos_freqs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Separate branches for keys and values
        self.key_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_features)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features)
        )
    
    def forward(self, z):
        batch_size = z.shape[0]
        
        # Expand z and combine with positional encodings
        z_expanded = z.unsqueeze(1).expand(-1, self.num_tokens, -1)
        pos_expanded = self.pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([z_expanded, pos_expanded], dim=-1)
        
        # Generate features
        shared_features = self.shared_net(combined)
        
        # Generate and normalize keys and values
        keys = F.normalize(self.key_net(shared_features), dim=-1)
        values = F.normalize(self.value_net(shared_features), dim=-1)
        
        return keys, values

class TokenDiscriminatorNet(nn.Module):
    def __init__(
        self,
        in_features=128,
        hidden_dim=128,
        num_tokens=32
    ):
        super().__init__()
        
        self.key_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.value_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, keys, values):
        # Encode keys and values
        key_features = self.key_encoder(keys)  # [B, num_tokens, hidden_dim]
        value_features = self.value_encoder(values)  # [B, num_tokens, hidden_dim]
        
        # Global pooling
        key_features = torch.mean(key_features, dim=1)  # [B, hidden_dim]
        value_features = torch.mean(value_features, dim=1)  # [B, hidden_dim]
        
        # Combine features
        combined = torch.cat([key_features, value_features], dim=-1)
        
        # Classify
        return self.classifier(combined)

class TokenAGAN:
    def __init__(
        self,
        latent_dim=64,
        hidden_dim=128,
        num_tokens=32,
        in_features=128,
        out_features=128,
        lr=1e-4
    ):
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.generator = TokenGeneratorNet(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            in_features=in_features,
            out_features=out_features
        ).to(self.device)
        
        self.discriminator = TokenDiscriminatorNet(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens
        ).to(self.device)
        
        # Initialize optimizers with default Adam
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Initialize loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Track history
        self.g_losses = []
        self.d_losses = []
    
    def train_step(self, real_keys, real_values):
        batch_size = real_keys.shape[0]
        real_keys = real_keys.to(self.device)
        real_values = real_values.to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        with torch.no_grad():
            fake_keys, fake_values = self.generator(z)
        
        # Get predictions
        real_pred = self.discriminator(real_keys, real_values)
        fake_pred = self.discriminator(fake_keys, fake_values)
        
        # Calculate loss
        d_real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        d_fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        d_loss = d_real_loss + d_fake_loss
        
        # Update discriminator
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate new fake data
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_keys, fake_values = self.generator(z)
        
        # Get new predictions for fake data
        fake_pred = self.discriminator(fake_keys, fake_values)
        
        # Calculate loss with diversity term
        g_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        
        # Add diversity loss
        diversity_loss = -torch.mean(torch.pdist(fake_keys.view(batch_size, -1)))
        g_loss = g_loss + 0.1 * diversity_loss
        
        # Update generator
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()

    def generate(self, num_samples=1):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            keys, values = self.generator(z)
        return keys.cpu(), values.cpu()

def visualize_embeddings(keys, values, epoch, save_path):
    """Create a visualization of the embeddings."""
    # Take first 3 dimensions for visualization
    keys = keys[0, :, :3].numpy()
    values = values[0, :, :3].numpy()
    
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Generated Token Embeddings (Epoch {epoch})")
    
    # Plot keys
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(keys[:, 0], keys[:, 1], keys[:, 2], c='b', alpha=0.6)
    ax1.set_title("Key Embeddings")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Plot values
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(values[:, 0], values[:, 1], values[:, 2], c='r', alpha=0.6)
    ax2.set_title("Value Embeddings")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    plt.savefig(save_path)
    plt.close()

def create_animation(image_dir, output_path):
    """Create a GIF animation from a directory of images."""
    import glob
    from PIL import Image
    
    # Get all PNG files
    files = sorted(glob.glob(str(image_dir / "epoch_*.png")))
    if not files:
        print("No images found for animation")
        return
        
    frames = []
    for filename in files:
        frame = Image.open(filename)
        frames.append(frame)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0
    )

def main():
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize aGAN
    agan = TokenAGAN(
        latent_dim=64,
        hidden_dim=128,
        num_tokens=32,
        in_features=128,
        out_features=128,
        lr=1e-4
    )
    
    # Training parameters
    num_epochs = 100
    batch_size = 32
    save_interval = 5
    
    logger.info("Starting training...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Generate random "real" data for demonstration
        real_keys = F.normalize(torch.randn(batch_size, agan.num_tokens, 128), dim=-1)
        real_values = F.normalize(torch.randn(batch_size, agan.num_tokens, 128), dim=-1)
        
        # Train step
        d_loss, g_loss = agan.train_step(real_keys, real_values)
        
        logger.info(f"Epoch {epoch}: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}")
        
        # Save visualization periodically
        if epoch % save_interval == 0:
            keys, values = agan.generate(num_samples=1)
            save_path = output_dir / f"epoch_{epoch:03d}.png"
            visualize_embeddings(keys, values, epoch, save_path)
        
        # Save model checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': agan.generator.state_dict(),
                'discriminator_state_dict': agan.discriminator.state_dict(),
                'g_optimizer_state_dict': agan.g_optimizer.state_dict(),
                'd_optimizer_state_dict': agan.d_optimizer.state_dict(),
                'g_losses': agan.g_losses,
                'd_losses': agan.d_losses
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")

    # Create final animation
    create_animation(output_dir, output_dir / "training.gif")
    logger.info("Training complete! Animation saved to outputs/training.gif")

if __name__ == "__main__":
    main()