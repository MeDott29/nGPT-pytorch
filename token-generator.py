import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from pathlib import Path

class TokenDataGenerator(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_tokens=32,
        hidden_dim=64,
        pos_freqs=32
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        
        # Generate positional frequencies
        positions = torch.linspace(0, 1, num_tokens).unsqueeze(-1)  # [num_tokens, 1]
        freqs = torch.linspace(1, 10, pos_freqs).unsqueeze(0)  # [1, pos_freqs]
        angles = positions * freqs * 2 * math.pi  # [num_tokens, pos_freqs]
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [num_tokens, 2*pos_freqs]
        
        self.register_buffer('pos_encodings', pos_enc)
        
        # Token processing networks
        self.token_net = nn.Sequential(
            nn.Linear(2 * pos_freqs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projections
        self.key_proj = nn.Linear(hidden_dim, in_features, bias=False)
        self.value_proj = nn.Linear(hidden_dim, out_features, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self):
        # Process tokens through network
        token_features = self.token_net(self.pos_encodings)
        
        # Project to key/value space
        keys = F.normalize(self.key_proj(token_features), dim=-1)
        values = F.normalize(self.value_proj(token_features), dim=-1)
        
        return keys, values

def generate_and_visualize(
    in_features=128,
    out_features=128,
    num_tokens=32,
    frames=100,
    save_path="token_animation.gif"
):
    # Create generator
    generator = TokenDataGenerator(
        in_features=in_features,
        out_features=out_features,
        num_tokens=num_tokens
    )
    
    # Generate initial data
    with torch.no_grad():
        keys, values = generator()
    
    # Convert to numpy for visualization
    keys = keys.numpy()
    values = values.numpy()
    
    # Setup the figure
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Token Linear Generated Data Distribution", fontsize=14)
    
    # Create two subplots for keys and values
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Initial scatter plots
    scatter1 = ax1.scatter([], [], [], c='b', alpha=0.6)
    scatter2 = ax2.scatter([], [], [], c='r', alpha=0.6)
    
    ax1.set_title("Key Embeddings")
    ax2.set_title("Value Embeddings")
    
    # Set axis labels
    for ax in [ax1, ax2]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    
    def init():
        # Project initial points to 3D using PCA-like projection
        key_points = keys[:, :3]  # Take first 3 dimensions for visualization
        val_points = values[:, :3]
        
        scatter1._offsets3d = (key_points[:, 0], key_points[:, 1], key_points[:, 2])
        scatter2._offsets3d = (val_points[:, 0], val_points[:, 1], val_points[:, 2])
        return scatter1, scatter2

    def update(frame):
        # Rotate points
        theta = frame * 2 * np.pi / frames
        
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        key_points = keys[:, :3] @ rotation
        val_points = values[:, :3] @ rotation
        
        scatter1._offsets3d = (key_points[:, 0], key_points[:, 1], key_points[:, 2])
        scatter2._offsets3d = (val_points[:, 0], val_points[:, 1], val_points[:, 2])
        return scatter1, scatter2
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=frames,
        init_func=init, blit=True,
        interval=50  # 20 fps
    )
    
    # Save animation
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(save_path), writer='pillow', fps=20)
    plt.close()
    
    print(f"Animation saved to {save_path}")
    
    return keys, values

if __name__ == "__main__":
    # Generate data and create visualization
    keys, values = generate_and_visualize(
        in_features=128,
        out_features=128,
        num_tokens=32,
        frames=100,
        save_path="outputs/token_animation.gif"
    )