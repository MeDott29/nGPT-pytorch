"""
Hypersphere Visualization for nGPT

This script creates a 3D visualization of token embeddings on the hypersphere,
helping to understand how nGPT represents tokens in normalized space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import argparse

from nGPT_pytorch import nGPT

def visualize_embeddings_3d(model, num_tokens=256, save_path=None):
    """
    Visualize token embeddings on a 3D hypersphere
    """
    # Get token embeddings
    token_embed = model.token_embed.weight.detach().cpu().numpy()
    
    # Use PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    
    # Get a subset of tokens for visualization
    num_vis_tokens = min(100, num_tokens)
    token_indices = np.linspace(0, num_tokens-1, num_vis_tokens).astype(int)
    
    # Get embeddings for these tokens
    embeddings = token_embed[token_indices]
    
    # Apply PCA
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Normalize to unit sphere
    norms = np.sqrt(np.sum(embeddings_3d**2, axis=1, keepdims=True))
    embeddings_3d = embeddings_3d / norms
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=token_indices,
        cmap='viridis',
        alpha=0.8
    )
    
    # Add some token labels
    for i in range(min(10, num_vis_tokens)):
        token_idx = token_indices[i]
        token_char = chr(max(32, min(126, token_idx)))
        ax.text(
            embeddings_3d[i, 0],
            embeddings_3d[i, 1],
            embeddings_3d[i, 2],
            token_char,
            size=12
        )
    
    # Draw wireframe sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x, y, z, color='r', alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Token Embeddings on the Hypersphere (3D Projection)')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add explanation
    plt.figtext(
        0.5, 0.01,
        "nGPT normalizes all token embeddings to lie on the hypersphere.\n"
        "This visualization shows a 3D projection of the high-dimensional embeddings.",
        ha='center',
        fontsize=12
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Token Index')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

def visualize_attention_cosine(model, save_path=None):
    """
    Visualize how cosine similarity affects attention
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate sample vectors
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Plot unit circle
    ax1.plot(x, y, 'k-', alpha=0.3)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title('Vectors on the Hypersphere (2D)')
    
    # Plot some example vectors
    vectors = [
        (1, 0, 'Query'),
        (0.866, 0.5, 'Key 1 (30°)'),
        (0.5, 0.866, 'Key 2 (60°)'),
        (0, 1, 'Key 3 (90°)'),
        (-0.5, 0.866, 'Key 4 (120°)')
    ]
    
    colors = ['r', 'g', 'b', 'm', 'c']
    
    for i, (x, y, label) in enumerate(vectors):
        ax1.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc=colors[i], ec=colors[i])
        ax1.text(x*1.1, y*1.1, label, color=colors[i])
    
    # Plot attention weights based on cosine similarity
    query = np.array([1, 0])
    keys = np.array([
        [1, 0],
        [0.866, 0.5],
        [0.5, 0.866],
        [0, 1],
        [-0.5, 0.866]
    ])
    
    # Calculate cosine similarities
    similarities = np.dot(keys, query)
    
    # Apply softmax
    exp_sim = np.exp(similarities)
    attention = exp_sim / np.sum(exp_sim)
    
    # Plot attention weights
    key_labels = ['Query', 'Key 1', 'Key 2', 'Key 3', 'Key 4']
    bars = ax2.bar(key_labels, attention, color=colors)
    ax2.set_title('Attention Weights (Softmax of Cosine Similarities)')
    ax2.set_ylabel('Attention Weight')
    ax2.set_ylim(0, 1)
    
    # Add values on top of bars
    for bar, value in zip(bars, attention):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{value:.3f}',
            ha='center',
            fontsize=9
        )
    
    # Add explanation
    plt.figtext(
        0.5, 0.01,
        "nGPT uses cosine similarity for attention. As the angle between vectors increases,\n"
        "the attention weight decreases. This is why vectors on the hypersphere are effective for attention.",
        ha='center',
        fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize nGPT embeddings on the hypersphere")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--depth", type=int, default=2, help="Model depth")
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens")
    parser.add_argument("--save", type=str, help="Save path for visualizations")
    parser.add_argument("--attention", action="store_true", help="Visualize attention mechanism")
    parser.add_argument("--interactive", action="store_true", help="Show interactive 3D plot (requires display)")
    
    args = parser.parse_args()
    
    # Create a model
    print(f"Creating model with dim={args.dim}, depth={args.depth}, tokens={args.tokens}")
    try:
        model = nGPT(
            num_tokens=args.tokens,
            dim=args.dim,
            depth=args.depth,
            attn_norm_qk=True
        )
        
        # Visualize
        if args.attention:
            save_path = f"{args.save}_attention.png" if args.save else None
            visualize_attention_cosine(model, save_path)
        else:
            save_path = f"{args.save}_embeddings.png" if args.save else None
            visualize_embeddings_3d(model, args.tokens, save_path)
            
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    main() 