import numpy as np
import matplotlib.pyplot as plt
import random

def generate_sandplot(text, width=100, height=100, density=0.3, output_file='sandplot.png'):
    """
    Generate a sandplot visualization from input text.
    
    Parameters:
    - text: Input text to visualize
    - width: Width of the output image
    - height: Height of the output image
    - density: Density of points (0-1)
    - output_file: Name of output file
    """
    # Create a white background
    plt.figure(figsize=(10, 10))
    plt.rcParams['axes.facecolor'] = 'white'

    # Convert text to numerical values for color generation
    ascii_values = [ord(c) for c in text]
    
    # Calculate number of points based on density
    num_points = int(width * height * density)
    
    # Generate random positions
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, height, num_points)
    
    # Generate colors based on text
    colors = []
    for i in range(num_points):
        # Use characters to influence color channels
        char_idx = i % len(ascii_values)
        r = (ascii_values[char_idx] * 13 % 100) / 100
        g = (ascii_values[char_idx] * 17 % 100) / 100
        b = (ascii_values[char_idx] * 19 % 100) / 100
        
        # Add slight random variation
        r = min(1, max(0, r + random.uniform(-0.1, 0.1)))
        g = min(1, max(0, g + random.uniform(-0.1, 0.1)))
        b = min(1, max(0, b + random.uniform(-0.1, 0.1)))
        
        colors.append([r, g, b])

    # Plot points with small size and low alpha for sand-like effect
    plt.scatter(x, y, c=colors, s=1, alpha=0.5)
    
    # Remove axes for clean look
    plt.axis('off')
    
    # Set equal aspect ratio
    plt.gca().set_aspect('equal')
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage
text = """
The quick brown fox jumps over the lazy dog.
This text will be converted into a beautiful sandplot visualization.
Each character influences the colors and patterns in the final image.
"""

generate_sandplot(
    text,
    width=100,
    height=100,
    density=0.5,
    output_file='sandplot_output.png'
)