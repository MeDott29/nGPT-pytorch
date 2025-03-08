"""
nGPT Explorer - Interactive Learning Suite

This script creates an interactive visualization of the nGPT model,
allowing you to explore and understand how it works using an Xbox controller.

Controls:
- Left Stick: Navigate through the hypersphere (2D projection)
- Right Stick: Adjust model parameters (temperature, attention heads)
- A Button: Generate text from current position
- B Button: Reset view
- X Button: Toggle visualization mode
- Y Button: Save current state
- Left Bumper: Decrease model complexity
- Right Bumper: Increase model complexity
- Start: Exit the application
"""

import os
import sys
import time
import math
import numpy as np
import torch
import pygame
from pygame.locals import *
import matplotlib
matplotlib.use("Agg")  # Use Agg backend to avoid threading issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Import nGPT
from nGPT_pytorch import nGPT

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Controller deadzone
DEADZONE = 0.2

# Default model parameters
MODEL_DIM = 64
MODEL_DEPTH = 2
MODEL_HEADS = 4
NUM_TOKENS = 256
ATTN_NORM_QK = True
USE_CONTROLLER = True

# Try to load configuration if it exists
try:
    if os.path.exists("explorer_config.py"):
        print("Loading configuration from explorer_config.py")
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from explorer_config import *
        print(f"Loaded configuration: dim={MODEL_DIM}, depth={MODEL_DEPTH}, heads={MODEL_HEADS}")
except Exception as e:
    print(f"Error loading configuration: {e}")
    print("Using default parameters")

def figure_to_surface(fig):
    """Convert a matplotlib figure to a pygame surface"""
    # Draw the figure to the canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    
    # Convert ARGB to RGBA for pygame
    buf = np.roll(buf, 3, axis=2)
    
    # Create a pygame surface
    surface = pygame.image.frombuffer(buf, (w, h), "RGBA")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return surface

class NGPTExplorer:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("nGPT Explorer")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.large_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Initialize controller
        pygame.joystick.init()
        self.controller = None
        if USE_CONTROLLER:
            self.init_controller()
        else:
            print("Controller support disabled. Using keyboard controls.")
            print("WASD: Navigate, Arrow keys: Adjust parameters")
            print("Space: Generate text, R: Reset view, Tab: Toggle visualization mode")
        
        # Model parameters
        self.model_dim = MODEL_DIM
        self.model_depth = MODEL_DEPTH
        self.model_heads = MODEL_HEADS
        self.temperature = 1.0
        self.num_tokens = NUM_TOKENS
        
        # Visualization parameters
        self.viz_mode = 0  # 0: Attention, 1: Embeddings, 2: Activations
        self.viz_modes = ["Attention Visualization", "Token Embeddings", "Layer Activations"]
        self.position = [0.0, 0.0]  # Position on the hypersphere (2D projection)
        self.zoom = 1.0
        self.rotation = 0.0
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model()
        
        # Sample text for generation
        self.sample_text = "The quick brown fox jumps over the lazy dog."
        self.tokens = self.text_to_tokens(self.sample_text)
        self.generated_text = ""
        
        # Visualization surfaces
        self.viz_surface = None
        self.info_surface = None
        
        try:
            self.update_visualization()
        except Exception as e:
            print(f"Error initializing visualization: {e}")
            # Create a fallback visualization
            self.viz_surface = pygame.Surface((600, SCREEN_HEIGHT))
            self.viz_surface.fill(BLACK)
            text = self.large_font.render("Error initializing visualization", True, WHITE)
            self.viz_surface.blit(text, (100, SCREEN_HEIGHT // 2 - 20))
            error_text = self.font.render(str(e), True, RED)
            self.viz_surface.blit(error_text, (100, SCREEN_HEIGHT // 2 + 20))
            self.update_info_surface()
        
        # State
        self.running = True
    
    def init_controller(self):
        """Initialize the Xbox controller if available"""
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"Controller initialized: {self.controller.get_name()}")
        else:
            print("No controller found. Using keyboard controls.")
            print("WASD: Navigate, Arrow keys: Adjust parameters")
            print("Space: Generate text, R: Reset view, Tab: Toggle visualization mode")
    
    def init_model(self):
        """Initialize the nGPT model with current parameters"""
        print(f"Initializing model with dim={self.model_dim}, depth={self.model_depth}, heads={self.model_heads}")
        try:
            self.model = nGPT(
                num_tokens=self.num_tokens,
                dim=self.model_dim,
                depth=self.model_depth,
                heads=self.model_heads,
                attn_norm_qk=ATTN_NORM_QK
            ).to(self.device)
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.model = None
    
    def text_to_tokens(self, text):
        """Convert text to token IDs (simple ASCII encoding for demo)"""
        return torch.tensor([[ord(c) % self.num_tokens for c in text]], device=self.device)
    
    def tokens_to_text(self, tokens):
        """Convert token IDs back to text"""
        return "".join([chr(max(32, min(126, t))) for t in tokens.cpu().numpy().flatten()])
    
    def generate_text(self):
        """Generate text from the current model state"""
        if self.model is None:
            return "Model not initialized"
        
        try:
            with torch.no_grad():
                # Use the current position to influence the generation
                # by adjusting the temperature and sampling
                pos_tensor = torch.tensor([self.position], device=self.device)
                
                # Generate text
                input_ids = self.tokens
                max_length = 50
                
                # Simple greedy generation
                for _ in range(max_length):
                    logits = self.model(input_ids)
                    next_token_logits = logits[:, -1, :]
                    
                    # Apply position influence (as a form of control)
                    # Scale logits based on position
                    scale_x = 1.0 + 0.5 * self.position[0]
                    scale_y = 1.0 + 0.5 * self.position[1]
                    temperature = self.temperature * (scale_x + scale_y) / 2
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / max(0.1, temperature)
                    
                    # Get the most likely token
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Stop if we generate a newline
                    if next_token.item() == 10:
                        break
                
                generated_text = self.tokens_to_text(input_ids[0, self.tokens.shape[1]:])
                return generated_text
        except Exception as e:
            return f"Error generating text: {e}"
    
    def get_attention_visualization(self):
        """Create a visualization of the attention mechanism"""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Create a simple attention visualization
        if self.model is not None:
            try:
                with torch.no_grad():
                    # Get attention weights from the first layer
                    input_ids = self.tokens
                    
                    # Forward pass to get attention weights
                    # This is a simplified version - in a real implementation,
                    # you would need to modify the model to return attention weights
                    
                    # For now, we'll create a simulated attention map
                    seq_len = input_ids.shape[1]
                    attn_map = np.zeros((seq_len, seq_len))
                    
                    # Create a pattern influenced by the controller position
                    for i in range(seq_len):
                        for j in range(seq_len):
                            # Distance-based attention with position influence
                            dist = abs(i - j)
                            attn_map[i, j] = math.exp(-dist / (5 + 5 * self.position[0]))
                    
                    # Normalize
                    attn_map = attn_map / attn_map.sum(axis=1, keepdims=True)
                    
                    # Plot the attention map
                    im = ax.imshow(attn_map, cmap='viridis')
                    ax.set_title(f"Attention Map (Position: {self.position[0]:.2f}, {self.position[1]:.2f})")
                    ax.set_xlabel("Key Position")
                    ax.set_ylabel("Query Position")
                    fig.colorbar(im)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Model not initialized", ha='center', va='center', transform=ax.transAxes)
        
        # Convert to pygame surface using our helper function
        return figure_to_surface(fig)
    
    def get_embedding_visualization(self):
        """Create a visualization of token embeddings on the hypersphere"""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        if self.model is not None:
            try:
                with torch.no_grad():
                    # Get token embeddings
                    token_embed = self.model.token_embed.weight
                    
                    # Use PCA to project to 2D for visualization
                    # For simplicity, we'll just use the first two dimensions
                    # In a real implementation, you would use PCA or t-SNE
                    
                    # Get a subset of tokens for visualization
                    num_vis_tokens = min(100, self.num_tokens)
                    token_indices = torch.linspace(0, self.num_tokens-1, num_vis_tokens).long()
                    
                    # Get embeddings for these tokens
                    embeddings = token_embed[token_indices]
                    
                    # Project to 2D (simplified)
                    # In a real implementation, use proper dimensionality reduction
                    x = embeddings[:, 0].cpu().numpy()
                    y = embeddings[:, 1].cpu().numpy()
                    
                    # Apply controller position as rotation and zoom
                    theta = self.rotation + self.position[0] * math.pi
                    zoom = self.zoom * (1 + 0.5 * self.position[1])
                    
                    # Apply rotation and zoom
                    x_rot = x * math.cos(theta) - y * math.sin(theta)
                    y_rot = x * math.sin(theta) + y * math.cos(theta)
                    
                    x = x_rot * zoom
                    y = y_rot * zoom
                    
                    # Plot points
                    ax.scatter(x, y, alpha=0.7)
                    ax.set_title(f"Token Embeddings (Rotation: {theta:.2f}, Zoom: {zoom:.2f})")
                    
                    # Draw a unit circle to represent the hypersphere
                    circle = plt.Circle((0, 0), 1, fill=False, color='r')
                    ax.add_patch(circle)
                    
                    # Add some token labels
                    for i in range(min(10, num_vis_tokens)):
                        token_idx = token_indices[i].item()
                        token_char = chr(max(32, min(126, token_idx)))
                        ax.annotate(token_char, (x[i], y[i]))
                    
                    ax.set_xlim(-1.5, 1.5)
                    ax.set_ylim(-1.5, 1.5)
                    ax.set_aspect('equal')
                    ax.grid(True)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Model not initialized", ha='center', va='center', transform=ax.transAxes)
        
        # Convert to pygame surface using our helper function
        return figure_to_surface(fig)
    
    def get_activations_visualization(self):
        """Create a visualization of layer activations"""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        if self.model is not None:
            try:
                with torch.no_grad():
                    # Forward pass with the sample text
                    input_ids = self.tokens
                    
                    # This is a simplified visualization
                    # In a real implementation, you would need to modify the model
                    # to return intermediate activations
                    
                    # For now, we'll create a simulated activation pattern
                    seq_len = input_ids.shape[1]
                    layer_activations = np.zeros((self.model_depth, seq_len))
                    
                    # Create a pattern influenced by the controller position
                    for i in range(self.model_depth):
                        for j in range(seq_len):
                            # Position and layer-dependent activation
                            layer_activations[i, j] = 0.5 + 0.5 * math.sin(
                                j / 5 + i * math.pi / 4 + self.position[0] * math.pi
                            ) * math.cos(
                                j / 3 + self.position[1] * math.pi
                            )
                    
                    # Plot the activations
                    im = ax.imshow(layer_activations, cmap='plasma', aspect='auto')
                    ax.set_title(f"Layer Activations (Position: {self.position[0]:.2f}, {self.position[1]:.2f})")
                    ax.set_xlabel("Sequence Position")
                    ax.set_ylabel("Layer")
                    ax.set_yticks(range(self.model_depth))
                    fig.colorbar(im)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Model not initialized", ha='center', va='center', transform=ax.transAxes)
        
        # Convert to pygame surface using our helper function
        return figure_to_surface(fig)
    
    def update_visualization(self):
        """Update the visualization based on current mode"""
        try:
            if self.viz_mode == 0:
                self.viz_surface = self.get_attention_visualization()
            elif self.viz_mode == 1:
                self.viz_surface = self.get_embedding_visualization()
            else:
                self.viz_surface = self.get_activations_visualization()
            
            # Update info surface
            self.update_info_surface()
        except Exception as e:
            print(f"Error updating visualization: {e}")
            # Create a fallback visualization
            self.viz_surface = pygame.Surface((600, SCREEN_HEIGHT))
            self.viz_surface.fill(BLACK)
            text = self.large_font.render("Error updating visualization", True, WHITE)
            self.viz_surface.blit(text, (100, SCREEN_HEIGHT // 2 - 20))
            error_text = self.font.render(str(e), True, RED)
            self.viz_surface.blit(error_text, (100, SCREEN_HEIGHT // 2 + 20))
            self.update_info_surface()
    
    def update_info_surface(self):
        """Update the information panel"""
        info_surface = pygame.Surface((SCREEN_WIDTH - 600, SCREEN_HEIGHT))
        info_surface.fill(BLACK)
        
        # Draw title
        title = self.large_font.render("nGPT Explorer", True, WHITE)
        info_surface.blit(title, (20, 20))
        
        # Draw model parameters
        y = 70
        params = [
            f"Model Dimension: {self.model_dim}",
            f"Model Depth: {self.model_depth}",
            f"Attention Heads: {self.model_heads}",
            f"Temperature: {self.temperature:.2f}",
            f"Visualization Mode: {self.viz_modes[self.viz_mode]}",
            f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f})",
            f"Device: {self.device}"
        ]
        
        for param in params:
            text = self.font.render(param, True, WHITE)
            info_surface.blit(text, (20, y))
            y += 30
        
        # Draw controls
        y += 20
        controls_title = self.large_font.render("Controls:", True, YELLOW)
        info_surface.blit(controls_title, (20, y))
        y += 40
        
        if self.controller is not None:
            controls = [
                "Left Stick: Navigate hypersphere",
                "Right Stick: Adjust parameters",
                "A Button: Generate text",
                "B Button: Reset view",
                "X Button: Toggle visualization mode",
                "Y Button: Save current state",
                "LB/RB: Decrease/Increase complexity",
                "Start: Exit"
            ]
        else:
            controls = [
                "WASD: Navigate hypersphere",
                "Arrow Keys: Adjust parameters",
                "Space: Generate text",
                "R: Reset view",
                "Tab: Toggle visualization mode",
                "Ctrl+S: Save screenshot",
                "-/+: Decrease/Increase complexity",
                "Esc: Exit"
            ]
        
        for control in controls:
            text = self.font.render(control, True, WHITE)
            info_surface.blit(text, (20, y))
            y += 30
        
        # Draw generated text
        y += 20
        gen_title = self.large_font.render("Generated Text:", True, GREEN)
        info_surface.blit(gen_title, (20, y))
        y += 40
        
        # Wrap text to fit the panel
        max_width = SCREEN_WIDTH - 600 - 40
        words = self.generated_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            text_width = self.font.size(test_line)[0]
            
            if text_width < max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            text = self.font.render(line, True, WHITE)
            info_surface.blit(text, (20, y))
            y += 30
        
        self.info_surface = info_surface
    
    def handle_controller_input(self):
        """Handle input from the Xbox controller"""
        if self.controller is None:
            return
        
        try:
            # Left stick (navigation)
            left_x = self.controller.get_axis(0)
            left_y = self.controller.get_axis(1)
            
            # Apply deadzone
            if abs(left_x) < DEADZONE:
                left_x = 0
            if abs(left_y) < DEADZONE:
                left_y = 0
            
            # Update position
            self.position[0] += left_x * 0.05
            self.position[1] += left_y * 0.05
            
            # Clamp position to [-1, 1]
            self.position[0] = max(-1, min(1, self.position[0]))
            self.position[1] = max(-1, min(1, self.position[1]))
            
            # Right stick (parameter adjustment)
            right_x = self.controller.get_axis(3)
            right_y = self.controller.get_axis(4)
            
            # Apply deadzone
            if abs(right_x) < DEADZONE:
                right_x = 0
            if abs(right_y) < DEADZONE:
                right_y = 0
            
            # Adjust parameters
            self.temperature += right_y * 0.05
            self.temperature = max(0.1, min(5.0, self.temperature))
            
            self.rotation += right_x * 0.05
            
            # Buttons
            if self.controller.get_button(0):  # A Button
                self.generated_text = self.generate_text()
                time.sleep(0.2)  # Debounce
            
            if self.controller.get_button(1):  # B Button
                self.position = [0.0, 0.0]
                self.rotation = 0.0
                self.zoom = 1.0
                time.sleep(0.2)  # Debounce
            
            if self.controller.get_button(2):  # X Button
                self.viz_mode = (self.viz_mode + 1) % len(self.viz_modes)
                time.sleep(0.2)  # Debounce
            
            if self.controller.get_button(3):  # Y Button
                # Save current state (visualization)
                pygame.image.save(self.screen, f"ngpt_explorer_{time.strftime('%Y%m%d_%H%M%S')}.png")
                print("Screenshot saved")
                time.sleep(0.2)  # Debounce
            
            # Bumpers (model complexity)
            if self.controller.get_button(4):  # Left Bumper
                self.model_depth = max(1, self.model_depth - 1)
                self.init_model()
                time.sleep(0.2)  # Debounce
            
            if self.controller.get_button(5):  # Right Bumper
                self.model_depth += 1
                self.init_model()
                time.sleep(0.2)  # Debounce
            
            # Start button (exit)
            if self.controller.get_button(7):  # Start Button
                self.running = False
        
        except Exception as e:
            print(f"Controller error: {e}")
    
    def handle_keyboard_input(self):
        """Handle keyboard input as fallback"""
        keys = pygame.key.get_pressed()
        
        # WASD for navigation
        if keys[K_w]:
            self.position[1] -= 0.05
        if keys[K_s]:
            self.position[1] += 0.05
        if keys[K_a]:
            self.position[0] -= 0.05
        if keys[K_d]:
            self.position[0] += 0.05
        
        # Clamp position
        self.position[0] = max(-1, min(1, self.position[0]))
        self.position[1] = max(-1, min(1, self.position[1]))
        
        # Arrow keys for parameter adjustment
        if keys[K_UP]:
            self.temperature -= 0.05
        if keys[K_DOWN]:
            self.temperature += 0.05
        if keys[K_LEFT]:
            self.rotation -= 0.05
        if keys[K_RIGHT]:
            self.rotation += 0.05
        
        # Clamp temperature
        self.temperature = max(0.1, min(5.0, self.temperature))
        
        # Other controls
        if keys[K_SPACE]:
            self.generated_text = self.generate_text()
        if keys[K_r]:
            self.position = [0.0, 0.0]
            self.rotation = 0.0
            self.zoom = 1.0
        if keys[K_TAB]:
            self.viz_mode = (self.viz_mode + 1) % len(self.viz_modes)
        if keys[K_s] and (keys[K_LCTRL] or keys[K_RCTRL]):
            pygame.image.save(self.screen, f"ngpt_explorer_{time.strftime('%Y%m%d_%H%M%S')}.png")
            print("Screenshot saved")
        if keys[K_MINUS]:
            self.model_depth = max(1, self.model_depth - 1)
            self.init_model()
        if keys[K_EQUALS]:
            self.model_depth += 1
            self.init_model()
    
    def run(self):
        """Main loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
            
            # Handle input
            self.handle_controller_input()
            self.handle_keyboard_input()
            
            # Update visualization
            self.update_visualization()
            
            # Draw everything
            self.screen.fill(BLACK)
            
            # Draw visualization
            if self.viz_surface:
                self.screen.blit(self.viz_surface, (0, 0))
            
            # Draw info panel
            if self.info_surface:
                self.screen.blit(self.info_surface, (600, 0))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Clean up
        pygame.quit()

if __name__ == "__main__":
    explorer = NGPTExplorer()
    explorer.run() 