import math
import gzip
import random
import tqdm
import numpy as np
import pygame
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
from enum import Enum
from itertools import cycle

import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F

from nGPT_pytorch import nGPT
from token_linear import TokenLinearGL, TokenLinearSM

# Audio feedback system
class TrainingEvent(Enum):
    LOSS_DECREASE = "loss_decrease"
    LOSS_SPIKE = "loss_spike"
    MODEL_COMPARISON = "model_comparison"
    GENERATION_QUALITY = "generation_quality"
    GRADIENT_HEALTH = "gradient_health"

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    base_freq: float = 220.0  # A3
    duration: float = 0.3
    master_volume: float = 0.4

class TrainingAudioFeedback:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=self.config.sample_rate, size=-16, channels=2)
            except pygame.error:
                print("Warning: Could not initialize audio. Training will continue without audio feedback.")
                self.audio_enabled = False
                return
        
        self.audio_enabled = True
        self.training_state = {
            'model_a_loss': float('inf'),
            'model_b_loss': float('inf'),
            'loss_history_a': [],
            'loss_history_b': [],
            'gradient_norm_a': 0.0,
            'gradient_norm_b': 0.0
        }
        
        self._create_sound_buffers()
        
        # Background sound thread
        self.background_thread = None
        self.should_play_background = False
        
    def _create_sound_buffers(self):
        """Create sound buffers for different training events"""
        self.sound_buffers = {}
        
        # Loss decrease sound (descending pleasant tone)
        def create_loss_decrease_sound(magnitude: float) -> np.ndarray:
            duration = self.config.duration
            t = np.linspace(0, duration, int(self.config.sample_rate * duration))
            freq = self.config.base_freq * (2 + magnitude)
            freq_mod = np.linspace(1.2, 0.8, len(t))
            
            wave = np.sin(2 * np.pi * freq * freq_mod * t)
            wave *= self._create_envelope(len(wave), 0.1, 0.2)
            return wave
        
        # Loss spike warning (ascending harsh tone)
        def create_loss_spike_sound(magnitude: float) -> np.ndarray:
            duration = self.config.duration
            t = np.linspace(0, duration, int(self.config.sample_rate * duration))
            base_freq = self.config.base_freq * (1 + magnitude)
            
            wave = np.sin(2 * np.pi * base_freq * t) * 0.5
            wave += np.sin(2 * np.pi * base_freq * 2 * t) * 0.3  # Add harmonic
            wave *= self._create_envelope(len(wave), 0.05, 0.1)
            return wave
        
        # Model comparison sound (stereo comparison tone)
        def create_comparison_sound(diff: float) -> np.ndarray:
            duration = self.config.duration * 1.5
            t = np.linspace(0, duration, int(self.config.sample_rate * duration))
            
            # Create different tones for each model
            freq_a = self.config.base_freq * (1.5 + diff)
            freq_b = self.config.base_freq * (1.5 - diff)
            
            wave_a = np.sin(2 * np.pi * freq_a * t)
            wave_b = np.sin(2 * np.pi * freq_b * t)
            
            # Create stereo effect
            stereo = np.vstack((wave_a, wave_b))
            envelope = self._create_envelope(len(t), 0.1, 0.3)
            return stereo * envelope
        
        # Generation quality feedback (complex harmonic tone)
        def create_generation_sound(quality: float) -> np.ndarray:
            duration = self.config.duration * 2
            t = np.linspace(0, duration, int(self.config.sample_rate * duration))
            
            wave = np.zeros_like(t)
            harmonics = [1, 1.5, 2, 2.5, 3]
            
            for i, harmonic in enumerate(harmonics):
                amplitude = 0.7 ** i * quality
                wave += amplitude * np.sin(2 * np.pi * self.config.base_freq * harmonic * t)
            
            wave *= self._create_envelope(len(wave), 0.15, 0.3)
            return wave
        
        # Gradient health indicator (pulsing tone)
        def create_gradient_sound(health: float) -> np.ndarray:
            duration = self.config.duration
            t = np.linspace(0, duration, int(self.config.sample_rate * duration))
            
            # Modulate frequency based on gradient health
            freq_mod = 1 + 0.1 * np.sin(2 * np.pi * 8 * health * t)
            wave = np.sin(2 * np.pi * self.config.base_freq * freq_mod * t)
            
            wave *= self._create_envelope(len(wave), 0.05, 0.15)
            return wave
        
        # Create variations for each sound type
        magnitudes = np.linspace(0, 1, 5)
        
        self.sound_buffers = {
            TrainingEvent.LOSS_DECREASE: {
                mag: pygame.mixer.Sound((create_loss_decrease_sound(mag) * 32767).astype(np.int16))
                for mag in magnitudes
            },
            TrainingEvent.LOSS_SPIKE: {
                mag: pygame.mixer.Sound((create_loss_spike_sound(mag) * 32767).astype(np.int16))
                for mag in magnitudes
            },
            TrainingEvent.MODEL_COMPARISON: {
                mag: pygame.mixer.Sound((create_comparison_sound(mag) * 32767).astype(np.int16))
                for mag in magnitudes
            },
            TrainingEvent.GENERATION_QUALITY: {
                mag: pygame.mixer.Sound((create_generation_sound(mag) * 32767).astype(np.int16))
                for mag in magnitudes
            },
            TrainingEvent.GRADIENT_HEALTH: {
                mag: pygame.mixer.Sound((create_gradient_sound(mag) * 32767).astype(np.int16))
                for mag in magnitudes
            }
        }
    
    def _create_envelope(self, length: int, attack_time: float, decay_time: float) -> np.ndarray:
        """Create an ADSR envelope"""
        attack = int(length * attack_time)
        decay = int(length * decay_time)
        
        envelope = np.ones(length)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)
        return envelope
    
    def update_training_state(self, loss_a: float, loss_b: float, 
                            grad_norm_a: Optional[float] = None,
                            grad_norm_b: Optional[float] = None):
        """Update the training state and trigger appropriate audio feedback"""
        
        # Track loss history
        self.training_state['loss_history_a'].append(loss_a)
        self.training_state['loss_history_b'].append(loss_b)
        
        # Keep history length manageable
        max_history = 100
        if len(self.training_state['loss_history_a']) > max_history:
            self.training_state['loss_history_a'] = self.training_state['loss_history_a'][-max_history:]
            self.training_state['loss_history_b'] = self.training_state['loss_history_b'][-max_history:]
        
        # Check for significant loss decrease
        if loss_a < self.training_state['model_a_loss'] * 0.95:
            self.trigger_event(TrainingEvent.LOSS_DECREASE, magnitude=0.7)
        elif loss_b < self.training_state['model_b_loss'] * 0.95:
            self.trigger_event(TrainingEvent.LOSS_DECREASE, magnitude=0.5)
            
        # Check for loss spikes
        if loss_a > self.training_state['model_a_loss'] * 1.5:
            self.trigger_event(TrainingEvent.LOSS_SPIKE, magnitude=0.8)
        elif loss_b > self.training_state['model_b_loss'] * 1.5:
            self.trigger_event(TrainingEvent.LOSS_SPIKE, magnitude=0.6)
        
        # Compare models
        loss_diff = abs(loss_a - loss_b)
        if loss_diff > 0.1:
            self.trigger_event(TrainingEvent.MODEL_COMPARISON, 
                             magnitude=min(1.0, loss_diff / 2.0))
        
        # Update gradient health if provided
        if grad_norm_a is not None and grad_norm_b is not None:
            self.training_state['gradient_norm_a'] = grad_norm_a
            self.training_state['gradient_norm_b'] = grad_norm_b
            
            # Trigger gradient health sound if gradients are unusual
            if grad_norm_a > 10 or grad_norm_b > 10:
                self.trigger_event(TrainingEvent.GRADIENT_HEALTH, magnitude=0.7)
        
        # Update stored losses
        self.training_state['model_a_loss'] = loss_a
        self.training_state['model_b_loss'] = loss_b
    
    def trigger_event(self, event_type: TrainingEvent, magnitude: float = 0.5):
        """Trigger a specific audio event"""
        if event_type not in self.sound_buffers:
            return
            
        # Find closest magnitude
        magnitudes = list(self.sound_buffers[event_type].keys())
        closest_mag = min(magnitudes, key=lambda x: abs(x - magnitude))
        
        # Play sound
        sound = self.sound_buffers[event_type][closest_mag]
        sound.set_volume(self.config.master_volume)
        sound.play()
    
    def feedback_generation_quality(self, output_a: str, output_b: str):
        """Provide audio feedback on generation quality"""
        # Simple heuristic for generation quality (could be made more sophisticated)
        def assess_quality(text: str) -> float:
            # Check for repetition
            repetition_penalty = len(set(text)) / len(text)
            # Check for printable characters
            printable_ratio = sum(c.isprintable() for c in text) / len(text)
            return min(1.0, repetition_penalty * printable_ratio)
        
        quality_a = assess_quality(output_a)
        quality_b = assess_quality(output_b)
        
        # Trigger sounds for both models
        self.trigger_event(TrainingEvent.GENERATION_QUALITY, magnitude=quality_a)
        time.sleep(self.config.duration)  # Wait between sounds
        self.trigger_event(TrainingEvent.GENERATION_QUALITY, magnitude=quality_b)

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient norm for a model"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

class SyntheticDataset(Dataset):
    def __init__(self, seq_len, num_samples=1000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Generate synthetic token sequences
        print(f"Generating {num_samples} synthetic sequences...")
        self.data = []
        for _ in tqdm.tqdm(range(num_samples)):
            # Generate random token indices between 0 and 255
            x = torch.randint(0, 256, (seq_len,), dtype=torch.long)
            y = torch.randint(0, 256, (seq_len,), dtype=torch.long)  # Target sequence
            self.data.append((x, y))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def decode_tokens(tokens):
    """Convert token indices to ASCII characters"""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().detach().numpy()
    if isinstance(tokens, tuple):
        tokens = tokens[0]  # Get first element if tuple
    return ''.join([chr(int(t)) for t in tokens])

def base_decoding(model, prompt, max_length):
    """Basic greedy decoding for text generation"""
    generated = prompt.clone()
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_length - prompt.size(1)):
            output = model(generated)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated

def main():
    # Constants
    NUM_BATCHES = int(1e5)
    BATCH_SIZE = 4
    GRAD_ACCUM_EVERY = 4
    LEARNING_RATE = 1e-3
    VALIDATE_EVERY = 100
    PRIME_LENGTH = 128
    GENERATE_EVERY = 500
    GENERATE_LENGTH = 512
    SEQ_LEN = 512

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    model_a = nGPT(
        num_tokens=256,
        dim=512,
        depth=8,
        manual_norm_weights=True,
        tied_embedding=True,
        add_value_residual=True
    ).to(device)

    model_b = nGPT(
        num_tokens=256,
        dim=512,
        depth=8,
        manual_norm_weights=True,
        tied_embedding=True,
        add_value_residual=False
    ).to(device)

    # Load and prepare data
    try:
        with gzip.open("./data/enwik8.gz") as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            data_train = torch.from_numpy(np_train)
            data_val = torch.from_numpy(np_valid)
    except FileNotFoundError:
        print("Error: enwik8.gz file not found in ./data/ directory")
        return

    # Setup datasets and dataloaders
    train_dataset = SyntheticDataset(SEQ_LEN, num_samples=1000)
    val_dataset = SyntheticDataset(SEQ_LEN, num_samples=100)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup optimizers
    optim_a = Adam(model_a.parameters(), lr=LEARNING_RATE)
    optim_b = Adam(model_b.parameters(), lr=LEARNING_RATE)

    # Initialize audio feedback
    audio_feedback = TrainingAudioFeedback()

    # Create cyclic iterators
    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    try:
        # Training loop
        for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
            model_a.train()
            model_b.train()

            # Training step
            for _ in range(GRAD_ACCUM_EVERY):
                x, y = next(train_loader)
                x, y = x.to(device), y.to(device)

                # Model A forward and backward
                output_a = model_a(x)
                loss_a = F.cross_entropy(output_a.view(-1, output_a.size(-1)), y.view(-1))
                (loss_a / GRAD_ACCUM_EVERY).backward()

                # Model B forward and backward
                output_b = model_b(x)
                loss_b = F.cross_entropy(output_b.view(-1, output_b.size(-1)), y.view(-1))
                (loss_b / GRAD_ACCUM_EVERY).backward()

            # Compute gradient norms
            grad_norm_a = compute_gradient_norm(model_a)
            grad_norm_b = compute_gradient_norm(model_b)

            # Update audio feedback
            audio_feedback.update_training_state(
                loss_a=loss_a.item(),
                loss_b=loss_b.item(),
                grad_norm_a=grad_norm_a,
                grad_norm_b=grad_norm_b
            )

            print(f"\ntraining loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

            # Update models
            optim_a.step()
            optim_b.step()
            optim_a.zero_grad()
            optim_b.zero_grad()

            model_a.norm_weights_()
            model_b.norm_weights_()

            # Validation
            if i % VALIDATE_EVERY == 0:
                model_a.eval()
                model_b.eval()
                with torch.no_grad():
                    x, y = next(val_loader)
                    x, y = x.to(device), y.to(device)
                    
                    output_a = model_a(x)
                    loss_a = F.cross_entropy(output_a.view(-1, output_a.size(-1)), y.view(-1))
                    
                    output_b = model_b(x)
                    loss_b = F.cross_entropy(output_b.view(-1, output_b.size(-1)), y.view(-1))
                    
                    audio_feedback.update_training_state(
                        loss_a=loss_a.item(),
                        loss_b=loss_b.item()
                    )
                    
                    print(f"\nvalidation loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

            # Generation
            if i % GENERATE_EVERY == 0:
                model_a.eval()
                model_b.eval()
                with torch.no_grad():
                    inp = random.choice(val_dataset)
                    x, _ = inp  # Unpack the tuple to get input sequence
                    prime = decode_tokens(x)
                    print(f"\nPrime text: {prime}\n\n{'*' * 100}")
                    prompt = x[None, ...].to(device)  # Add batch dimension

                    print("\nGeneration from Model A (with value residual):")
                    sampled_a = base_decoding(model_a, prompt, GENERATE_LENGTH)
                    generated_text_a = decode_tokens(sampled_a[0])
                    print(generated_text_a)

                    print("\nGeneration from Model B (without value residual):")
                    sampled_b = base_decoding(model_b, prompt, GENERATE_LENGTH)
                    generated_text_b = decode_tokens(sampled_b[0])
                    print(generated_text_b)

                    # Provide audio feedback on generation quality
                    audio_feedback.feedback_generation_quality(
                        output_a=generated_text_a,
                        output_b=generated_text_b
                    )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        # Save models
        print("\nSaving models...")
        torch.save(model_a.state_dict(), 'model_a_with_value_residual.pt')
        torch.save(model_b.state_dict(), 'model_b_without_value_residual.pt')
        print("Models saved successfully.")

if __name__ == "__main__":
    main()