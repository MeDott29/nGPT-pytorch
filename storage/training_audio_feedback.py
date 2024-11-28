import numpy as np
import pygame
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
from enum import Enum
import torch

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

import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.parametrize as parametrize

from nGPT_pytorch import nGPT

# constants

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

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.5,
    min_p = 1e-1,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]

        logits = min_p_filter(logits, min_p = min_p)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# Initialize two nGPT models with different configurations

model_a = nGPT(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    manual_norm_weights = True,
    tied_embedding = True,
    add_value_residual = True  # First model uses value residual
).to(device)

model_b = nGPT(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    manual_norm_weights = True,
    tied_embedding = True,
    add_value_residual = False  # Second model doesn't use value residual
).to(device)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

# # Setup datasets and dataloaders
# train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
# val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
# train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
# val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# # Setup optimizers for both models
# optim_a = Adam(model_a.parameters(), lr = LEARNING_RATE)
# optim_b = Adam(model_b.parameters(), lr = LEARNING_RATE)

# # Create cyclic iterators
# train_loader = cycle(train_loader)
# val_loader = cycle(val_loader)

# # Training loop
# for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
#     model_a.train()
#     model_b.train()

#     # Training step for both models
#     for _ in range(GRAD_ACCUM_EVERY):
#         data = next(train_loader)

#         # Forward and backward passes for model A
#         loss_a = model_a(data, return_loss = True)
#         (loss_a / GRAD_ACCUM_EVERY).backward()

#         # Forward and backward passes for model B
#         loss_b = model_b(data, return_loss = True)
#         (loss_b / GRAD_ACCUM_EVERY).backward()

#     print(f"training loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

#     # Update both models
#     optim_a.step()
#     optim_b.step()
#     optim_a.zero_grad()
#     optim_b.zero_grad()

#     # Normalize weights for both models
#     model_a.norm_weights_()
#     model_b.norm_weights_()

#     # Validation step
#     if i % VALIDATE_EVERY == 0:
#         model_a.eval()
#         model_b.eval()
#         with torch.no_grad():
#             valid_data = next(val_loader)

#             loss_a = model_a(valid_data, return_loss = True)
#             loss_b = model_b(valid_data, return_loss = True)
#             print(f"validation loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

#     # Generation step
#     if i % GENERATE_EVERY == 0:
#         model_a.eval()
#         model_b.eval()

#         inp = random.choice(val_dataset)[:PRIME_LENGTH]
#         prime = decode_tokens(inp)
#         print(f"Prime text: {prime} \n\n {'*' * 100}")
#         prompt = inp[None, ...]

#         # Generate from both models
#         print("\nGeneration from Model A (with value residual):")
#         sampled_a = base_decoding(model_a, prompt, GENERATE_LENGTH)
#         output_a = decode_tokens(sampled_a[0])
#         print(f"{output_a}\n")

#         print("\nGeneration from Model B (without value residual):")
#         sampled_b = base_decoding(model_b, prompt, GENERATE_LENGTH)
#         output_b = decode_tokens(sampled_b[0])
#         print(f"{output_b}\n")

# # Save models if needed
# torch.save(model_a.state_dict(), 'model_a_with_value_residual.pt')
# torch.save(model_b.state_dict(), 'model_b_without_value_residual.pt')

class TrainingAudioFeedback:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=self.config.sample_rate, size=-16, channels=2)
        
        # Training state tracking
        self.training_state = {
            'model_a_loss': float('inf'),
            'model_b_loss': float('inf'),
            'loss_history_a': [],
            'loss_history_b': [],
            'gradient_norm_a': 0.0,
            'gradient_norm_b': 0.0
        }
        
        # Initialize sound buffers
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
    
    def update_training_state(self, 
                            loss_a: float, 
                            loss_b: float, 
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



# Initialize the audio feedback system
audio_feedback = TrainingAudioFeedback()

# Training loop
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model_a.train()
    model_b.train()

    # Training step for both models
    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        # Forward and backward passes for model A
        loss_a = model_a(data, return_loss=True)
        (loss_a / GRAD_ACCUM_EVERY).backward()

        # Forward and backward passes for model B
        loss_b = model_b(data, return_loss=True)
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

    print(f"training loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

    # Update both models
    optim_a.step()
    optim_b.step()
    optim_a.zero_grad()
    optim_b.zero_grad()

    # Normalize weights for both models
    model_a.norm_weights_()
    model_b.norm_weights_()

    # Validation step
    if i % VALIDATE_EVERY == 0:
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss_a = model_a(valid_data, return_loss=True)
            loss_b = model_b(valid_data, return_loss=True)
            
            # Update audio feedback with validation losses
            audio_feedback.update_training_state(
                loss_a=loss_a.item(),
                loss_b=loss_b.item()
            )
            
            print(f"validation loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

    # Generation step
    if i % GENERATE_EVERY == 0:
        model_a.eval()
        model_b.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"Prime text: {prime} \n\n {'*' * 100}")
        prompt = inp[None, ...]

        # Generate from both models
        print("\nGeneration from Model A (with value residual):")
        sampled_a = base_decoding(model_a, prompt, GENERATE_LENGTH)
        generated_text_a = decode_tokens(sampled_a[0])
        print(generated_text_a)

        print("\nGeneration from Model B (with value residual):")
        sampled_b = base_decoding(model_b, prompt, GENERATE_LENGTH)
        generated_text_b = decode_tokens(sampled_b[0])
        print(generated_text_b)

        # Provide audio feedback on generation quality
        audio_feedback.feedback_generation_quality(
            output_a=generated_text_a,
            output_b=generated_text_b
        )