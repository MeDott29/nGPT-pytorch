import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class GeneratorConfig:
    """Configuration for waveform generation parameters"""
    sequence_length: int = 128
    num_features: int = 16
    noise_scale: float = 0.1
    sample_rate: int = 44100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class WaveformGenerator:
    """Generate various waveform patterns with controllable parameters"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def generate_sine_wave(self, 
                          freq: float = 440.0, 
                          amplitude: float = 1.0,
                          phase: float = 0.0) -> torch.Tensor:
        """Generate a sine wave with given frequency and amplitude
        
        Args:
            freq: Frequency in Hz
            amplitude: Wave amplitude (0.0 to 1.0)
            phase: Phase offset in radians
        
        Returns:
            torch.Tensor: Generated waveform
        """
        t = torch.linspace(0, 
                          self.config.sequence_length / self.config.sample_rate,
                          self.config.sequence_length,
                          device=self.device)
        wave = amplitude * torch.sin(2 * np.pi * freq * t + phase)
        return self._add_noise(wave)
    
    def generate_harmonic_series(self, 
                               base_freq: float = 440.0,
                               num_harmonics: int = 3) -> torch.Tensor:
        """Generate a harmonic series based on a fundamental frequency
        
        Args:
            base_freq: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to include
        
        Returns:
            torch.Tensor: Generated harmonic series
        """
        wave = torch.zeros(self.config.sequence_length, device=self.device)
        for i in range(num_harmonics):
            # Each harmonic's amplitude decreases as 1/n
            harmonic = self.generate_sine_wave(
                freq=base_freq * (i + 1),
                amplitude=1.0 / (i + 1)
            )
            wave += harmonic
        
        # Normalize to preserve amplitude
        wave = wave / num_harmonics
        return self._add_noise(wave)
    
    def generate_fm_synthesis(self,
                            carrier_freq: float = 440.0,
                            modulator_freq: float = 110.0,
                            modulation_index: float = 2.0) -> torch.Tensor:
        """Generate FM synthesis waveform
        
        Args:
            carrier_freq: Carrier frequency in Hz
            modulator_freq: Modulator frequency in Hz
            modulation_index: Modulation index (affects modulation depth)
        
        Returns:
            torch.Tensor: Generated FM waveform
        """
        t = torch.linspace(0, 
                          self.config.sequence_length / self.config.sample_rate,
                          self.config.sequence_length,
                          device=self.device)
        
        # Calculate modulation
        modulation = modulation_index * torch.sin(2 * np.pi * modulator_freq * t)
        wave = torch.sin(2 * np.pi * carrier_freq * t + modulation)
        return self._add_noise(wave)
    
    def generate_multi_channel(self) -> torch.Tensor:
        """Generate multi-channel data with different patterns
        
        Returns:
            torch.Tensor: Generated multi-channel data [sequence_length, num_features]
        """
        data = torch.zeros(
            (self.config.sequence_length, self.config.num_features),
            device=self.device
        )
        
        # Fill different channels with various patterns
        for i in range(self.config.num_features):
            # Vary frequency based on channel index
            freq = 220.0 * (1 + i/4)
            
            if i % 3 == 0:
                data[:, i] = self.generate_sine_wave(freq=freq)
            elif i % 3 == 1:
                data[:, i] = self.generate_harmonic_series(base_freq=freq)
            else:
                data[:, i] = self.generate_fm_synthesis(carrier_freq=freq)
        
        return data
    
    def _add_noise(self, wave: torch.Tensor) -> torch.Tensor:
        """Add controlled amount of noise to the signal"""
        if self.config.noise_scale > 0:
            noise = torch.randn_like(wave) * self.config.noise_scale
            wave = wave + noise
        return wave.clamp(-1, 1)  # Ensure signal stays in valid range

    def get_waveform_stats(self, wave: torch.Tensor) -> Dict:
        """Calculate basic statistics for the waveform
        
        Args:
            wave: Input waveform tensor
        
        Returns:
            Dict containing statistics
        """
        return {
            'mean': wave.mean().item(),
            'std': wave.std().item(),
            'min': wave.min().item(),
            'max': wave.max().item(),
            'peak_to_peak': (wave.max() - wave.min()).item()
        }

def demo_generator():
    """Demonstrate usage of the WaveformGenerator"""
    config = GeneratorConfig(
        sequence_length=1024,
        num_features=16,
        noise_scale=0.05
    )
    
    generator = WaveformGenerator(config)
    
    # Generate some example waveforms
    sine = generator.generate_sine_wave(freq=440)
    harmonics = generator.generate_harmonic_series(base_freq=220)
    fm = generator.generate_fm_synthesis(carrier_freq=440)
    multi = generator.generate_multi_channel()
    
    # Print some statistics
    print("\nSine Wave Stats:")
    print(generator.get_waveform_stats(sine))
    
    print("\nHarmonic Series Stats:")
    print(generator.get_waveform_stats(harmonics))
    
    print("\nFM Synthesis Stats:")
    print(generator.get_waveform_stats(fm))
    
    print("\nMulti-channel Data Shape:", multi.shape)
    
    return sine, harmonics, fm, multi

if __name__ == "__main__":
    demo_generator()