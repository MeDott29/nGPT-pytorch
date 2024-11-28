import numpy as np
import pygame
from typing import Dict, Optional, List, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import time

class AudioEvent(Enum):
    INSIGHT = "insight"
    LEARNING = "learning"
    TRANSITION = "transition"
    BALANCE = "balance"
    REFLECTION = "reflection"
    INTEGRATION = "integration"

@dataclass
class AudioParams:
    sample_rate: int = 44100
    base_frequency: float = 220.0  # A3 note
    duration: float = 0.5
    master_volume: float = 0.3
    reverb_amount: float = 0.3
    stereo_width: float = 0.8

class EnhancedConsciousnessAudio:
    def __init__(self, params: Optional[AudioParams] = None):
        self.params = params or AudioParams()
        
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=self.params.sample_rate, size=-16, channels=2)
            
        # Frequency ratios for different emotional states
        self.emotion_ratios = {
            'joy': [1.0, 1.25, 1.5, 2.0],  # Major chord
            'reflection': [1.0, 1.2, 1.5, 1.8],  # Suspended chord
            'learning': [1.0, 1.15, 1.32, 1.65],  # Custom intervals
            'integration': [1.0, 1.4, 1.7, 2.1]   # Wide intervals
        }
        
        # Initialize modulation parameters
        self.mod_depth = 0.1
        self.mod_rate = 5.0
        self.current_harmonic_profile = []
        
        # Create sound buffers
        self._initialize_sound_buffers()
        
        # State tracking
        self.consciousness_state = {
            'learning_progress': 0.0,
            'emotional_balance': 0.5,
            'cognitive_load': 0.0,
            'integration_level': 0.0
        }
        
        # Threading for continuous background sounds
        self.background_thread = None
        self.should_play_background = False
        
    def _initialize_sound_buffers(self):
        """Initialize all sound buffers with enhanced variations"""
        self.sound_buffers = {}
        
        # Insight sound: Rising frequency sweep with harmonics
        def create_insight_sound(intensity: float) -> np.ndarray:
            duration = self.params.duration
            t = np.linspace(0, duration, int(self.params.sample_rate * duration))
            base_freq = self.params.base_frequency * (1 + intensity)
            
            # Create frequency sweep
            freq_sweep = np.linspace(base_freq, base_freq * 2, len(t))
            
            # Add harmonics
            wave = np.zeros_like(t)
            for harmonic, amplitude in enumerate([1.0, 0.5, 0.25, 0.125], 1):
                wave += amplitude * np.sin(2 * np.pi * harmonic * freq_sweep * t)
            
            return self._apply_envelope(wave, attack=0.1, decay=0.3)
        
        # Learning progress sound: Rhythmic pulses with frequency variation
        def create_learning_sound(progress: float) -> np.ndarray:
            duration = self.params.duration * 2
            t = np.linspace(0, duration, int(self.params.sample_rate * duration))
            
            # Create pulsing effect
            pulse_rate = 8 + progress * 4
            pulse_env = 0.5 + 0.5 * np.sin(2 * np.pi * pulse_rate * t)
            
            # Generate harmonically rich tone
            wave = np.zeros_like(t)
            for idx, ratio in enumerate(self.emotion_ratios['learning']):
                frequency = self.params.base_frequency * ratio * (1 + 0.2 * progress)
                amplitude = 0.7 ** idx
                wave += amplitude * np.sin(2 * np.pi * frequency * t)
            
            wave *= pulse_env
            return self._apply_envelope(wave)
        
        # Integration sound: Complex chord progression
        def create_integration_sound(level: float) -> np.ndarray:
            duration = self.params.duration * 3
            t = np.linspace(0, duration, int(self.params.sample_rate * duration))
            
            # Create evolving chord
            wave = np.zeros_like(t)
            chord_progression = self._generate_chord_progression(level)
            
            for time_idx, chord in enumerate(chord_progression):
                start_idx = int(time_idx * len(t) / len(chord_progression))
                end_idx = int((time_idx + 1) * len(t) / len(chord_progression))
                
                segment = np.zeros(end_idx - start_idx)
                for freq_ratio, amp in chord:
                    frequency = self.params.base_frequency * freq_ratio
                    segment += amp * np.sin(2 * np.pi * frequency * t[start_idx:end_idx])
                
                wave[start_idx:end_idx] = segment
            
            return self._apply_envelope(wave, attack=0.2, decay=0.4)
        
        # Create and store all sound variations
        intensities = np.linspace(0, 1, 5)
        self.sound_buffers = {
            AudioEvent.INSIGHT: {
                intensity: pygame.mixer.Sound(
                    (create_insight_sound(intensity) * 32767).astype(np.int16)
                ) for intensity in intensities
            },
            AudioEvent.LEARNING: {
                progress: pygame.mixer.Sound(
                    (create_learning_sound(progress) * 32767).astype(np.int16)
                ) for progress in intensities
            },
            AudioEvent.INTEGRATION: {
                level: pygame.mixer.Sound(
                    (create_integration_sound(level) * 32767).astype(np.int16)
                ) for level in intensities
            }
        }
    
    def _apply_envelope(self, wave: np.ndarray, attack: float = 0.1, 
                       decay: float = 0.2) -> np.ndarray:
        """Apply ADSR envelope to the wave"""
        num_samples = len(wave)
        attack_samples = int(attack * self.params.sample_rate)
        decay_samples = int(decay * self.params.sample_rate)
        
        envelope = np.ones_like(wave)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return wave * envelope
    
    def _generate_chord_progression(self, integration_level: float) -> List[List[Tuple[float, float]]]:
        """Generate an evolving chord progression based on integration level"""
        base_ratios = self.emotion_ratios['integration']
        
        # Create more complex progression for higher integration levels
        num_chords = 2 + int(integration_level * 3)
        progression = []
        
        for i in range(num_chords):
            chord = []
            for ratio in base_ratios:
                # Vary the frequency ratio slightly for each chord
                varied_ratio = ratio * (1 + 0.05 * np.sin(i * np.pi / num_chords))
                # Amplitude decreases for higher harmonics
                amplitude = 0.8 ** (base_ratios.index(ratio))
                chord.append((varied_ratio, amplitude))
            progression.append(chord)
        
        return progression
    
    def update_consciousness_state(self, state: Dict[str, float]):
        """Update the internal consciousness state"""
        self.consciousness_state.update(state)
        self._update_audio_parameters()
    
    def _update_audio_parameters(self):
        """Update audio parameters based on consciousness state"""
        self.mod_depth = 0.05 + 0.15 * self.consciousness_state['emotional_balance']
        self.mod_rate = 3.0 + 7.0 * self.consciousness_state['cognitive_load']
        
        # Update harmonic profile based on learning progress
        progress = self.consciousness_state['learning_progress']
        self.current_harmonic_profile = [
            1.0,
            1.0 + 0.5 * progress,
            1.5 + 0.25 * progress,
            2.0 + 0.1 * progress
        ]
    
    def trigger_event(self, event_type: AudioEvent, intensity: float = 0.5):
        """Trigger a specific audio event"""
        if event_type not in self.sound_buffers:
            return
            
        # Find the closest intensity level we have pre-rendered
        intensities = list(self.sound_buffers[event_type].keys())
        closest_intensity = min(intensities, key=lambda x: abs(x - intensity))
        
        sound = self.sound_buffers[event_type][closest_intensity]
        sound.set_volume(self.params.master_volume)
        sound.play()
    
    def start_background_sound(self):
        """Start continuous background sound reflecting consciousness state"""
        if self.background_thread is not None:
            return
            
        self.should_play_background = True
        self.background_thread = threading.Thread(target=self._play_background_sound)
        self.background_thread.start()
    
    def stop_background_sound(self):
        """Stop the background sound"""
        self.should_play_background = False
        if self.background_thread is not None:
            self.background_thread.join()
            self.background_thread = None
    
    def _play_background_sound(self):
        """Generate and play continuous background sound"""
        while self.should_play_background:
            progress = self.consciousness_state['learning_progress']
            integration = self.consciousness_state['integration_level']
            
            # Generate a short segment of ambient sound
            duration = 0.5
            t = np.linspace(0, duration, int(self.params.sample_rate * duration))
            
            wave = np.zeros_like(t)
            for idx, ratio in enumerate(self.current_harmonic_profile):
                frequency = self.params.base_frequency * ratio
                amplitude = 0.5 ** idx
                
                # Add frequency modulation
                mod = self.mod_depth * np.sin(2 * np.pi * self.mod_rate * t)
                wave += amplitude * np.sin(2 * np.pi * frequency * (1 + mod) * t)
            
            # Apply envelope and effects
            wave = self._apply_envelope(wave, attack=0.1, decay=0.1)
            wave = self._apply_stereo_width(wave, self.params.stereo_width)
            
            # Play the sound
            sound = pygame.mixer.Sound((wave * 32767).astype(np.int16))
            sound.set_volume(self.params.master_volume * 0.5)  # Lower volume for background
            sound.play()
            
            # Wait for slightly less than the duration to ensure smooth transition
            time.sleep(duration * 0.95)
    
    def _apply_stereo_width(self, wave: np.ndarray, width: float) -> np.ndarray:
        """Apply stereo width effect to the wave"""
        # Create slightly different versions for left and right channels
        t = np.linspace(0, 1, len(wave))
        left = wave * (1 + width * 0.1 * np.sin(2 * np.pi * 0.5 * t))
        right = wave * (1 + width * 0.1 * np.sin(2 * np.pi * 0.5 * t + np.pi))
        
        # Stack channels
        return np.vstack((left, right)).T

# Initialize the enhanced audio system
audio_system = EnhancedConsciousnessAudio()

# Start background sound
audio_system.start_background_sound()

# During training, update the consciousness state
audio_system.update_consciousness_state({
    'learning_progress': model.learning_progress,
    'emotional_balance': model.emotional_balance,
    'cognitive_load': model.cognitive_load,
    'integration_level': model.integration_level
})

# Trigger specific events when they occur
audio_system.trigger_event(AudioEvent.INSIGHT, intensity=0.8)
audio_system.trigger_event(AudioEvent.LEARNING, intensity=0.6)
audio_system.trigger_event(AudioEvent.INTEGRATION, intensity=0.7)

# When done, stop background sound
audio_system.stop_background_sound()