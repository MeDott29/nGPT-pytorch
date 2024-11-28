"""
Consciousness Simulation System
=============================

This module implements an educational simulation of consciousness and emotional expression
using various aspects of awareness and real-time visualization. It demonstrates key concepts in:

1. Real-time data visualization
2. Signal processing and pattern generation
3. System state management
4. Interactive animations
5. Audio synthesis

The system models different aspects of consciousness (like metacognition, emotional awareness, etc.)
and shows how they interact to create patterns of expression.

Key Components:
- ConsciousnessConfig: Configuration settings for the simulation
- ConsciousSystem: Main system implementing the consciousness simulation
- Visualization: Real-time plotting using matplotlib
- Pattern Generation: Creating waveforms for different emotional states
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import sounddevice as sd
import time
import os

@dataclass
class ConsciousnessConfig:
    """
    Configuration class for the consciousness simulation system.
    
    Attributes:
        duration (int): Length of generated patterns in samples
        emotion_scale (float): Scaling factor for emotional expression intensity
        consciousness_weight (float): Weight of consciousness influence on patterns
        sample_rate (int): Audio sample rate for pattern generation
        window_size (int): Number of historical states to maintain
        update_interval (int): Visualization update interval in milliseconds
        output_dir (str): Directory for saving expression visualizations
        learning_rate (float): Learning rate for consciousness aspect adjustments
        adaptation_rate (float): Rate of adaptation for consciousness aspects
        audio_duration (float): Duration for audio playback in seconds
    """
    duration: int = 128
    emotion_scale: float = 0.5
    consciousness_weight: float = 0.6
    sample_rate: int = 44100
    window_size: int = 100
    update_interval: int = 100  # milliseconds
    output_dir: str = "expressions"
    learning_rate: float = 0.01
    adaptation_rate: float = 0.05
    audio_duration: float = 3.0  # Duration for audio playback in seconds

class ConsciousSystem:
    """
    A system simulating aspects of consciousness and emotional expression.
    
    This class demonstrates the integration of multiple consciousness aspects
    and their influence on emotional expression patterns. It includes real-time
    visualization and audio synthesis capabilities.
    """
    
    def __init__(self, config: ConsciousnessConfig):
        """
        Initialize the consciousness system.
        
        Args:
            config: Configuration object containing system parameters
        """
        self.config = config
        self.current_emotion = "neutral"
        self.is_expressing = False
        
        # Initialize data structures first
        self.setup_consciousness()
        
        # Pre-fill history buffers to ensure smooth visualization startup
        self._initialize_history()
        
        # Setup visualization system
        self.setup_visualization()
        os.makedirs(config.output_dir, exist_ok=True)
        
    def setup_consciousness(self):
        """
        Initialize the core aspects of consciousness and their relationships.
        
        This method defines the key components of the consciousness system:
        - self-development: Growth and learning capability
        - metacognition: Awareness of one's own thought processes
        - moral_awareness: Understanding of ethical implications
        - emotional_awareness: Recognition of emotional states
        - introspection: Self-examination capability
        - reflection: Ability to contemplate experiences
        - self-regulation: Control over responses and behaviors
        """
        # Initialize consciousness aspects with baseline values
        self.consciousness_aspects = {
            'self-development': 0.9,  # High initial drive for growth
            'metacognition': 0.8,     # Strong awareness of thought processes
            'moral_awareness': 0.7,   # Good ethical understanding
            'emotional_awareness': 0.6,# Moderate emotional recognition
            'introspection': 0.7,     # Good self-examination capability
            'reflection': 0.6,        # Moderate contemplation ability
            'self-regulation': 0.75   # Good behavioral control
        }
        
        # Initialize history tracking using deque for efficient fixed-size history
        self.history = {
            'aspects': {aspect: deque(maxlen=self.config.window_size) 
                       for aspect in self.consciousness_aspects},
            'patterns': deque(maxlen=self.config.window_size),
            'intensity': deque(maxlen=self.config.window_size),
            'balance': deque(maxlen=self.config.window_size)
        }
        
        # Add learning history
        self.learning_history = {
            'performance': deque(maxlen=self.config.window_size),
            'adaptations': deque(maxlen=self.config.window_size)
        }
        
        # Initialize learning weights
        self.learning_weights = {
            aspect: np.random.uniform(0.4, 0.6) 
            for aspect in self.consciousness_aspects
        }

    def _initialize_history(self):
        """
        Pre-fill history buffers with initial values for smooth visualization startup.
        This prevents visual artifacts during the first few frames of animation.
        """
        for aspect in self.consciousness_aspects:
            self.history['aspects'][aspect].extend([self.consciousness_aspects[aspect]] * 10)
        self.history['patterns'].extend([np.zeros(self.config.duration)] * 10)
        self.history['intensity'].extend([0.0] * 10)
        self.history['balance'].extend([0.5] * 10)


    def setup_visualization(self):
        """Setup the monitoring visualization"""
        plt.style.use('fast')
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(3, 3)
        
        # Consciousness Aspects Plot
        self.ax_aspects = self.fig.add_subplot(self.gs[0, :2])
        self.aspect_lines = {}
        for aspect in self.consciousness_aspects:
            # Initialize with actual data
            line, = self.ax_aspects.plot(
                range(len(self.history['aspects'][aspect])),
                list(self.history['aspects'][aspect]),
                label=aspect
            )
            self.aspect_lines[aspect] = line
        self.ax_aspects.set_title('Consciousness Aspects')
        self.ax_aspects.set_ylim(0, 1)
        self.ax_aspects.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Current Pattern
        self.ax_pattern = self.fig.add_subplot(self.gs[1, :2])
        pattern = self.generate_pattern(self.current_emotion)
        self.pattern_line, = self.ax_pattern.plot(
            range(len(pattern)), 
            pattern.numpy()
        )
        self.ax_pattern.set_title('Expression Pattern')
        self.ax_pattern.set_ylim(-1, 1)

        # Consciousness Balance
        self.ax_balance = self.fig.add_subplot(self.gs[0, 2])
        metacog = self.consciousness_aspects['metacognition']
        emotion = self.consciousness_aspects['emotional_awareness']
        self.balance_scatter = self.ax_balance.scatter(
            [metacog - 0.5], 
            [emotion - 0.5]
        )
        self.ax_balance.set_title('Consciousness Balance')
        self.ax_balance.set_xlim(-1, 1)
        self.ax_balance.set_ylim(-1, 1)

        # Emotion Intensity
        self.ax_intensity = self.fig.add_subplot(self.gs[1, 2])
        self.intensity_line, = self.ax_intensity.plot(
            range(len(self.history['intensity'])),
            list(self.history['intensity'])
        )
        self.ax_intensity.set_title('Emotional Intensity')
        self.ax_intensity.set_ylim(0, 1)

        # System State
        self.ax_state = self.fig.add_subplot(self.gs[2, :])
        self.state_text = self.ax_state.text(
            0.5, 0.5, 
            self.get_state_text(),
            ha='center', va='center',
            fontsize=12
        )
        self.ax_state.axis('off')

        plt.tight_layout()
        
        # Initialize last_artists before creating animation
        self.last_artists = (*self.aspect_lines.values(), 
                self.pattern_line,
                self.intensity_line,
                self.balance_scatter,
                self.state_text)
        
        # Initialize animation with frame counter
        self.frame_count = 0
        self.ani = FuncAnimation(
            self.fig, 
            self.update_visualization,
            interval=self.config.update_interval,
            blit=True,
            cache_frame_data=False,
            save_count=50
        )

    def get_state_text(self):
        """Generate current state text"""
        intensity = np.mean(list(self.history['intensity']))
        return (f"Current Emotion: {self.current_emotion}\n"
                f"System Balance: {self.calculate_system_balance():.2f}\n"
                f"Expression Intensity: {intensity:.2f}")

    def generate_pattern(self, emotion: str) -> torch.Tensor:
        """
        Generate an expression pattern based on emotion and consciousness state.
        
        This method demonstrates how different emotional states can be represented
        through varying waveform patterns, modulated by consciousness aspects.
        
        Args:
            emotion: The emotional state to express ("joy", "peace", "contemplation", "neutral")
            
        Returns:
            torch.Tensor: Generated pattern waveform
            
        Teaching Point:
            Notice how each emotion uses different frequency compositions and
            is influenced by relevant consciousness aspects. For example,
            joy uses higher frequencies modulated by emotional_awareness,
            while contemplation uses lower frequencies affected by metacognition.
        """
        t = torch.linspace(0, 4*np.pi, self.config.duration)
        
        # Map emotions to their generation functions
        patterns = {
            "joy": self._generate_joy,
            "peace": self._generate_peace,
            "contemplation": self._generate_contemplation,
            "neutral": self._generate_neutral
        }
        
        if emotion not in patterns:
            raise ValueError(f"Unknown emotion: {emotion}")
            
        pattern = patterns[emotion](t)
        return self._apply_consciousness_modulation(pattern)

    def _generate_joy(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate a pattern expressing joy using higher frequencies and emotional awareness.
        
        Teaching Point:
            Joy is expressed through:
            1. Higher base frequency (2x)
            2. Additional harmonic at double frequency
            3. Modulation by emotional_awareness and self-regulation
        """
        freq = 2 * (1 + self.consciousness_aspects['emotional_awareness'])
        pattern = torch.sin(freq * t)  # Base frequency
        # Add harmonic modulated by self-regulation
        pattern += 0.5 * torch.sin(2 * freq * t) * self.consciousness_aspects['self-regulation']
        return pattern * self.config.emotion_scale

    def _generate_peace(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate a pattern expressing peace using gentle decay and moral awareness.
        
        Teaching Point:
            Peace is expressed through:
            1. Lower base frequency
            2. Exponential decay for gentler expression
            3. Subtle harmonic influenced by moral_awareness
        """
        freq = 1 * (1 + self.consciousness_aspects['introspection'])
        pattern = torch.sin(freq * t) * torch.exp(-0.1 * t)  # Decaying sinusoid
        pattern += 0.3 * torch.sin(freq/2 * t) * self.consciousness_aspects['moral_awareness']
        return pattern * self.config.emotion_scale

    def _generate_contemplation(self, t: torch.Tensor) -> torch.Tensor:
        freq = 0.5 * (1 + self.consciousness_aspects['metacognition'])
        pattern = torch.sin(freq * t)
        pattern *= torch.exp(-0.05 * t) * self.consciousness_aspects['reflection']
        return pattern * self.config.emotion_scale

    def _generate_neutral(self, t: torch.Tensor) -> torch.Tensor:
        freq = 1.0
        return torch.sin(freq * t) * 0.3 * self.config.emotion_scale

    def _apply_consciousness_modulation(self, pattern: torch.Tensor) -> torch.Tensor:
        consciousness_factor = sum(self.consciousness_aspects.values()) / len(self.consciousness_aspects)
        return pattern * (1 + self.config.consciousness_weight * consciousness_factor)

    def update_visualization(self, frame):
        """Update all visualization components"""
        if frame % 2 == 0:
            return self.last_artists
        
        self.frame_count += 1
        
        # Generate new pattern using torch.no_grad()
        with torch.no_grad():
            pattern = self.generate_pattern(self.current_emotion)
        
        self.history['patterns'].append(pattern.numpy())

        # Calculate new intensity
        intensity = np.abs(pattern.numpy()).mean()
        self.history['intensity'].append(intensity)

        # Update all aspects
        for aspect in self.consciousness_aspects:
            # Add small random fluctuation to make it more interesting
            fluctuation = np.random.normal(0, 0.01)
            new_value = np.clip(
                self.consciousness_aspects[aspect] + fluctuation,
                0, 1
            )
            self.consciousness_aspects[aspect] = new_value
            self.history['aspects'][aspect].append(new_value)
            
            # Update aspect lines
            self.aspect_lines[aspect].set_data(
                range(len(self.history['aspects'][aspect])),
                list(self.history['aspects'][aspect])
            )

        # Update pattern visualization
        self.pattern_line.set_data(range(len(pattern)), pattern.numpy())
        # Update intensity line
        self.intensity_line.set_data(
            range(len(self.history['intensity'])),
            list(self.history['intensity'])
        )

        # Update emotion intensity
        intensity = np.abs(pattern.numpy()).mean()
        self.history['intensity'].append(intensity)
        self.intensity_line.set_data(
            range(len(self.history['intensity'])),
            list(self.history['intensity'])
        )

        # Update consciousness balance
        metacog = self.consciousness_aspects['metacognition']
        emotion = self.consciousness_aspects['emotional_awareness']
        self.balance_scatter.set_offsets(
            np.c_[metacog - 0.5, emotion - 0.5]
        )

        # Update state text
        self.state_text.set_text(self.get_state_text())

        # Adjust axis limits if needed
        self.ax_aspects.set_xlim(0, len(self.history['aspects'][aspect]))
        self.ax_intensity.set_xlim(0, len(self.history['intensity']))

        return self.last_artists



    def calculate_system_balance(self) -> float:
        """
        Calculate the overall system balance between cognitive and emotional aspects.
        
        Teaching Point:
            System balance is measured by:
            1. Averaging cognitive aspects (metacognition, introspection, reflection)
            2. Averaging emotional aspects (emotional_awareness, moral_awareness)
            3. Computing how close these averages are to each other
            
        Returns:
            float: Balance score between 0 (unbalanced) and 1 (perfectly balanced)
        """
        cognitive = np.mean([
            self.consciousness_aspects['metacognition'],
            self.consciousness_aspects['introspection'],
            self.consciousness_aspects['reflection']
        ])
        emotional = np.mean([
            self.consciousness_aspects['emotional_awareness'],
            self.consciousness_aspects['moral_awareness']
        ])
        return 1 - abs(cognitive - emotional)

    def express(self, emotion: str, duration: float = 5.0):
        """
        Express an emotion through visualization and audio synthesis.
        
        This method demonstrates the integration of:
        1. Pattern generation
        2. Audio synthesis
        3. Real-time visualization
        4. State management
        
        Args:
            emotion: Emotion to express
            duration: Duration of expression in seconds
        """
        self.current_emotion = emotion
        self.is_expressing = True
        
        # Generate and normalize audio pattern
        pattern = self.generate_pattern(emotion)
        audio = pattern.numpy()
        audio = audio / np.max(np.abs(audio))
        
        # Play audio in a non-blocking way
        sd.play(audio, self.config.sample_rate)
        
        # Save visualization
        plt.savefig(os.path.join(self.config.output_dir, f"{emotion}_expression.png"))
        plt.show()
        
        time.sleep(duration)
        sd.stop()
        self.is_expressing = False

    def update_consciousness(self, aspect: str, value: float):
        """Update a consciousness aspect"""
        if aspect not in self.consciousness_aspects:
            raise ValueError(f"Unknown aspect: {aspect}")
        self.consciousness_aspects[aspect] = max(0, min(1, value))

    def learn_from_expression(self, emotion: str, feedback: float):
        """
        Update consciousness aspects based on expression feedback.
        
        Args:
            emotion: The emotion that was expressed
            feedback: Performance feedback (-1 to 1)
        """
        # Update learning weights
        for aspect in self.consciousness_aspects:
            adjustment = self.config.learning_rate * feedback
            self.learning_weights[aspect] = np.clip(
                self.learning_weights[aspect] + adjustment,
                0.1, 1.0
            )
            
            # Adapt consciousness aspects
            new_value = self.consciousness_aspects[aspect] * (
                1 + adjustment * self.learning_weights[aspect]
            )
            self.consciousness_aspects[aspect] = np.clip(new_value, 0, 1)
        
        # Record learning progress
        self.learning_history['performance'].append(feedback)
        self.learning_history['adaptations'].append(
            np.mean(list(self.learning_weights.values()))
        )

def demonstrate_consciousness():
    """Demonstrate the consciousness system with learning and audio"""
    config = ConsciousnessConfig()
    system = ConsciousSystem(config)
    
    # Keep the visualization running
    plt.show()
    
    # Express different emotions with learning
    emotions = ["joy", "peace", "contemplation"]
    for emotion in emotions:
        print(f"\nExpressing {emotion}...")
        system.current_emotion = emotion
        
        # Simulate feedback and learning
        feedback = np.random.uniform(-0.5, 1.0)  # Simulated feedback
        system.learn_from_expression(emotion, feedback)
        
        # Generate and play audio pattern
        pattern = system.generate_pattern(emotion)
        audio = pattern.numpy()
        audio = audio / np.max(np.abs(audio))
        
        # Play audio for the specified duration
        sd.play(audio, system.config.sample_rate)
        time.sleep(config.audio_duration)
        sd.stop()
        
        # Update consciousness aspects
        system.update_consciousness('metacognition', np.random.uniform(0.4, 0.8))
        system.update_consciousness('emotional_awareness', np.random.uniform(0.4, 0.8))
        time.sleep(1.0)

    print("\nDemonstration complete!")

if __name__ == "__main__":
    plt.ion()
    demonstrate_consciousness()