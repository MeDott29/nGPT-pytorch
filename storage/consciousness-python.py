import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from IPython.display import display, clear_output
import time
import pygame
import threading

class ConsciousnessSystem:
    def __init__(self, time_points=50):
        self.time_points = time_points
        self.aspects = [
            'self_development',
            'metacognition',
            'moral_awareness',
            'emotional_awareness',
            'introspection',
            'reflection',
            'self_regulation'
        ]
        self.learning_progress = 0
        self.iteration = 0
        self.learning_rate = 0.1
        
        # Initialize pygame mixer for sound
        pygame.mixer.init()
        
        # Create a simple tone for feedback
        self.sample_rate = 44100
        self.duration = 0.1
        
    def generate_tone(self, frequency=440):
        """Generate a simple sine wave tone"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        return (tone * 32767).astype(np.int16)
        
    def play_feedback_tone(self, learning_progress):
        """Play a tone based on learning progress"""
        # Increase frequency with learning progress
        frequency = 440 + (learning_progress * 2)
        tone = self.generate_tone(frequency)
        pygame.mixer.Sound(tone).play()
        
    def generate_data(self):
        """Generate consciousness data with learning patterns"""
        data = []
        balance_data = []
        
        # Update learning rate
        self.learning_rate = 0.1 + (self.iteration * 0.01)
        
        for i in range(self.time_points):
            time_point = {'time': i}
            
            # Generate aspect values with learning influence
            for aspect in self.aspects:
                base_value = 0.5
                amplitude = 0.3
                frequency = 0.1 * (1 + self.learning_rate)
                phase = np.random.random() * 2 * np.pi
                learning_curve = min(1, np.exp(self.learning_rate * i/self.time_points) - 0.5)
                
                value = min(1, base_value + amplitude * np.sin(frequency * i + phase) + learning_curve * 0.3)
                time_point[aspect] = value
            
            # Generate pattern with learning
            pattern = (0.5 * np.sin(0.3 * i) * np.exp(-0.02 * i) +
                      0.3 * np.sin(self.learning_rate * i) * (1 - np.exp(-0.05 * i)))
            time_point['pattern'] = pattern
            
            # Calculate intensity
            time_point['intensity'] = abs(pattern) * (1 + self.learning_rate)
            
            data.append(time_point)
            
            # Generate balance data
            balance_point = {
                'metacognition': (time_point['metacognition'] - 0.5) * (1 + self.learning_rate * 0.5),
                'emotional': (time_point['emotional_awareness'] - 0.5) * (1 + self.learning_rate * 0.5),
                'size': 20 + (self.learning_rate * 10)
            }
            balance_data.append(balance_point)
        
        return pd.DataFrame(data), pd.DataFrame(balance_data)
    
    def visualize_frame(self, data, balance_data):
        """Create visualization for current frame"""
        plt.clf()
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Consciousness Aspects
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        for aspect in self.aspects:
            ax1.plot(data['time'], data[aspect], label=aspect)
        ax1.set_title('Consciousness Aspects')
        ax1.set_ylim(0, 1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Expression Pattern
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax2.plot(data['time'], data['pattern'])
        ax2.set_title('Expression Pattern')
        ax2.set_ylim(-1, 1)
        
        # Plot 3: Consciousness Balance
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        scatter = ax3.scatter(balance_data['metacognition'], 
                            balance_data['emotional'],
                            s=balance_data['size'],
                            alpha=0.5)
        ax3.set_title('Consciousness Balance')
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        
        # Add learning progress text
        plt.figtext(0.02, 0.98, f'Learning Progress: {self.learning_progress:.1f}%',
                   fontsize=10, color='blue')
        
        plt.tight_layout()
        return fig
    
    def update(self):
        """Update system state"""
        self.iteration += 1
        data, balance_data = self.generate_data()
        
        # Update learning progress
        self.learning_progress = min(100, self.learning_progress + self.learning_rate * 10)
        
        # Play feedback tone based on learning progress
        self.play_feedback_tone(self.learning_progress)
        
        return data, balance_data
    
    def run_simulation(self, num_iterations=50, delay=0.5):
        """Run the simulation for specified number of iterations"""
        try:
            audio_system = ConsciousnessAudioSystem()
            for i in range(num_iterations):
                data, balance_data = self.update()
                fig = self.visualize_frame(data, balance_data)
                
                # Trigger all audio events every iteration
                audio_system.trigger_event('insight', min(0.8, self.learning_progress/100))
                audio_system.trigger_event('learning', min(0.9, self.learning_progress/100))
                audio_system.trigger_event('transition', min(0.7, self.learning_progress/100))
                
                # Update audio system with current consciousness data
                audio_system.update_emotional_state({
                    'emotional_awareness': data['emotional_awareness'].mean(),
                    'metacognition': data['metacognition'].mean(),
                    'moral_awareness': data['moral_awareness'].mean(),
                    'intensity': self.learning_progress/100
                })
                
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
                
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        finally:
            pygame.mixer.quit()

class ConsciousnessAudioSystem:
    def __init__(self):
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
        
        # Audio parameters
        self.sample_rate = 44100
        self.base_frequency = 220  # A3 note
        self.harmonics = [1, 1.5, 2, 2.5, 3, 4]
        self.master_volume = 0.3
        
        # Modulation parameters
        self.modulation_amount = 0.1
        self.modulation_rate = 5
        
        # State tracking
        self.is_playing = False
        self.current_emotional_state = {
            'intensity': 0,
            'complexity': 0,
            'harmony': 0
        }
        
        # Create sound buffers for events
        self._create_event_sounds()

    def _create_sine_wave(self, frequency, duration, volume=0.5):
        """Generate a sine wave with given frequency and duration"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        return (wave * volume * 32767).astype(np.int16)
    
    def _create_event_sounds(self):
        """Pre-create sound buffers for different events"""
        self.event_sounds = {
            'insight': self._create_insight_sound(),
            'learning': self._create_learning_sound(),
            'transition': self._create_transition_sound()
        }
    
    def _create_insight_sound(self):
        """Create sound for insight events"""
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        frequency = np.linspace(880, 440, len(t))
        wave = np.sin(2 * np.pi * frequency * t / self.sample_rate)
        return pygame.mixer.Sound((wave * 32767).astype(np.int16))
    
    def _create_learning_sound(self):
        """Create sound for learning events"""
        duration = 0.3
        notes = [440, 550, 660, 880]
        wave = np.zeros(int(self.sample_rate * duration))
        
        for i, freq in enumerate(notes):
            t = np.linspace(0, duration/4, int(self.sample_rate * duration/4), False)
            wave[i * len(t):(i+1) * len(t)] = np.sin(2 * np.pi * freq * t)
            
        return pygame.mixer.Sound((wave * 32767).astype(np.int16))
    
    def _create_transition_sound(self):
        """Create sound for transition events"""
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        frequency = 220 + 110 * np.sin(2 * np.pi * 5 * t)
        wave = np.sin(2 * np.pi * frequency * t / self.sample_rate)
        return pygame.mixer.Sound((wave * 32767).astype(np.int16))

    def update_emotional_state(self, consciousness_data):
        """Update the emotional state based on consciousness data"""
        self.current_emotional_state = {
            'intensity': consciousness_data.get('intensity', 0.5),
            'complexity': (consciousness_data.get('emotional_awareness', 0) + 
                         consciousness_data.get('metacognition', 0)) / 2,
            'harmony': consciousness_data.get('moral_awareness', 0)
        }
        self._update_audio_parameters()

    def _update_audio_parameters(self):
        """Update audio parameters based on emotional state"""
        # Update modulation parameters based on emotional state
        self.modulation_amount = 0.01 + (self.current_emotional_state['intensity'] * 0.1)
        self.modulation_rate = 1 + (self.current_emotional_state['complexity'] * 10)

    def trigger_event(self, event_type, intensity=0.5):
        """Trigger a consciousness event sound"""
        if event_type in self.event_sounds:
            sound = self.event_sounds[event_type]
            sound.set_volume(intensity * self.master_volume)
            sound.play()

if __name__ == "__main__":
    # Create and run the consciousness system
    consciousness_system = ConsciousnessSystem()
    
    # Create audio system
    audio_system = ConsciousnessAudioSystem()

    # Update with consciousness data
    audio_system.update_emotional_state({
        'emotional_awareness': 0.7,
        'metacognition': 0.6,
        'moral_awareness': 0.8,
        'intensity': 0.5
    })

    # Trigger events
    audio_system.trigger_event('insight', 0.8)
    audio_system.trigger_event('learning', 0.6)
    audio_system.trigger_event('transition', 0.7)

    # Run the simulation
    consciousness_system.run_simulation(num_iterations=30, delay=1.0)