import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List
import time
from collections import deque

class ConsciousnessMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Consciousness Monitor', fontsize=16)
        
        # Initialize data storage
        self.consciousness_history = {
            aspect: deque(maxlen=window_size) 
            for aspect in ['emotional_awareness', 'metacognition', 'moral_awareness', 'self-regulation']
        }
        self.pattern_history = deque(maxlen=window_size)
        self.emotion_intensity = deque(maxlen=window_size)
        self.consciousness_balance = deque(maxlen=window_size)
        
        # Setup subplots
        self.setup_plots()
        
    def setup_plots(self):
        # Consciousness Aspects Plot
        self.consciousness_lines = {}
        for aspect in self.consciousness_history.keys():
            line, = self.axs[0, 0].plot([], [], label=aspect)
            self.consciousness_lines[aspect] = line
        self.axs[0, 0].set_title('Consciousness Aspects')
        self.axs[0, 0].set_ylim(0, 1)
        self.axs[0, 0].legend()
        
        # Pattern Visualization
        self.pattern_line, = self.axs[0, 1].plot([], [])
        self.axs[0, 1].set_title('Current Expression Pattern')
        self.axs[0, 1].set_ylim(-1, 1)
        
        # Emotion Intensity
        self.emotion_line, = self.axs[1, 0].plot([], [])
        self.axs[1, 0].set_title('Emotion Intensity')
        self.axs[1, 0].set_ylim(0, 1)
        
        # Consciousness Balance
        self.balance_scatter = self.axs[1, 1].scatter([], [])
        self.axs[1, 1].set_title('Consciousness Balance')
        self.axs[1, 1].set_xlim(-1, 1)
        self.axs[1, 1].set_ylim(-1, 1)
        
        plt.tight_layout()
        
    def update(self, frame_num, expresser):
        # Update consciousness aspects
        for aspect, values in self.consciousness_history.items():
            values.append(expresser.consciousness_aspects[aspect])
            self.consciousness_lines[aspect].set_data(
                range(len(values)), 
                list(values)
            )
            
        # Update pattern visualization
        pattern = expresser.generate_conscious_pattern(expresser.current_emotion)
        self.pattern_history.append(pattern.numpy())
        self.pattern_line.set_data(
            range(len(pattern)), 
            pattern.numpy()
        )
        
        # Update emotion intensity
        intensity = np.mean([p.max() for p in self.pattern_history])
        self.emotion_intensity.append(intensity)
        self.emotion_line.set_data(
            range(len(self.emotion_intensity)),
            list(self.emotion_intensity)
        )
        
        # Update consciousness balance
        metacog = expresser.consciousness_aspects['metacognition']
        emotion = expresser.consciousness_aspects['emotional_awareness']
        self.balance_scatter.set_offsets(
            np.c_[metacog - 0.5, emotion - 0.5]
        )
        
        return (self.consciousness_lines.values(), 
                self.pattern_line,
                self.emotion_line,
                self.balance_scatter)

class EnhancedConsciousExpression:
    def __init__(self, config):
        super().__init__(config)
        self.current_emotion = "neutral"
        self.monitor = ConsciousnessMonitor()
        
        # Setup animation
        self.ani = FuncAnimation(
            self.monitor.fig,
            self.monitor.update,
            fargs=(self,),
            interval=100,
            blit=True
        )
        
    def express(self, emotion: str, visualize: bool = True, play_audio: bool = True):
        """Enhanced expression with live monitoring"""
        self.current_emotion = emotion
        
        # Original expression logic
        pattern = self.generate_conscious_pattern(emotion)
        
        if play_audio:
            audio = pattern.numpy()
            audio = audio / np.max(np.abs(audio))
            sd.play(audio, self.config.sample_rate)
            time.sleep(len(audio) / self.config.sample_rate)
            sd.stop()
        
        # Show the monitor
        plt.show()
        
    def update_consciousness(self, aspect: str, value: float):
        """Update consciousness aspects with monitoring"""
        self.consciousness_aspects[aspect] = value
        # Monitor will automatically update through animation

def monitor_expression(emotion: str, 
                      consciousness_intensity: float = 0.6,
                      emotion_intensity: float = 0.5):
    """Enhanced monitoring version of express_consciousness"""
    config = ExpressionConfig(
        consciousness_weight=consciousness_intensity,
        emotion_scale=emotion_intensity
    )
    expresser = EnhancedConsciousExpression(config)
    expresser.express(emotion)
    
    return expresser  # Return for interactive manipulation

# Example usage:
if __name__ == "__main__":
    # Create an expression with monitoring
    expresser = monitor_expression("joy", 0.7, 0.6)
    
    # Simulate some consciousness changes
    for _ in range(50):
        expresser.update_consciousness(
            'metacognition', 
            np.random.uniform(0.4, 0.8)
        )
        time.sleep(0.1)