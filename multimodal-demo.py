import streamlit as st
import plotly.graph_objects as go
import librosa
import numpy as np
from PIL import Image
import io
import torch
import torchaudio
from pathlib import Path

class MultimodalDemo:
    def __init__(self, data_dir='./output'):
        self.data_dir = Path(data_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_image(self, image_path):
        return Image.open(image_path)
    
    def load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        return waveform.numpy()[0], sr
    
    def plot_waveform(self, waveform, sr):
        time = np.linspace(0, len(waveform) / sr, len(waveform))
        fig = go.Figure(data=go.Scatter(x=time, y=waveform, mode='lines'))
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude"
        )
        return fig
    
    def plot_spectrogram(self, waveform, sr):
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(waveform)), ref=np.max
        )
        fig = go.Figure(data=go.Heatmap(z=D))
        fig.update_layout(
            title="Audio Spectrogram",
            xaxis_title="Time",
            yaxis_title="Frequency"
        )
        return fig

    def run_demo(self):
        st.title("Multimodal Conversion Demo")
        
        # Display original MNIST grid
        mnist_images = sorted(self.data_dir.glob('screenshots/*.jpg'))
        if mnist_images:
            st.header("Original MNIST Grid")
            img = self.load_image(mnist_images[0])
            st.image(img, caption="MNIST Digit Grid")
        
        # Display audio visualization
        audio_files = sorted(self.data_dir.glob('audio/*.wav'))
        if audio_files:
            st.header("Audio Generation")
            audio_path = audio_files[0]
            waveform, sr = self.load_audio(audio_path)
            
            # Audio player
            st.audio(str(audio_path))
            
            # Waveform plot
            st.plotly_chart(self.plot_waveform(waveform, sr))
            
            # Spectrogram plot
            st.plotly_chart(self.plot_spectrogram(waveform, sr))
        
        # Display regenerated spectrogram
        spectrograms = sorted(self.data_dir.glob('spectrograms/*_spec.png'))
        if spectrograms:
            st.header("Regenerated Spectrogram")
            spec_img = self.load_image(spectrograms[0])
            st.image(spec_img, caption="Regenerated Spectrogram")

def main():
    demo = MultimodalDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()