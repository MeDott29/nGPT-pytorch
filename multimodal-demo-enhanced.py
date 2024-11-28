import streamlit as st
import plotly.graph_objects as go
import librosa
import numpy as np
from PIL import Image
import io
import torch
import torchaudio
import json
from pathlib import Path
import base64

class UnifiedFormat:
    def __init__(self):
        self.content = {
            "image": None,
            "audio": None,
            "spectrogram": None,
            "metadata": {}
        }
    
    def add_image(self, image_path):
        with open(image_path, "rb") as f:
            self.content["image"] = base64.b64encode(f.read()).decode()
            
    def add_audio(self, audio_path):
        with open(audio_path, "rb") as f:
            self.content["audio"] = base64.b64encode(f.read()).decode()
            
    def add_spectrogram(self, spec_path):
        with open(spec_path, "rb") as f:
            self.content["spectrogram"] = base64.b64encode(f.read()).decode()
            
    def save(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.content, f)
            
    def load(self, input_path):
        with open(input_path, "r") as f:
            self.content = json.load(f)
            
    def get_image_bytes(self):
        return base64.b64decode(self.content["image"]) if self.content["image"] else None
        
    def get_audio_bytes(self):
        return base64.b64decode(self.content["audio"]) if self.content["audio"] else None
        
    def get_spectrogram_bytes(self):
        return base64.b64decode(self.content["spectrogram"]) if self.content["spectrogram"] else None

class MultimodalDemo:
    def __init__(self, data_dir='./output'):
        self.data_dir = Path(data_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unified_format = UnifiedFormat()
    
    def create_unified_file(self):
        mnist_images = sorted(self.data_dir.glob('screenshots/*.jpg'))
        audio_files = sorted(self.data_dir.glob('audio/*.wav'))
        spectrograms = sorted(self.data_dir.glob('spectrograms/*_spec.png'))
        
        if mnist_images and audio_files and spectrograms:
            self.unified_format.add_image(mnist_images[0])
            self.unified_format.add_audio(audio_files[0])
            self.unified_format.add_spectrogram(spectrograms[0])
            self.unified_format.save(self.data_dir / "unified_data.json")
            return True
        return False

    def display_unified_format(self):
        st.header("Unified Multimodal Format")
        try:
            self.unified_format.load(self.data_dir / "unified_data.json")
            
            # Display image
            if self.unified_format.content["image"]:
                img_bytes = self.unified_format.get_image_bytes()
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, caption="MNIST Grid from Unified Format")
            
            # Display audio
            if self.unified_format.content["audio"]:
                audio_bytes = self.unified_format.get_audio_bytes()
                st.audio(audio_bytes)
            
            # Display spectrogram
            if self.unified_format.content["spectrogram"]:
                spec_bytes = self.unified_format.get_spectrogram_bytes()
                spec_img = Image.open(io.BytesIO(spec_bytes))
                st.image(spec_img, caption="Spectrogram from Unified Format")
                
        except Exception as e:
            st.error(f"Error loading unified format: {str(e)}")

    def run_demo(self):
        st.title("Multimodal Conversion Demo")
        
        # Create and display unified format
        if self.create_unified_file():
            self.display_unified_format()
        
        st.divider()
        
        # Original demo content continues here...
        mnist_images = sorted(self.data_dir.glob('screenshots/*.jpg'))
        if mnist_images:
            st.header("Original MNIST Grid")
            img = Image.open(mnist_images[0])
            st.image(img, caption="MNIST Digit Grid")
        
        audio_files = sorted(self.data_dir.glob('audio/*.wav'))
        if audio_files:
            st.header("Audio Generation")
            audio_path = audio_files[0]
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.numpy()[0]
            
            st.audio(str(audio_path))
            
            time = np.linspace(0, len(waveform) / sr, len(waveform))
            fig = go.Figure(data=go.Scatter(x=time, y=waveform, mode='lines'))
            fig.update_layout(title="Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
            st.plotly_chart(fig)
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
            fig = go.Figure(data=go.Heatmap(z=D))
            fig.update_layout(title="Audio Spectrogram", xaxis_title="Time", yaxis_title="Frequency")
            st.plotly_chart(fig)
        
        spectrograms = sorted(self.data_dir.glob('spectrograms/*_spec.png'))
        if spectrograms:
            st.header("Regenerated Spectrogram")
            spec_img = Image.open(spectrograms[0])
            st.image(spec_img, caption="Regenerated Spectrogram")

def main():
    demo = MultimodalDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()