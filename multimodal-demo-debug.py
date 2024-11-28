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
import logging

logging.basicConfig(level=logging.INFO)

class UnifiedFormat:
    def __init__(self):
        from datetime import datetime
        self.content = {
            "image": None,
            "audio": None,
            "spectrogram": None,
            "metadata": {
                "creation_time": datetime.now().isoformat(),
                "file_sizes": {},
                "formats": {},
                "stats": {}
            }
        }
    
    def add_image(self, image_path):
        try:
            img = Image.open(image_path)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG')
            self.content["image"] = base64.b64encode(img_byte_arr.getvalue()).decode()
            self.content["metadata"]["formats"]["image"] = img.format
            self.content["metadata"]["file_sizes"]["image"] = len(img_byte_arr.getvalue())
            logging.info(f"Added image: {image_path}")
        except Exception as e:
            logging.error(f"Error adding image: {e}")
            raise
            
    def add_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            buffer = io.BytesIO()
            torchaudio.save(buffer, waveform, sr, format="wav")
            self.content["audio"] = base64.b64encode(buffer.getvalue()).decode()
            self.content["metadata"]["formats"]["audio"] = "wav"
            self.content["metadata"]["file_sizes"]["audio"] = len(buffer.getvalue())
            logging.info(f"Added audio: {audio_path}")
        except Exception as e:
            logging.error(f"Error adding audio: {e}")
            raise
            
    def add_spectrogram(self, spec_path):
        try:
            spec_img = Image.open(spec_path)
            img_byte_arr = io.BytesIO()
            spec_img.save(img_byte_arr, format=spec_img.format or 'PNG')
            self.content["spectrogram"] = base64.b64encode(img_byte_arr.getvalue()).decode()
            self.content["metadata"]["formats"]["spectrogram"] = spec_img.format
            self.content["metadata"]["file_sizes"]["spectrogram"] = len(img_byte_arr.getvalue())
            logging.info(f"Added spectrogram: {spec_path}")
        except Exception as e:
            logging.error(f"Error adding spectrogram: {e}")
            raise

    def verify_content(self):
        missing = []
        for key in ["image", "audio", "spectrogram"]:
            if not self.content[key]:
                missing.append(key)
        return len(missing) == 0, missing

class MultimodalDemo:
    def __init__(self, data_dir='./output'):
        self.data_dir = Path(data_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unified_format = UnifiedFormat()
        
    def find_files(self):
        files = {
            'mnist': list(self.data_dir.glob('screenshots/*.jpg')),
            'audio': list(self.data_dir.glob('audio/*.wav')),
            'spectrograms': list(self.data_dir.glob('spectrograms/*_spec.png'))
        }
        return files

    def create_unified_file(self):
        try:
            files = self.find_files()
            logging.info(f"Found files: {files}")
            
            if not all(files.values()):
                missing = [k for k, v in files.items() if not v]
                logging.error(f"Missing files in directories: {missing}")
                return False
            
            self.unified_format.add_image(files['mnist'][0])
            self.unified_format.add_audio(files['audio'][0])
            self.unified_format.add_spectrogram(files['spectrograms'][0])
            
            is_complete, missing = self.unified_format.verify_content()
            if not is_complete:
                logging.error(f"Incomplete unified format. Missing: {missing}")
                return False
                
            unified_path = self.data_dir / "unified_data.json"
            with open(unified_path, 'w') as f:
                json.dump(self.unified_format.content, f)
            logging.info(f"Created unified file: {unified_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating unified file: {e}")
            return False

    def display_unified_format(self):
        st.header("Unified Multimodal Format")
        try:
            unified_path = self.data_dir / "unified_data.json"
            if not unified_path.exists():
                st.error("Unified format file not found")
                return
                
            with open(unified_path, 'r') as f:
                content = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if content["image"]:
                    img_bytes = base64.b64decode(content["image"])
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption="MNIST Grid")
                    
            with col2:
                if content["audio"]:
                    audio_bytes = base64.b64decode(content["audio"])
                    st.audio(audio_bytes)
                    
            with col3:
                if content["spectrogram"]:
                    spec_bytes = base64.b64decode(content["spectrogram"])
                    spec_img = Image.open(io.BytesIO(spec_bytes))
                    st.image(spec_img, caption="Spectrogram")
                    
            with st.expander("View Metadata"):
                st.json(content["metadata"])
                
            st.write("Total Data Size: ", 
                    f"{sum(content['metadata']['file_sizes'].values()) / (1024*1024):.2f} MB")
                
        except Exception as e:
            logging.error(f"Error displaying unified format: {e}")
            st.error(f"Error displaying content: {str(e)}")

    def run_demo(self):
        st.title("Multimodal Conversion Demo")
        
        if not self.create_unified_file():
            st.error("Failed to create unified format file. Check logs for details.")
            return
            
        self.display_unified_format()
        st.divider()

def main():
    demo = MultimodalDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()