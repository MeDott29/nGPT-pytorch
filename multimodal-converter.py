import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft, fftfreq
from PIL import Image, ImageOps, ImageEnhance
import os
from pathlib import Path
import logging
from datasets import load_dataset
import torch
import torchaudio
import torch.nn as nn

logging.basicConfig(level=logging.INFO)

class SpectrogramGenerator(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=1024, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        
    def forward(self, waveform):
        return self.spectrogram(waveform)

def enhance_image(img):
    """Apply image enhancements for better quality conversion"""
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(1.2)

def preprocess_mnist(img):
    img = ImageOps.expand(img, border=2, fill='black').convert('RGB')
    return enhance_image(img)

def concat_mnist_images(imgs, grid_size=34):
    img_size = 32
    canvas_size = grid_size * img_size
    img_new = Image.new('RGB', (canvas_size, canvas_size), 'white')
    
    for i, img in enumerate(imgs):
        if i >= grid_size * grid_size:
            break
        x = (i % grid_size) * img_size
        y = (i // grid_size) * img_size
        img_new.paste(img, (x, y))
    
    return img_new.resize((1080, 1080), Image.LANCZOS)

def convert_audio_to_spectral_image(audio_path: str, output_folder: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        
        # Initialize spectrogram generator
        spec_gen = SpectrogramGenerator(sample_rate=sample_rate).to(device)
        
        # Generate spectrogram
        with torch.no_grad():
            spectrogram = spec_gen(waveform)
        
        # Convert to log scale and normalize
        spectrogram = torch.log1p(spectrogram)
        spectrogram = (spectrogram * 255 / torch.max(spectrogram)).cpu().numpy().astype(np.uint8)
        
        # Create enhanced image
        img = Image.fromarray(spectrogram[0])
        img = enhance_image(img)
        
        output_path = Path(output_folder) / f"{Path(audio_path).stem}_spec.png"
        img.save(output_path)
        logging.info(f"Created enhanced spectrogram: {output_path}")
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error in spectrogram generation: {e}")
        raise

class AudioGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def generate_from_image(self, image_path: str, output_folder: str):
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img) / 255.0
            
            # Apply frequency scaling
            frequencies = fftfreq(len(img_array.flatten()), 1/self.sample_rate)
            spectrum = fft(img_array.flatten())
            
            # Phase reconstruction with Griffin-Lim algorithm
            n_iterations = 50
            phase = np.random.randn(len(spectrum))
            
            for _ in range(n_iterations):
                complex_spec = np.abs(spectrum) * np.exp(1j * phase)
                audio = np.real(ifft(complex_spec))
                phase = np.angle(fft(audio))
            
            # Normalize and save
            audio = audio / np.max(np.abs(audio))
            output_path = Path(output_folder) / f"{Path(image_path).stem}.wav"
            wav.write(str(output_path), self.sample_rate, (audio * 32767).astype(np.int16))
            
            logging.info(f"Generated enhanced audio: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error in audio generation: {e}")
            raise

def process_mnist_dataset(batch_size=1156, output_base='./output'):
    output_folders = {
        'images': Path(output_base) / 'screenshots',
        'audio': Path(output_base) / 'audio',
        'spectrograms': Path(output_base) / 'spectrograms'
    }
    
    for folder in output_folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('mnist', split='train')
    audio_gen = AudioGenerator()
    batch = []
    count = 0
    
    for ex in dataset:
        batch.append(preprocess_mnist(ex['image']))
        if len(batch) == batch_size:
            count += 1
            
            # Generate and save image
            img_path = output_folders['images'] / f'mnist_{count}.jpg'
            img = concat_mnist_images(batch)
            img.save(img_path)
            
            # Convert to audio
            audio_path = audio_gen.generate_from_image(img_path, output_folders['audio'])
            
            # Generate spectrogram from audio
            convert_audio_to_spectral_image(audio_path, output_folders['spectrograms'])
            
            batch = []
            logging.info(f"Completed full conversion cycle for batch {count}")
    
    if batch:
        process_final_batch(batch, batch_size, count + 1, output_folders, audio_gen)

def process_final_batch(batch, batch_size, count, output_folders, audio_gen):
    batch += [Image.new('RGB', (32,32), 'white')] * (batch_size - len(batch))
    img_path = output_folders['images'] / f'mnist_{count}.jpg'
    img = concat_mnist_images(batch)
    img.save(img_path)
    
    audio_path = audio_gen.generate_from_image(img_path, output_folders['audio'])
    convert_audio_to_spectral_image(audio_path, output_folders['spectrograms'])
    logging.info(f"Processed final batch {count}")

def main():
    process_mnist_dataset()

if __name__ == "__main__":
    main()