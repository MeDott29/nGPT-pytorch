import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
from PIL import Image, ImageOps
import os
from pathlib import Path
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

def preprocess_mnist(img):
    return ImageOps.expand(img, border=2, fill='black').convert('RGB')

def concat_mnist_images(imgs):
    img_new = Image.new('RGB', (34*32, 34*32), 'white')
    for i, img in enumerate(imgs):
        img_new.paste(img, ((i%34)*32, (i//34)*32))
    return img_new.resize((1080,1080), Image.LANCZOS)

def convert_audio_to_spectral_image(audio_path: str, output_folder: str):
    try:
        sample_rate, audio = wav.read(audio_path)
        audio = audio.astype(float) / 32767.0
        
        # Use STFT for better frequency representation
        window_size = 1024
        hop_size = 512
        spectrogram = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            spectrum = np.abs(fft(window))
            spectrogram.append(spectrum[:window_size//2])
            
        spectrogram = np.array(spectrogram)
        
        # Convert to logarithmic scale and normalize
        spectrogram = np.log1p(spectrogram)
        spectrogram = (spectrogram * 255 / np.max(spectrogram)).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(spectrogram)
        output_path = Path(output_folder) / f"{Path(audio_path).stem}_spec.png"
        img.save(output_path)
        logging.info(f"Created spectrogram: {output_path}")
        
    except Exception as e:
        logging.error(f"Error in spectrogram generation: {e}")

def convert_image_to_audio(image_path: str, output_folder: str):
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img) / 255.0
        
        # Convert back to frequency domain
        frequencies = fft(img_array.flatten())
        
        # Generate audio with phase reconstruction
        audio_signal = np.real(ifft(frequencies))
        audio_signal = audio_signal / np.max(np.abs(audio_signal))
        
        output_path = Path(output_folder) / f"{Path(image_path).stem}.wav"
        wav.write(str(output_path), 44100, (audio_signal * 32767).astype(np.int16))
        logging.info(f"Generated audio: {output_path}")
        
    except Exception as e:
        logging.error(f"Error in audio generation: {e}")

def process_mnist_dataset():
    output_folder = Path("./output/screenshots")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('ylecun/mnist', split='train')
    batch = []
    count = 0
    
    for ex in dataset:
        batch.append(preprocess_mnist(ex['image']))
        if len(batch) == 1156:
            count += 1
            img = concat_mnist_images(batch)
            img.save(f'./output/screenshots/mnist_{count}.jpg')
            
            # Convert to audio
            convert_image_to_audio(f'./output/screenshots/mnist_{count}.jpg', './output/audio')
            batch = []
            logging.info(f"Processed batch {count}")
            
    if batch:
        batch += [Image.new('RGB', (32,32), 'white')] * (1156 - len(batch))
        count += 1
        img = concat_mnist_images(batch)
        img.save(f'./output/screenshots/mnist_{count}.jpg')
        convert_image_to_audio(f'./output/screenshots/mnist_{count}.jpg', './output/audio')
        logging.info(f"Processed final batch {count}")

def main():
    # Create output folders
    os.makedirs("./output/audio", exist_ok=True)
    os.makedirs("./output/spectrograms", exist_ok=True)
    
    # Process MNIST dataset
    process_mnist_dataset()
    
    # Convert generated audio back to spectrograms
    for audio_file in Path("/home/ath/Pictures/Data-Generation-and-Testing/nGPT-pytorch/converted").glob("*.wav"):
        convert_audio_to_spectral_image(str(audio_file), "./output/spectrograms")

if __name__ == "__main__":
    main()