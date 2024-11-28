import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
from PIL import Image
import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)

def convert_audio_to_image(audio_path: str, output_folder: str):
    try:
        sample_rate, audio = wav.read(audio_path)
        audio = audio.astype(float) / 32767.0  # Normalize
        
        # Convert to frequency domain
        frequencies = fft(audio)
        
        # Reshape to square-ish image
        size = int(np.sqrt(len(frequencies)))
        img_data = np.abs(frequencies[:size*size]).reshape(size, size)
        
        # Normalize and convert to uint8
        img_data = (img_data * 255 / np.max(img_data)).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_data)
        output_path = Path(output_folder) / f"{Path(audio_path).stem}.png"
        img.save(output_path)
        logging.info(f"Converted {audio_path} to {output_path}")
        
    except Exception as e:
        logging.error(f"Error converting {audio_path}: {e}")

def convert_image_to_audio(image_path: str, output_folder: str):
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img) / 255.0  # Normalize
        
        # Convert to audio signal
        frequencies = fft(img_array.flatten())
        audio_signal = np.real(ifft(frequencies))
        audio_signal = audio_signal / np.max(np.abs(audio_signal))
        
        # Save audio
        output_path = Path(output_folder) / f"{Path(image_path).stem}.wav"
        wav.write(str(output_path), 44100, (audio_signal * 32767).astype(np.int16))
        logging.info(f"Converted {image_path} to {output_path}")
        
    except Exception as e:
        logging.error(f"Error converting {image_path}: {e}")

def main():
    # Create output folder
    output_folder = Path("converted")
    output_folder.mkdir(exist_ok=True)
    
    # Process all WAV files in current directory
    for file in Path(".").glob("*.wav"):
        convert_audio_to_image(str(file), str(output_folder))
    
    # Process all PNG/JPG files in current directory
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for file in Path(".").glob(ext):
            convert_image_to_audio(str(file), str(output_folder))

if __name__ == "__main__":
    main()