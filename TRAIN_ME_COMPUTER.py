import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from typing import Tuple, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodecConfig:
    """Configuration parameters for the codec system"""
    sample_rate: int = 44100
    min_freq: int = 20
    max_freq: int = 20000
    bits_per_pixel: int = 8
    chunk_size: int = 1024
    error_correction: bool = True
    compression_level: int = 5
    
class AudioImageCodec:
    """Main codec system for converting between images and audio"""
    
    def __init__(self, config: CodecConfig):
        self.config = config
        self.metadata: Dict[str, Any] = {}
        
    def encode_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert image to audio signal"""
        logger.info(f"Encoding image: {image_path}")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img) / 255.0  # Normalize to 0-1
        
        # Store metadata
        self.metadata = {
            'original_size': img_array.shape,
            'bits_per_pixel': self.config.bits_per_pixel,
            'sample_rate': self.config.sample_rate,
            'checksum': np.sum(img_array)
        }
        
        # Convert to frequency domain
        frequencies = self._image_to_frequencies(img_array)
        
        # Generate audio signal
        audio_signal = self._frequencies_to_audio(frequencies)
        
        # Apply error correction if enabled
        if self.config.error_correction:
            audio_signal = self._add_error_correction(audio_signal)
        
        return audio_signal, self.metadata
    
    def decode_audio(self, audio_signal: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Convert audio signal back to image"""
        logger.info("Decoding audio to image")
        
        # Remove error correction if present
        if self.config.error_correction:
            audio_signal = self._remove_error_correction(audio_signal)
        
        # Convert back to frequencies
        frequencies = self._audio_to_frequencies(audio_signal)
        
        # Reconstruct image
        img_array = self._frequencies_to_image(frequencies, metadata['original_size'])
        
        # Verify checksum
        if abs(np.sum(img_array) - metadata['checksum']) > 1e-6:
            logger.warning("Checksum verification failed!")
        
        return img_array
    
    def _image_to_frequencies(self, img_array: np.ndarray) -> np.ndarray:
        """Convert image data to frequency domain"""
        rows, cols = img_array.shape
        frequencies = []
        
        for row in range(rows):
            row_data = img_array[row, :]
            row_freq = fft(row_data)
            frequencies.append(row_freq)
            
        return np.array(frequencies)
    
    def _frequencies_to_audio(self, frequencies: np.ndarray) -> np.ndarray:
        """Convert frequencies to audio signal"""
        # Store the original shape for reconstruction
        self.metadata['frequency_shape'] = frequencies.shape
        
        total_samples = int(self.config.sample_rate * 
                          (frequencies.shape[0] * frequencies.shape[1]) / 
                          self.config.max_freq)
        
        audio_signal = np.zeros(total_samples)
        t = np.linspace(0, 1, total_samples)
        
        for i, row_freq in enumerate(frequencies):
            for j, freq_component in enumerate(row_freq):
                if abs(freq_component) > 1e-6:  # Skip near-zero components
                    frequency = self._map_to_audible_freq(i, j)
                    audio_signal += abs(freq_component) * np.sin(2 * np.pi * frequency * t)
        
        # Normalize
        audio_signal /= np.max(np.abs(audio_signal))
        return audio_signal
    
    def _audio_to_frequencies(self, audio_signal: np.ndarray) -> np.ndarray:
        """Convert audio signal back to frequencies"""
        # Reconstruct the original frequency shape
        original_shape = self.metadata['frequency_shape']
        rows, cols = original_shape
        
        frequencies = np.zeros(original_shape, dtype=complex)
        chunk_size = cols  # Use the original column size as chunk size
        
        for i in range(rows):
            chunk = audio_signal[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) < chunk_size:  # Pad if necessary
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            frequencies[i] = fft(chunk)
        
        return frequencies
    
    def _frequencies_to_image(self, frequencies: np.ndarray, 
                            original_size: Tuple[int, int]) -> np.ndarray:
        """Convert frequencies back to image"""
        rows, cols = original_size
        img_array = np.zeros(original_size)
        
        for i in range(rows):
            row_data = ifft(frequencies[i])
            img_array[i, :] = np.real(row_data)
            
        # Normalize and clip
        img_array = np.clip(img_array, 0, 1)
        return img_array
    
    def _map_to_audible_freq(self, i: int, j: int) -> float:
        """Map image coordinates to audible frequencies"""
        freq_range = self.config.max_freq - self.config.min_freq
        mapped_freq = self.config.min_freq + (i * j * freq_range) / (self.metadata['original_size'][0] * self.metadata['original_size'][1])
        return mapped_freq
    
    def _add_error_correction(self, audio_signal: np.ndarray) -> np.ndarray:
        """Add error correction codes to audio signal"""
        # Simple repetition code for demonstration
        return np.repeat(audio_signal, 3)
    
    def _remove_error_correction(self, audio_signal: np.ndarray) -> np.ndarray:
        """Remove error correction and correct errors"""
        # Take median of each group of 3 samples
        return np.median(audio_signal.reshape(-1, 3), axis=1)

class CodecAnalyzer:
    """Analysis and visualization tools for the codec system"""
    
    def __init__(self, codec: AudioImageCodec):
        self.codec = codec
        
    def analyze_quality(self, original: np.ndarray, 
                       reconstructed: np.ndarray) -> Dict[str, float]:
        """Analyze the quality of reconstruction"""
        mse = np.mean((original - reconstructed) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        return {
            'mse': mse,
            'psnr': psnr,
            'max_error': np.max(np.abs(original - reconstructed)),
            'structural_similarity': self._calculate_ssim(original, reconstructed)
        }
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Simplified SSIM implementation
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
        return float(ssim)
    
    def visualize_results(self, original: np.ndarray, 
                         reconstructed: np.ndarray, 
                         audio_signal: np.ndarray) -> None:
        """Visualize original, reconstructed images and audio waveform"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        # Reconstructed image
        axes[0, 1].imshow(reconstructed, cmap='gray')
        axes[0, 1].set_title('Reconstructed Image')
        
        # Audio waveform
        axes[1, 0].plot(audio_signal[:1000])
        axes[1, 0].set_title('Audio Waveform (first 1000 samples)')
        
        # Error map
        error_map = np.abs(original - reconstructed)
        im = axes[1, 1].imshow(error_map, cmap='hot')
        axes[1, 1].set_title('Error Map')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate the system"""
    # Initialize system
    config = CodecConfig()
    codec = AudioImageCodec(config)
    analyzer = CodecAnalyzer(codec)
    
    # Test with sample image
    image_path = "test_image.png"
    
    # Create test image if it doesn't exist
    if not os.path.exists(image_path):
        test_img = Image.fromarray(np.random.randint(0, 255, (100, 100), 
                                                    dtype=np.uint8))
        test_img.save(image_path)
    
    # Encode image to audio
    audio_signal, metadata = codec.encode_image(image_path)
    
    # Save audio file
    wav.write("encoded_audio.wav", config.sample_rate, 
              (audio_signal * 32767).astype(np.int16))
    
    # Load original image for comparison
    original = np.array(Image.open(image_path).convert('L')) / 255.0
    
    # Decode audio back to image
    reconstructed = codec.decode_audio(audio_signal, metadata)
    
    # Analyze results
    quality_metrics = analyzer.analyze_quality(original, reconstructed)
    logger.info("Quality metrics:")
    for metric, value in quality_metrics.items():
        logger.info(f"{metric}: {value}")
    
    # Visualize results
    analyzer.visualize_results(original, reconstructed, audio_signal)

if __name__ == "__main__":
    main()