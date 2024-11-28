import numpy as np
import pytest
import soundfile as sf
from PIL import Image
import tempfile
import os
from typing import Tuple
import shutil

class AudioImageConverter:
    def __init__(self, sample_rate=44100, bit_depth=16):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.max_val = 2**(bit_depth-1) - 1

    def image_to_audio(self, img_array: np.ndarray) -> np.ndarray:
        """Convert image data to audio signal"""
        # Normalize image to [-1, 1] with better precision
        normalized = img_array.astype(np.float32) / 255.0 * 2 - 1
        # Scale to audio range while preserving precision
        audio = (normalized * self.max_val).astype(np.int16)
        return audio.flatten()

    def audio_to_image(self, audio: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Convert audio signal back to image"""
        # Convert to float32 for better precision
        audio_float = audio.astype(np.float32)
        # Denormalize carefully to avoid rounding errors
        normalized = ((audio_float / self.max_val) + 1) / 2
        # Clip values to ensure they're in [0, 1] range
        normalized = np.clip(normalized, 0, 1)
        # Scale to [0, 255] and reshape
        image = (normalized * 255).reshape(shape)
        return image.astype(np.uint8)

def test_audio_image_pipeline():
    # Setup
    converter = AudioImageConverter()
    test_shapes = [(28, 28), (64, 64), (128, 128)]
    temp_dir = tempfile.mkdtemp()
    
    try:
        for shape in test_shapes:
            # Generate test image with controlled patterns instead of random
            test_image = np.zeros(shape, dtype=np.uint8)
            test_image[::2, ::2] = 255  # Create a checkerboard pattern
            
            # Convert image to audio
            audio = converter.image_to_audio(test_image)
            
            # Save and load audio to verify file operations
            audio_path = os.path.join(temp_dir, f"test_{shape[0]}x{shape[1]}.wav")
            sf.write(audio_path, audio, converter.sample_rate)
            loaded_audio, _ = sf.read(audio_path)
            
            # Convert back to image
            reconstructed = converter.audio_to_image(loaded_audio, shape)
            
            # Verify quality with more specific metrics
            mse = np.mean((test_image - reconstructed) ** 2)
            psnr = 20 * np.log10(255 / np.sqrt(mse))
            max_diff = np.max(np.abs(test_image - reconstructed))
            
            # Print detailed results
            print(f"\nTest Results for {shape[0]}x{shape[1]}:")
            print(f"MSE: {mse:.2f}")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"Max Pixel Difference: {max_diff}")
            print(f"Mean Absolute Error: {np.mean(np.abs(test_image - reconstructed)):.2f}")
            
            # More specific assertions
            assert psnr > 35, f"PSNR too low: {psnr:.2f} dB (expected > 35 dB)"
            assert mse < 50, f"MSE too high: {mse:.2f} (expected < 50)"
            assert max_diff < 20, f"Max difference too high: {max_diff} (expected < 20)"
            
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__])
