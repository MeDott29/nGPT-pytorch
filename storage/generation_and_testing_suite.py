import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import seaborn as sns
from torch.utils.data import DataLoader
import pandas as pd
import scipy.stats as stats
from scipy.stats import entropy, linregress, normaltest, skew, kurtosis
import sounddevice as sd
import time
from scipy.io import wavfile
import os

@dataclass
class DataConfig:
    """Configuration for data generation parameters"""
    batch_size: int = 32
    in_features: int = 64
    max_seq_len: int = 128
    min_seq_len: int = 16
    noise_scale: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class StructuredDataGenerator:
    """Generates structured data patterns for testing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.device = config.device
        
    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of data, targets, and masks"""
        batch_size = self.config.batch_size
        max_len = self.config.max_seq_len
        min_len = self.config.min_seq_len
        features = self.config.in_features
        
        # Generate random sequence lengths
        seq_lengths = torch.randint(
            min_len, max_len + 1, 
            (batch_size,), 
            device=self.device
        )
        
        # Initialize tensors
        data = torch.zeros(
            (batch_size, max_len, features), 
            device=self.device
        )
        targets = torch.zeros(
            (batch_size, max_len, features), 
            device=self.device
        )
        masks = torch.zeros(
            (batch_size, max_len), 
            device=self.device
        )
        
        # Generate sequences
        for i in range(batch_size):
            length = seq_lengths[i]
            
            # Generate pattern
            pattern = self._generate_compositional_pattern(length)
            data[i, :length] = pattern
            
            # Generate target (e.g., next step prediction)
            targets[i, :length-1] = pattern[1:]
            
            # Set mask
            masks[i, :length] = 1
            
        return data, targets, masks
    
    def _generate_harmonic_pattern(self, length: int) -> torch.Tensor:
        """Generate harmonic pattern"""
        t = torch.linspace(0, 4*np.pi, length, device=self.device)
        pattern = torch.sin(t) + 0.5 * torch.sin(2*t) + 0.25 * torch.sin(3*t)
        return pattern.unsqueeze(-1).repeat(1, self.config.in_features)
    
    def _generate_ar_pattern(self, length: int) -> torch.Tensor:
        """Generate autoregressive pattern"""
        pattern = torch.randn(length, device=self.device)
        for i in range(2, length):
            pattern[i] = 0.6 * pattern[i-1] - 0.2 * pattern[i-2] + 0.1 * torch.randn(1, device=self.device)
        return pattern.unsqueeze(-1).repeat(1, self.config.in_features)
    
    def _generate_compositional_pattern(self, length: int) -> torch.Tensor:
        """Generate compositional pattern combining multiple patterns"""
        harmonic = self._generate_harmonic_pattern(length)
        ar = self._generate_ar_pattern(length)
        
        # Combine patterns with random weights
        weights = torch.rand(2, device=self.device)
        weights = weights / weights.sum()
        
        pattern = weights[0] * harmonic + weights[1] * ar
        
        # Add noise
        noise = torch.randn_like(pattern) * self.config.noise_scale
        return pattern + noise

class TestDataGenerator:
    """Comprehensive testing framework for data generation"""
    
    def __init__(self, configs: List[DataConfig]):
        self.configs = configs
        self.test_results = {}
        self.visualization_data = {}
        
    def run_shape_tests(self, generator: StructuredDataGenerator) -> Dict:
        """Test data shapes and basic properties"""
        data, targets, masks = generator.generate_batch()
        
        results = {
            'data_shape': data.shape,
            'targets_shape': targets.shape,
            'masks_shape': masks.shape,
            'data_mean': data.mean().item(),
            'data_std': data.std().item(),
            'target_mean': targets.mean().item(),
            'target_std': targets.std().item(),
            'mask_coverage': masks.float().mean().item()
        }
        
        return results
    
    def run_pattern_tests(self, generator: StructuredDataGenerator, length: int = 64) -> Dict:
        """Test individual pattern generators"""
        patterns = {}
        stats = {}
        
        # Test each pattern generator
        for name, func in [
            ('Harmonic', generator._generate_harmonic_pattern),
            ('AR', generator._generate_ar_pattern),
            ('Compositional', generator._generate_compositional_pattern)
        ]:
            pattern = func(length)
            patterns[name] = pattern.cpu().numpy()
            stats[f'{name}_mean'] = pattern.mean().item()
            stats[f'{name}_std'] = pattern.std().item()
            
            # Test stationarity
            if pattern.size(0) > 10:
                rolling_mean = torch.tensor([
                    pattern[i:i+10].mean() 
                    for i in range(length-10)
                ])
                stats[f'{name}_stationarity'] = rolling_mean.std().item()
        
        return {'patterns': patterns, 'stats': stats}
    
    def run_sequence_tests(self, generator: StructuredDataGenerator) -> Dict:
        """Test sequence properties"""
        _, _, masks = generator.generate_batch()
        seq_lengths = masks.sum(dim=1)
        
        results = {
            'min_length': seq_lengths.min().item(),
            'max_length': seq_lengths.max().item(),
            'mean_length': seq_lengths.float().mean().item(),
            'length_std': seq_lengths.float().std().item()
        }
        
        return results
    
    def run_correlation_tests(self, generator: StructuredDataGenerator) -> Dict:
        """Test correlations in generated data"""
        data, targets, _ = generator.generate_batch()
        
        # Sample a random sequence
        seq_idx = torch.randint(0, data.size(0), (1,)).item()
        sequence = data[seq_idx]  # shape: [seq_len, features]
        
        # Compute autocorrelation
        auto_corr = []
        for i in range(1, min(20, sequence.size(0))):
            # Flatten the features dimension and compute correlation
            seq1 = sequence[:-i].mean(dim=-1)  # Average across features
            seq2 = sequence[i:].mean(dim=-1)   # Average across features
            lagged_data = torch.stack([seq1, seq2])
            corr = torch.corrcoef(lagged_data)[0, 1].item()
            auto_corr.append(corr)
        
        auto_corr = torch.tensor(auto_corr)
        
        # Compute feature correlations
        feature_corr = torch.corrcoef(sequence.T)
        
        return {
            'autocorrelation': auto_corr.cpu().numpy(),
            'feature_correlation': feature_corr.cpu().numpy()
        }
    
    def test_data_loader(self, config: DataConfig, num_batches: int = 5) -> Dict:
        """Test data loader functionality"""
        loader = prepare_dataloader(config)
        batch_stats = []
        
        for _ in range(num_batches):
            data, targets, masks = next(iter(loader))
            stats = {
                'batch_data_mean': data.mean().item(),
                'batch_data_std': data.std().item(),
                'batch_mask_coverage': masks.float().mean().item()
            }
            batch_stats.append(stats)
            
        return pd.DataFrame(batch_stats).describe().to_dict()
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Create visualization plots for test results"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Pattern Visualization
        plt.subplot(3, 2, 1)
        for name, pattern in self.visualization_data['patterns'].items():
            plt.plot(pattern[:, 0], label=name, alpha=0.7)  # Plot first feature only
        plt.title('Generated Patterns')
        plt.legend()
        
        # 2. Sequence Length Distribution
        plt.subplot(3, 2, 2)
        seq_lengths = []
        for config_idx, results in self.test_results.items():
            seq_tests = results['sequence_tests']
            plt.hist(
                [seq_tests['mean_length']], 
                alpha=0.5, 
                label=f'Config {config_idx}'
            )
        plt.title('Sequence Length Distribution')
        plt.legend()
        
        # 3. Autocorrelation
        plt.subplot(3, 2, 3)
        for config_idx, results in self.test_results.items():
            acf = results['correlation_tests']['autocorrelation']
            plt.plot(acf, label=f'Config {config_idx}')
        plt.title('Autocorrelation Function')
        plt.legend()
        
        # 4. Feature Correlation Heatmap
        plt.subplot(3, 2, 4)
        feature_corr = self.test_results[0]['correlation_tests']['feature_correlation']
        plt.imshow(feature_corr, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('Feature Correlations')
        
        # 5. Batch Statistics
        plt.subplot(3, 2, 5)
        stats = []
        for res in self.test_results.values():
            stats.append(res['shape_tests']['data_mean'])
        plt.boxplot(stats)
        plt.title('Batch Statistics Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def run_all_tests(self):
        """Run all tests for each configuration"""
        for i, config in enumerate(self.configs):
            print(f"\nTesting Configuration {i+1}")
            print("-" * 40)
            
            generator = StructuredDataGenerator(config)
            
            # Run all tests
            shape_results = self.run_shape_tests(generator)
            pattern_results = self.run_pattern_tests(generator)
            sequence_results = self.run_sequence_tests(generator)
            correlation_results = self.run_correlation_tests(generator)
            loader_results = self.test_data_loader(config)
            
            # Store results
            self.test_results[i] = {
                'shape_tests': shape_results,
                'pattern_tests': pattern_results['stats'],
                'sequence_tests': sequence_results,
                'correlation_tests': correlation_results,
                'loader_tests': loader_results
            }
            
            # Store visualization data
            if i == 0:  # Store patterns from first config for visualization
                self.visualization_data['patterns'] = pattern_results['patterns']
            
            # Print results
            print("\nShape Tests:")
            for k, v in shape_results.items():
                print(f"{k}: {v}")
                
            print("\nPattern Tests:")
            for k, v in pattern_results['stats'].items():
                print(f"{k}: {v:.3f}")
                
            print("\nSequence Tests:")
            for k, v in sequence_results.items():
                print(f"{k}: {v:.3f}")
                
        return self.test_results

def run_statistical_tests(data_generator: StructuredDataGenerator, num_samples: int = 100):
    """Run statistical tests on generated data"""
    samples = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        data, _, _ = data_generator.generate_batch()
        samples.append(data.mean().item())
    
    samples = np.array(samples)
    
    return {
        'mean': np.mean(samples),
        'std': np.std(samples),
        'skewness': skew(samples),
        'kurtosis': kurtosis(samples),
        'normality_test': normaltest(samples)
    }

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    n_samples: int = 1000
    significance_level: float = 0.05
    visualization_dpi: int = 300
    correlation_threshold: float = 0.3
    stationarity_window: int = 20

class DataAnalyzer:
    """Advanced analysis tools for generated data"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analysis_results = {}
        
    def analyze_pattern_complexity(self, pattern: torch.Tensor) -> Dict:
        """Analyze complexity metrics of generated patterns"""
        pattern_np = pattern.cpu().numpy()
        
        # Compute histogram and normalize it to get probability distribution
        hist, _ = np.histogram(pattern_np, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities for entropy calculation
        
        # Compute various complexity metrics
        results = {
            'entropy': entropy(hist),  # Use the imported entropy function directly
            'peak_frequency': self._find_peak_frequency(pattern_np),
            'linear_trend': self._compute_linear_trend(pattern_np),
            'turning_points': self._count_turning_points(pattern_np)
        }
        
        # Add Hurst exponent for long-term dependency
        results['hurst_exponent'] = self._compute_hurst_exponent(pattern_np)
        
        return results
    
    def _find_peak_frequency(self, pattern: np.ndarray) -> float:
        """Find dominant frequency using FFT"""
        # Average across features if multi-dimensional
        if len(pattern.shape) > 1:
            pattern = pattern.mean(axis=1)
            
        fft = np.fft.fft(pattern)
        freqs = np.fft.fftfreq(len(pattern))
        peak_freq = freqs[np.argmax(np.abs(fft))]
        return abs(peak_freq)
    
    def _compute_linear_trend(self, pattern: np.ndarray) -> float:
        """Compute linear trend strength"""
        # Average across features if multi-dimensional
        if len(pattern.shape) > 1:
            pattern = pattern.mean(axis=1)
            
        x = np.arange(len(pattern))
        slope, _, r_value, _, _ = linregress(x, pattern)  # Use imported linregress directly
        return r_value ** 2
    
    def _count_turning_points(self, pattern: np.ndarray) -> int:
        """Count number of local maxima and minima"""
        diff = np.diff(pattern)
        return np.sum((diff[:-1] > 0) & (diff[1:] < 0)) + \
               np.sum((diff[:-1] < 0) & (diff[1:] > 0))
    
    def _compute_hurst_exponent(self, pattern: np.ndarray, max_lag: Optional[int] = None) -> float:
        """Compute Hurst exponent for long-term dependency analysis"""
        if max_lag is None:
            max_lag = len(pattern) // 4
            
        lags = range(2, max_lag)
        rs = []
        
        for lag in lags:
            rs.append(self._compute_rs(pattern, lag))
            
        if len(rs) > 1:
            h, _ = np.polyfit(np.log(lags), np.log(rs), 1)
            return h
        return np.nan
    
    def _compute_rs(self, pattern: np.ndarray, lag: int) -> float:
        """Compute R/S statistic for Hurst exponent calculation"""
        mean = pattern[:lag].mean()
        centered = pattern[:lag] - mean
        z = centered.cumsum()
        r = z.max() - z.min()
        s = np.std(pattern[:lag])
        if s == 0:
            return np.nan
        return r / s
    
    def analyze_temporal_structure(self, 
                                 generator: StructuredDataGenerator,
                                 sequence_length: int = 128) -> Dict:
        """Analyze temporal structure of generated sequences"""
        sequences = []
        for _ in range(self.config.n_samples):
            data, _, _ = generator.generate_batch()
            sequences.append(data[0, :sequence_length].cpu().numpy())
        
        sequences = np.array(sequences)
        
        results = {
            'temporal_correlation': self._compute_temporal_correlation(sequences),
            'stationarity': self._test_stationarity(sequences),
            'periodicity': self._analyze_periodicity(sequences),
            'conditional_entropy': self._compute_conditional_entropy(sequences)
        }
        
        return results
    
    def save_pattern_audio(self, pattern: torch.Tensor, filename: str, sample_rate: int = 44100):
        """Save the pattern as a WAV audio file"""
        # If pattern is 2D, take mean across features
        if len(pattern.shape) > 1:
            pattern = pattern.mean(dim=1)
            
        # Convert to numpy and ensure it's float32
        pattern = pattern.cpu().numpy().astype(np.float32)
        
        # Apply a gentle fade in/out to prevent clicks
        fade_length = min(1000, len(pattern) // 10)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        pattern[:fade_length] *= fade_in
        pattern[-fade_length:] *= fade_out
        
        # Normalize to [-1, 1] range, with some headroom
        max_val = np.max(np.abs(pattern))
        if max_val > 0:
            pattern = pattern / max_val * 0.9  # Leave some headroom
        
        # Convert to 16-bit PCM with dithering to reduce quantization noise
        pattern = (pattern * 32767).astype(np.int16)
        
        # Add a small amount of silence at the start and end
        silence = np.zeros(int(sample_rate * 0.1), dtype=np.int16)  # 0.1 seconds of silence
        pattern = np.concatenate([silence, pattern, silence])
        
        try:
            wavfile.write(filename, sample_rate, pattern)
            print(f"Successfully saved {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    def play_and_save_pattern(self, pattern: torch.Tensor, filename: str, sample_rate: int = 44100):
        """Play the pattern and save it to a file"""
        self.save_pattern_audio(pattern, filename, sample_rate)
        
        # Also play it if desired
        if len(pattern.shape) > 1:
            pattern = pattern.mean(dim=1)
        pattern = pattern.cpu().numpy()
        pattern = pattern / np.max(np.abs(pattern))
        
        sd.play(pattern, sample_rate)
        time.sleep(len(pattern) / sample_rate)
        sd.stop()

def prepare_dataloader(config: DataConfig) -> DataLoader:
    """Create a DataLoader for testing"""
    class SimpleDataset:
        def __init__(self, generator: StructuredDataGenerator):
            self.generator = generator
        
        def __getitem__(self, _):
            return self.generator.generate_batch()
        
        def __len__(self):
            return 1000  # Arbitrary length since we generate data on-the-fly
    
    generator = StructuredDataGenerator(config)
    dataset = SimpleDataset(generator)
    
    return DataLoader(
        dataset,
        batch_size=None,  # Already batched by generator
        shuffle=False
    )

if __name__ == "__main__":
    # Test configurations
    configs = [
        DataConfig(
            batch_size=32,
            in_features=64,
            max_seq_len=128,
            noise_scale=0.05,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ),
        DataConfig(
            batch_size=16,
            in_features=32,
            max_seq_len=64,
            min_seq_len=32,
            noise_scale=0.1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    ]
    
    # Initialize and run tests
    tester = TestDataGenerator(configs)
    results = tester.run_all_tests()
    
    # Create visualizations
    tester.visualize_results(save_path="data_generation_tests.png")
    
    # Run statistical tests for the first configuration
    print("\nRunning Statistical Tests")
    print("-" * 40)
    generator = StructuredDataGenerator(configs[0])
    stats = run_statistical_tests(generator)
    
    print("\nStatistical Test Results:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    # Add analysis tests
    analysis_config = AnalysisConfig()
    analyzer = DataAnalyzer(analysis_config)
    
    # Run analysis on the first generator
    generator = StructuredDataGenerator(configs[0])
    data, _, _ = generator.generate_batch()
    pattern = data[0]  # Analyze first sequence
    
    complexity_results = analyzer.analyze_pattern_complexity(pattern)
    print("\nPattern Complexity Analysis:")
    for k, v in complexity_results.items():
        print(f"{k}: {v}")
    
    # Create output directory if it doesn't exist
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating and saving audio patterns...")
    
    # Get a batch of data
    data, _, _ = generator.generate_batch()
    pattern = data[0]  # Take first sequence
    
    patterns = {
        'harmonic': generator._generate_harmonic_pattern(128),
        'ar': generator._generate_ar_pattern(128),
        'compositional': generator._generate_compositional_pattern(128)
    }
    
    # Save each pattern type
    for name, pat in patterns.items():
        filename = os.path.join(output_dir, f"{name}_pattern.wav")
        print(f"\nSaving {name} pattern to {filename}...")
        analyzer.save_pattern_audio(pat, filename)
        
        # Optionally play it as well
        print(f"Playing {name} pattern...")
        analyzer.play_and_save_pattern(pat, filename)
        time.sleep(1)  # Pause between patterns
    
    print(f"\nAudio files saved in {output_dir}/")
