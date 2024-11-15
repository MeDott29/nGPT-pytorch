import torch
import torch.nn.functional as F
import pytest
import numpy as np
from token_agan1 import TokenAGAN, TokenGeneratorNet, TokenDiscriminatorNet

@pytest.fixture
def model_params():
    return {
        'latent_dim': 64,
        'hidden_dim': 128,
        'num_tokens': 32,
        'in_features': 128,
        'out_features': 128,
        'lr': 1e-4
    }

@pytest.fixture
def agan(model_params):
    return TokenAGAN(**model_params)

def test_generator_output_shape(agan, model_params):
    batch_size = 16
    z = torch.randn(batch_size, model_params['latent_dim']).to(agan.device)
    keys, values = agan.generator(z)
    
    assert keys.shape == (batch_size, model_params['num_tokens'], model_params['in_features'])
    assert values.shape == (batch_size, model_params['num_tokens'], model_params['out_features'])

def test_generator_normalization(agan, model_params):
    batch_size = 16
    z = torch.randn(batch_size, model_params['latent_dim']).to(agan.device)
    keys, values = agan.generator(z)
    
    # Check if vectors are normalized (unit length)
    key_norms = torch.norm(keys, dim=-1)
    value_norms = torch.norm(values, dim=-1)
    
    assert torch.allclose(key_norms, torch.ones_like(key_norms), atol=1e-6)
    assert torch.allclose(value_norms, torch.ones_like(value_norms), atol=1e-6)

def test_discriminator_real_fake_separation(agan, model_params):
    batch_size = 100
    
    # Generate "real" data (normalized random vectors)
    real_keys = F.normalize(torch.randn(batch_size, model_params['num_tokens'], model_params['in_features']), dim=-1).to(agan.device)
    real_values = F.normalize(torch.randn(batch_size, model_params['num_tokens'], model_params['out_features']), dim=-1).to(agan.device)
    
    # Generate fake data
    z = torch.randn(batch_size, model_params['latent_dim']).to(agan.device)
    fake_keys, fake_values = agan.generator(z)
    
    # Get discriminator predictions
    with torch.no_grad():
        real_preds = agan.discriminator(real_keys, real_values)
        fake_preds = agan.discriminator(fake_keys, fake_values)
    
    # Convert to probabilities
    real_probs = torch.sigmoid(real_preds)
    fake_probs = torch.sigmoid(fake_preds)
    
    # Basic statistical tests
    real_mean = real_probs.mean().item()
    fake_mean = fake_probs.mean().item()
    
    # Log results
    print(f"\nDiscriminator Baseline Metrics:")
    print(f"Real predictions mean: {real_mean:.4f}")
    print(f"Fake predictions mean: {fake_mean:.4f}")
    
    # Initial baseline should show some separation
    assert abs(real_mean - fake_mean) > 0.1, "Discriminator should show initial bias between real and fake"

def test_generator_diversity(agan, model_params):
    num_samples = 100
    z = torch.randn(num_samples, model_params['latent_dim']).to(agan.device)
    keys, values = agan.generator(z)
    
    # Compute pairwise cosine similarities
    keys_flat = keys.view(num_samples, -1)
    values_flat = values.view(num_samples, -1)
    
    key_sims = F.cosine_similarity(keys_flat.unsqueeze(1), keys_flat.unsqueeze(0), dim=2)
    value_sims = F.cosine_similarity(values_flat.unsqueeze(1), values_flat.unsqueeze(0), dim=2)
    
    # Remove self-similarities from diagonal
    mask = ~torch.eye(num_samples, dtype=bool)
    key_sims = key_sims[mask]
    value_sims = value_sims[mask]
    
    # Compute diversity metrics
    key_sim_mean = key_sims.mean().item()
    key_sim_std = key_sims.std().item()
    value_sim_mean = value_sims.mean().item()
    value_sim_std = value_sims.std().item()
    
    print(f"\nDiversity Metrics:")
    print(f"Key similarities - Mean: {key_sim_mean:.4f}, Std: {key_sim_std:.4f}")
    print(f"Value similarities - Mean: {value_sim_mean:.4f}, Std: {value_sim_std:.4f}")
    
    # Basic diversity checks
    assert key_sim_mean < 0.9, "Keys should not be too similar"
    assert value_sim_mean < 0.9, "Values should not be too similar"
    assert key_sim_std > 0.01, "Keys should have some variance"
    assert value_sim_std > 0.01, "Values should have some variance"

def test_training_step(agan, model_params):
    batch_size = 32
    
    # Generate random "real" data
    real_keys = F.normalize(torch.randn(batch_size, model_params['num_tokens'], model_params['in_features']), dim=-1)
    real_values = F.normalize(torch.randn(batch_size, model_params['num_tokens'], model_params['out_features']), dim=-1)
    
    # Perform one training step
    d_loss, g_loss = agan.train_step(real_keys, real_values)
    
    print(f"\nInitial Training Losses:")
    print(f"Discriminator loss: {d_loss:.4f}")
    print(f"Generator loss: {g_loss:.4f}")
    
    # Basic loss checks
    assert not torch.isnan(torch.tensor(d_loss)), "Discriminator loss should not be NaN"
    assert not torch.isnan(torch.tensor(g_loss)), "Generator loss should not be NaN"
    assert d_loss > 0, "Discriminator loss should be positive"
    assert g_loss > 0, "Generator loss should be positive"
