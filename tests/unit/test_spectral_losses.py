"""
Unit tests for spectral losses (BSP and SA-BSP).

Tests cover:
- BinnedSpectralLoss: 1D FFT, binning, loss computation
- SelfAdaptiveBSPLoss: Adaptive weights, weight updates
- CombinedLoss: Base + spectral combination
- Loss factory: Configuration and creation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.core.evaluation.binned_spectral_loss import BinnedSpectralLoss
from src.core.evaluation.adaptive_spectral_loss import (
    SelfAdaptiveWeights,
    SelfAdaptiveBSPLoss,
    create_optimizers_for_sa_bsp
)
from src.core.evaluation.loss_factory import CombinedLoss, create_loss, create_loss_from_dict
from src.core.evaluation.metrics import RelativeL2Loss
from configs.loss_config import LossConfig, BASELINE_CONFIG, BSP_CONFIG, SA_BSP_CONFIG


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_signal():
    """Create sample temporal signal [B, C, T]."""
    B, C, T = 4, 1, 1000
    # Generate signal with mixed frequencies
    t = torch.linspace(0, 1, T)
    signal = torch.sin(2 * np.pi * 5 * t) + 0.5 * torch.sin(2 * np.pi * 20 * t)
    signal = signal.unsqueeze(0).unsqueeze(0).repeat(B, C, 1)
    return signal


@pytest.fixture
def sample_prediction(sample_signal):
    """Create sample prediction (signal + noise)."""
    noise = 0.1 * torch.randn_like(sample_signal)
    return sample_signal + noise


# ============================================================================
# BSP Loss Tests
# ============================================================================

def test_bsp_loss_initialization():
    """Test BinnedSpectralLoss initialization."""
    # Default parameters
    loss = BinnedSpectralLoss()
    assert loss.n_bins == 32
    assert loss.lambda_bsp == 1.0
    assert loss.epsilon == 1e-8
    assert loss.binning_mode == 'linear'

    # Custom parameters
    loss = BinnedSpectralLoss(n_bins=16, lambda_bsp=0.5, binning_mode='log')
    assert loss.n_bins == 16
    assert loss.lambda_bsp == 0.5
    assert loss.binning_mode == 'log'

    # Invalid binning mode
    with pytest.raises(ValueError):
        BinnedSpectralLoss(binning_mode='invalid')


def test_bsp_loss_forward_shape(sample_prediction, sample_signal):
    """Test BSP loss output is scalar."""
    loss_fn = BinnedSpectralLoss(n_bins=32)
    loss = loss_fn(sample_prediction, sample_signal)

    # Output should be scalar
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_bsp_loss_identical_inputs():
    """Test BSP loss is zero for identical inputs."""
    signal = torch.randn(4, 1, 1000)
    loss_fn = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)
    loss = loss_fn(signal, signal)

    # Should be very close to zero (not exactly due to numerical precision)
    assert loss.item() < 1e-6, f"Loss should be near zero, got {loss.item()}"


def test_bsp_loss_gradient_flow(sample_prediction, sample_signal):
    """Test gradients flow through BSP loss."""
    # Make prediction require gradients
    pred = sample_prediction.clone().requires_grad_(True)
    target = sample_signal.clone()

    loss_fn = BinnedSpectralLoss(n_bins=32)
    loss = loss_fn(pred, target)

    # Backpropagate
    loss.backward()

    # Check gradients exist and are non-zero
    assert pred.grad is not None, "Gradients should exist"
    assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad)), \
        "Gradients should be non-zero"


def test_bsp_loss_linear_vs_log_binning(sample_prediction, sample_signal):
    """Test both binning modes produce valid losses."""
    loss_linear = BinnedSpectralLoss(n_bins=32, binning_mode='linear')
    loss_log = BinnedSpectralLoss(n_bins=32, binning_mode='log')

    loss_val_linear = loss_linear(sample_prediction, sample_signal)
    loss_val_log = loss_log(sample_prediction, sample_signal)

    # Both should be positive
    assert loss_val_linear.item() > 0
    assert loss_val_log.item() > 0

    # They should be different (different binning strategies)
    assert not torch.allclose(loss_val_linear, loss_val_log)


def test_bsp_loss_bin_energy_shape():
    """Test binning produces correct output shape."""
    loss_fn = BinnedSpectralLoss(n_bins=16)

    B, C, T = 4, 1, 1000
    signal = torch.randn(B, C, T)

    # Compute FFT manually
    fft = torch.fft.rfft(signal, dim=-1)
    energy = torch.abs(fft) ** 2  # [B, C, T//2+1]

    # Bin energy
    binned = loss_fn._bin_energy_1d(energy, T)

    # Should have shape [B, C, n_bins]
    assert binned.shape == (B, C, 16), f"Expected (4, 1, 16), got {binned.shape}"


def test_bsp_loss_compute_bin_errors(sample_prediction, sample_signal):
    """Test per-bin error computation."""
    loss_fn = BinnedSpectralLoss(n_bins=32)
    bin_errors = loss_fn.compute_bin_errors(sample_prediction, sample_signal)

    # Should have shape [n_bins]
    assert bin_errors.shape == (32,), f"Expected (32,), got {bin_errors.shape}"

    # All errors should be non-negative
    assert torch.all(bin_errors >= 0), "Bin errors should be non-negative"


def test_bsp_loss_frequency_analysis(sample_prediction, sample_signal):
    """Test frequency analysis helper."""
    loss_fn = BinnedSpectralLoss(n_bins=32)
    analysis = loss_fn.get_frequency_analysis(sample_prediction, sample_signal)

    # Check all expected keys
    assert 'bin_edges' in analysis
    assert 'bin_centers' in analysis
    assert 'pred_spectrum' in analysis
    assert 'target_spectrum' in analysis
    assert 'relative_errors' in analysis
    assert 'loss' in analysis

    # Check shapes
    assert len(analysis['bin_edges']) == 33  # n_bins + 1
    assert len(analysis['bin_centers']) == 32
    assert len(analysis['pred_spectrum']) == 32
    assert len(analysis['target_spectrum']) == 32
    assert len(analysis['relative_errors']) == 32
    assert isinstance(analysis['loss'], float)


def test_bsp_loss_lambda_weighting(sample_prediction, sample_signal):
    """Test lambda_bsp weight is applied correctly."""
    loss_1 = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)
    loss_2 = BinnedSpectralLoss(n_bins=32, lambda_bsp=2.0)

    val_1 = loss_1(sample_prediction, sample_signal)
    val_2 = loss_2(sample_prediction, sample_signal)

    # loss_2 should be exactly 2× loss_1
    assert torch.allclose(val_2, 2.0 * val_1, rtol=1e-5), \
        f"Expected {2.0 * val_1.item()}, got {val_2.item()}"


# ============================================================================
# Combined Loss Tests
# ============================================================================

def test_combined_loss_initialization():
    """Test CombinedLoss initialization."""
    base = RelativeL2Loss()
    spectral = BinnedSpectralLoss(n_bins=32)

    combined = CombinedLoss(base, spectral, lambda_spectral=1.0)

    assert combined.base_loss is base
    assert combined.spectral_loss is spectral
    assert combined.lambda_spectral == 1.0


def test_combined_loss_forward(sample_prediction, sample_signal):
    """Test combined loss computation."""
    base = RelativeL2Loss()
    spectral = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)
    combined = CombinedLoss(base, spectral, lambda_spectral=1.0)

    # Compute combined loss
    total_loss = combined(sample_prediction, sample_signal)

    # Compute components separately
    base_loss = base(sample_prediction, sample_signal)
    spectral_loss = spectral(sample_prediction, sample_signal)
    expected_total = base_loss + 1.0 * spectral_loss

    # Should match
    assert torch.allclose(total_loss, expected_total, rtol=1e-5), \
        f"Expected {expected_total.item()}, got {total_loss.item()}"


def test_combined_loss_components(sample_prediction, sample_signal):
    """Test get_loss_components method."""
    base = RelativeL2Loss()
    spectral = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)
    combined = CombinedLoss(base, spectral, lambda_spectral=2.0)

    components = combined.get_loss_components(sample_prediction, sample_signal)

    # Check keys
    assert 'base' in components
    assert 'spectral' in components
    assert 'total' in components

    # Check total = base + λ × spectral
    expected_total = components['base'] + 2.0 * components['spectral']
    assert torch.allclose(components['total'], expected_total, rtol=1e-5)


def test_combined_loss_gradient_flow(sample_prediction, sample_signal):
    """Test gradients flow through combined loss."""
    pred = sample_prediction.clone().requires_grad_(True)

    base = RelativeL2Loss()
    spectral = BinnedSpectralLoss(n_bins=32)
    combined = CombinedLoss(base, spectral, lambda_spectral=1.0)

    loss = combined(pred, sample_signal)
    loss.backward()

    assert pred.grad is not None
    assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))


# ============================================================================
# Loss Factory Tests
# ============================================================================

def test_loss_factory_relative_l2():
    """Test factory creates RelativeL2Loss."""
    config = LossConfig(loss_type='relative_l2', loss_params={})
    loss = create_loss(config)

    assert isinstance(loss, RelativeL2Loss)


def test_loss_factory_bsp():
    """Test factory creates BinnedSpectralLoss."""
    config = LossConfig(
        loss_type='bsp',
        loss_params={'n_bins': 16, 'lambda_bsp': 2.0}
    )
    loss = create_loss(config)

    assert isinstance(loss, BinnedSpectralLoss)
    assert loss.n_bins == 16
    assert loss.lambda_bsp == 2.0


def test_loss_factory_combined():
    """Test factory creates CombinedLoss."""
    config = LossConfig(
        loss_type='combined',
        loss_params={
            'base_loss': 'relative_l2',
            'spectral_loss': 'bsp',
            'lambda_spectral': 1.5,
            'n_bins': 32
        }
    )
    loss = create_loss(config)

    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.base_loss, RelativeL2Loss)
    assert isinstance(loss.spectral_loss, BinnedSpectralLoss)
    assert loss.lambda_spectral == 1.5


def test_loss_factory_from_dict(sample_prediction, sample_signal):
    """Test creating loss from dictionary."""
    config_dict = {
        'loss_type': 'combined',
        'loss_params': {
            'base_loss': 'relative_l2',
            'spectral_loss': 'bsp',
            'lambda_spectral': 1.0,
            'n_bins': 32
        }
    }

    loss = create_loss_from_dict(config_dict)
    assert isinstance(loss, CombinedLoss)

    # Test it works
    loss_val = loss(sample_prediction, sample_signal)
    assert loss_val.item() > 0


def test_loss_factory_predefined_configs(sample_prediction, sample_signal):
    """Test predefined config constants."""
    # Baseline config
    loss_baseline = create_loss(BASELINE_CONFIG)
    assert isinstance(loss_baseline, RelativeL2Loss)

    # BSP config
    loss_bsp = create_loss(BSP_CONFIG)
    assert isinstance(loss_bsp, CombinedLoss)

    # Both should work
    val_baseline = loss_baseline(sample_prediction, sample_signal)
    val_bsp = loss_bsp(sample_prediction, sample_signal)
    assert val_baseline.item() > 0
    assert val_bsp.item() > 0


def test_loss_factory_invalid_type():
    """Test factory raises error for invalid loss type."""
    # Should raise error during config creation (not factory call)
    with pytest.raises(ValueError, match="Invalid loss_type"):
        config = LossConfig(loss_type='invalid_type', loss_params={})


# ============================================================================
# SA-BSP Loss Tests
# ============================================================================

def test_adaptive_weights_initialization():
    """Test SelfAdaptiveWeights initialization."""
    # Per-bin mode
    weights = SelfAdaptiveWeights(n_components=32, mode='per-bin', init_value=1.0)
    assert weights.n_components == 32
    assert weights.mode == 'per-bin'
    assert isinstance(weights.log_weights, nn.Parameter)

    # Check initial values
    w = weights()
    assert w.shape == (32,)
    assert torch.allclose(w, torch.ones(32), rtol=0.1)

    # Global mode
    weights = SelfAdaptiveWeights(n_components=1, mode='global')
    assert weights.n_components == 1

    # None mode (fixed weights)
    weights = SelfAdaptiveWeights(n_components=32, mode='none')
    assert not isinstance(weights.log_weights, nn.Parameter)  # Should be buffer, not parameter


def test_adaptive_weights_forward():
    """Test adaptive weights are always positive."""
    weights = SelfAdaptiveWeights(n_components=32, mode='per-bin', init_value=1.0)

    # Get weights
    w = weights()
    assert torch.all(w > 0), "All weights should be positive"

    # Modify log_weights to test positivity
    with torch.no_grad():
        weights.log_weights.data = torch.randn(32) * 2  # Random values

    w = weights()
    assert torch.all(w > 0), "Weights should still be positive after modification"


def test_adaptive_weights_gradient_flow():
    """Test gradients flow through adaptive weights."""
    weights = SelfAdaptiveWeights(n_components=32, mode='per-bin')

    # Create dummy loss
    w = weights()
    loss = w.sum()

    # Backprop
    loss.backward()

    # Check gradients
    assert weights.log_weights.grad is not None
    assert not torch.allclose(weights.log_weights.grad, torch.zeros_like(weights.log_weights.grad))


def test_adaptive_weights_statistics():
    """Test weight statistics helper."""
    weights = SelfAdaptiveWeights(n_components=32, mode='per-bin', init_value=2.0)
    stats = weights.get_statistics()

    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'weights' in stats

    # Check values make sense
    assert stats['mean'] > 0
    assert stats['min'] > 0
    assert stats['max'] >= stats['min']
    assert len(stats['weights']) == 32


def test_sa_bsp_loss_initialization():
    """Test SelfAdaptiveBSPLoss initialization."""
    loss = SelfAdaptiveBSPLoss(n_bins=32, lambda_sa=1.0, adapt_mode='per-bin')

    assert loss.n_bins == 32
    assert loss.lambda_sa == 1.0
    assert loss.adapt_mode == 'per-bin'
    assert isinstance(loss.bsp_module, BinnedSpectralLoss)
    assert isinstance(loss.adaptive_weights, SelfAdaptiveWeights)


def test_sa_bsp_loss_forward(sample_prediction, sample_signal):
    """Test SA-BSP loss forward pass."""
    loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')
    loss = loss_fn(sample_prediction, sample_signal)

    # Should be scalar
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_sa_bsp_loss_gradient_flow(sample_prediction, sample_signal):
    """Test gradients flow through SA-BSP loss."""
    # Create model parameters
    pred = sample_prediction.clone().requires_grad_(True)

    # Create loss
    loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')

    # Forward
    loss = loss_fn(pred, sample_signal)

    # Backward
    loss.backward()

    # Check prediction gradients
    assert pred.grad is not None
    assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

    # Check weight gradients
    assert loss_fn.adaptive_weights.log_weights.grad is not None


def test_sa_bsp_loss_adaptation_modes(sample_prediction, sample_signal):
    """Test different adaptation modes."""
    modes = ['per-bin', 'global', 'both', 'none']

    for mode in modes:
        loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode=mode)
        loss = loss_fn(sample_prediction, sample_signal)

        assert torch.isfinite(loss), f"Loss should be finite for mode={mode}"
        assert loss.item() >= 0, f"Loss should be non-negative for mode={mode}"


def test_sa_bsp_loss_weight_updates(sample_prediction, sample_signal):
    """Test that adaptive weights actually update during optimization."""
    loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')
    weight_optimizer = torch.optim.SGD(loss_fn.adaptive_weights.parameters(), lr=0.1)

    # Get initial weights
    initial_weights = loss_fn.adaptive_weights().clone()

    # Train for a few steps
    for _ in range(5):
        weight_optimizer.zero_grad()
        loss = loss_fn(sample_prediction, sample_signal)
        loss.backward()
        weight_optimizer.step()

    # Get final weights
    final_weights = loss_fn.adaptive_weights()

    # Weights should have changed
    assert not torch.allclose(initial_weights, final_weights, rtol=1e-5), \
        "Adaptive weights should change during optimization"


def test_sa_bsp_loss_vs_bsp(sample_prediction, sample_signal):
    """Test SA-BSP degenerates to BSP when mode='none'."""
    # SA-BSP with mode='none' should behave like BSP
    sa_bsp_loss = SelfAdaptiveBSPLoss(
        n_bins=32,
        lambda_sa=1.0,
        adapt_mode='none',
        init_weight=1.0
    )

    # Regular BSP
    bsp_loss = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)

    # Should produce similar results (not exact due to implementation details)
    sa_val = sa_bsp_loss(sample_prediction, sample_signal)
    bsp_val = bsp_loss(sample_prediction, sample_signal)

    assert torch.allclose(sa_val, bsp_val, rtol=0.1), \
        "SA-BSP with mode='none' should be similar to BSP"


def test_sa_bsp_loss_components(sample_prediction, sample_signal):
    """Test get_loss_components method."""
    loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')
    components = loss_fn.get_loss_components(sample_prediction, sample_signal)

    # Check all keys exist
    assert 'bin_errors' in components
    assert 'weights' in components
    assert 'weighted_errors' in components
    assert 'total_loss' in components
    assert 'weight_stats' in components

    # Check shapes
    assert len(components['bin_errors']) == 32
    assert len(components['weights']) == 32
    assert len(components['weighted_errors']) == 32
    assert isinstance(components['total_loss'], float)


def test_sa_bsp_loss_frequency_analysis(sample_prediction, sample_signal):
    """Test frequency analysis with adaptive weights."""
    loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')
    analysis = loss_fn.get_frequency_analysis(sample_prediction, sample_signal)

    # Check BSP fields
    assert 'bin_edges' in analysis
    assert 'bin_centers' in analysis
    assert 'pred_spectrum' in analysis
    assert 'target_spectrum' in analysis

    # Check SA-BSP specific fields
    assert 'adaptive_weights' in analysis
    assert 'weight_stats' in analysis
    assert 'adapt_mode' in analysis

    assert len(analysis['adaptive_weights']) == 32


def test_create_optimizers_helper(sample_prediction, sample_signal):
    """Test helper function for creating optimizers."""
    # Create simple model
    model = nn.Linear(1000, 1000)

    # Create SA-BSP loss
    loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')

    # Create optimizers
    model_opt, weight_opt = create_optimizers_for_sa_bsp(
        model, loss_fn, model_lr=1e-3, weight_lr=1e-3
    )

    # Check optimizers exist
    assert model_opt is not None
    assert weight_opt is not None

    # Test with mode='none' (no trainable weights)
    loss_fn_none = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='none')
    model_opt, weight_opt = create_optimizers_for_sa_bsp(model, loss_fn_none)

    assert model_opt is not None
    assert weight_opt is None  # No trainable weights


def test_loss_factory_sa_bsp():
    """Test factory creates SA-BSP loss."""
    config = LossConfig(
        loss_type='sa_bsp',
        loss_params={
            'n_bins': 16,
            'lambda_sa': 2.0,
            'adapt_mode': 'per-bin'
        }
    )
    loss = create_loss(config)

    assert isinstance(loss, SelfAdaptiveBSPLoss)
    assert loss.n_bins == 16
    assert loss.lambda_sa == 2.0
    assert loss.adapt_mode == 'per-bin'


def test_loss_factory_combined_with_sa_bsp(sample_prediction, sample_signal):
    """Test factory creates combined loss with SA-BSP."""
    config = LossConfig(
        loss_type='combined',
        loss_params={
            'base_loss': 'relative_l2',
            'spectral_loss': 'sa_bsp',
            'lambda_spectral': 1.0,
            'n_bins': 32,
            'adapt_mode': 'per-bin'
        }
    )
    loss = create_loss(config)

    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.base_loss, RelativeL2Loss)
    assert isinstance(loss.spectral_loss, SelfAdaptiveBSPLoss)

    # Test it works
    loss_val = loss(sample_prediction, sample_signal)
    assert torch.isfinite(loss_val)


def test_sa_bsp_config_constant(sample_prediction, sample_signal):
    """Test predefined SA-BSP config."""
    loss = create_loss(SA_BSP_CONFIG)
    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.spectral_loss, SelfAdaptiveBSPLoss)

    # Test it works
    loss_val = loss(sample_prediction, sample_signal)
    assert loss_val.item() > 0


# ============================================================================
# Edge Cases and Robustness
# ============================================================================

def test_bsp_loss_with_zeros():
    """Test BSP loss handles zero signals gracefully."""
    pred = torch.zeros(4, 1, 1000)
    target = torch.randn(4, 1, 1000)

    loss_fn = BinnedSpectralLoss(n_bins=32)
    loss = loss_fn(pred, target)

    # Should not be NaN or Inf
    assert torch.isfinite(loss), "Loss should be finite with zero prediction"


def test_bsp_loss_batch_size_one():
    """Test BSP loss works with batch size 1."""
    pred = torch.randn(1, 1, 1000)
    target = torch.randn(1, 1, 1000)

    loss_fn = BinnedSpectralLoss(n_bins=32)
    loss = loss_fn(pred, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_bsp_loss_different_lengths():
    """Test BSP loss works with different temporal lengths."""
    for T in [500, 1000, 2000, 4000]:
        pred = torch.randn(4, 1, T)
        target = torch.randn(4, 1, T)

        loss_fn = BinnedSpectralLoss(n_bins=32)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss), f"Loss should be finite for T={T}"


def test_bsp_loss_multi_channel():
    """Test BSP loss works with multiple channels."""
    B, C, T = 4, 3, 1000  # 3 channels
    pred = torch.randn(B, C, T)
    target = torch.randn(B, C, T)

    loss_fn = BinnedSpectralLoss(n_bins=32)
    loss = loss_fn(pred, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_bsp_loss_training_simulation():
    """Simulate a mini training loop with BSP loss."""
    # Create simple model (linear layer)
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)

    # Generate data
    input_signal = torch.randn(8, 1, 1000)
    target_signal = torch.randn(8, 1, 1000)

    initial_loss = None
    for i in range(5):
        optimizer.zero_grad()

        # Forward pass (reshape for linear layer)
        pred = model(input_signal.squeeze(1)).unsqueeze(1)

        # Compute loss
        loss = loss_fn(pred, target_signal)

        # Backward pass
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()

    # Loss should have changed (model is learning)
    final_loss = loss.item()
    assert final_loss != initial_loss, "Loss should change during training"


def test_combined_loss_training_simulation():
    """Simulate training with combined loss."""
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    base_loss = RelativeL2Loss()
    spectral_loss = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)
    combined_loss = CombinedLoss(base_loss, spectral_loss, lambda_spectral=1.0)

    input_signal = torch.randn(8, 1, 1000)
    target_signal = torch.randn(8, 1, 1000)

    for i in range(5):
        optimizer.zero_grad()
        pred = model(input_signal.squeeze(1)).unsqueeze(1)
        loss = combined_loss(pred, target_signal)
        loss.backward()
        optimizer.step()

        # Get components for logging
        if i == 0:
            components = combined_loss.get_loss_components(pred.detach(), target_signal)
            assert 'base' in components
            assert 'spectral' in components
            assert 'total' in components
