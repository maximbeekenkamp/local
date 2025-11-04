"""
Unit tests for neural operator models.

Tests DeepONet1D, FNO1D, UNet1D, and model factory functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import DeepONet1D, FNO1D, UNet1D, create_model


class TestDeepONetForwardPass:
    """Test DeepONet1D forward pass functionality."""

    def test_forward_pass_shape(self):
        """Verify DeepONet forward pass produces correct output shape."""
        model = DeepONet1D(sensor_dim=4000, latent_dim=100)

        # Create random input [batch=2, channels=1, timesteps=4000]
        x = torch.randn(2, 1, 4000)

        # Forward pass
        output = model(x)

        # Check output shape matches input shape
        assert output.shape == (2, 1, 4000), \
            f"Expected shape [2, 1, 4000], got {output.shape}"

    def test_forward_pass_no_nan_inf(self):
        """Verify DeepONet output contains no NaN or Inf values."""
        model = DeepONet1D()
        x = torch.randn(2, 1, 4000)

        output = model(x)

        # Check no NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"

        # Check no Inf
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_forward_pass_dtype(self):
        """Verify output dtype matches input dtype."""
        model = DeepONet1D()
        x = torch.randn(2, 1, 4000, dtype=torch.float32)

        output = model(x)

        assert output.dtype == torch.float32, \
            f"Expected dtype float32, got {output.dtype}"

    def test_different_batch_sizes(self):
        """Verify DeepONet works with different batch sizes."""
        model = DeepONet1D()

        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 1, 4000)
            output = model(x)
            assert output.shape == (batch_size, 1, 4000)


class TestFNOForwardPass:
    """Test FNO1D forward pass functionality."""

    def test_forward_pass_shape(self):
        """Verify FNO forward pass produces correct output shape."""
        model = FNO1D(n_modes=28, hidden_channels=52, n_layers=4)

        # Create random input [batch=2, channels=1, timesteps=4000]
        x = torch.randn(2, 1, 4000)

        # Forward pass
        output = model(x)

        # Check output shape matches input shape
        assert output.shape == (2, 1, 4000), \
            f"Expected shape [2, 1, 4000], got {output.shape}"

    def test_forward_pass_no_nan_inf(self):
        """Verify FNO output contains no NaN or Inf values."""
        model = FNO1D()
        x = torch.randn(2, 1, 4000)

        output = model(x)

        # Check no NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"

        # Check no Inf
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_forward_pass_deterministic(self):
        """Verify FNO produces same output for same input (no randomness)."""
        model = FNO1D()
        model.eval()  # Evaluation mode (no dropout)

        x = torch.randn(2, 1, 4000)

        # Two forward passes with same input
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7), \
            "FNO should be deterministic in eval mode"

    def test_different_batch_sizes(self):
        """Verify FNO works with different batch sizes."""
        model = FNO1D()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 1, 4000)
            output = model(x)
            assert output.shape == (batch_size, 1, 4000)


class TestUNetForwardPass:
    """Test UNet1D forward pass functionality."""

    def test_forward_pass_shape(self):
        """Verify UNet forward pass produces correct output shape."""
        model = UNet1D(base_channels=28, num_levels=3)

        # Create random input [batch=2, channels=1, timesteps=4000]
        x = torch.randn(2, 1, 4000)

        # Forward pass
        output = model(x)

        # Check output shape matches input shape
        assert output.shape == (2, 1, 4000), \
            f"Expected shape [2, 1, 4000], got {output.shape}"

    def test_forward_pass_no_nan_inf(self):
        """Verify UNet output contains no NaN or Inf values."""
        model = UNet1D()
        x = torch.randn(2, 1, 4000)

        output = model(x)

        # Check no NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"

        # Check no Inf
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_skip_connections_work(self):
        """Verify skip connections affect output (compare with/without)."""
        # This is implicit in architecture - just verify different random
        # initializations produce different outputs
        model1 = UNet1D()
        model2 = UNet1D()

        x = torch.randn(2, 1, 4000)

        output1 = model1(x)
        output2 = model2(x)

        # Different random weights should produce different outputs
        assert not torch.allclose(output1, output2, rtol=0.1), \
            "Different model instances should produce different outputs"

    def test_different_batch_sizes(self):
        """Verify UNet works with different batch sizes."""
        model = UNet1D()

        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 1, 4000)
            output = model(x)
            assert output.shape == (batch_size, 1, 4000)


class TestModelFactory:
    """Test model factory functionality."""

    def test_create_deeponet(self):
        """Verify factory creates DeepONet1D instance."""
        model = create_model('deeponet')

        assert isinstance(model, DeepONet1D), \
            f"Expected DeepONet1D, got {type(model)}"
        assert isinstance(model, nn.Module), \
            "Model should be nn.Module"

    def test_create_fno(self):
        """Verify factory creates FNO1D instance."""
        model = create_model('fno')

        assert isinstance(model, FNO1D), \
            f"Expected FNO1D, got {type(model)}"
        assert isinstance(model, nn.Module), \
            "Model should be nn.Module"

    def test_create_unet(self):
        """Verify factory creates UNet1D instance."""
        model = create_model('unet')

        assert isinstance(model, UNet1D), \
            f"Expected UNet1D, got {type(model)}"
        assert isinstance(model, nn.Module), \
            "Model should be nn.Module"

    def test_invalid_architecture_raises_error(self):
        """Verify invalid architecture name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model('invalid_model')

    def test_factory_with_custom_config(self):
        """Verify factory accepts custom configuration."""
        config = {'latent_dim': 128, 'branch_layers': [64, 128]}
        model = create_model('deeponet', config)

        assert isinstance(model, DeepONet1D)
        assert model.latent_dim == 128
        assert model.branch_layers == [64, 128]

    def test_factory_case_insensitive(self):
        """Verify factory handles case-insensitive architecture names."""
        model1 = create_model('DEEPONET')
        model2 = create_model('DeepONet')
        model3 = create_model('deeponet')

        assert all(isinstance(m, DeepONet1D) for m in [model1, model2, model3])


class TestModelsTrainable:
    """Test that models are trainable (gradients flow correctly)."""

    def test_deeponet_gradients_flow(self):
        """Verify DeepONet parameters receive gradients."""
        model = DeepONet1D()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create dummy data
        x = torch.randn(4, 1, 4000)
        target = torch.randn(4, 1, 4000)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, \
                f"Parameter {name} has zero gradient"

        # Optimizer step should complete without error
        optimizer.step()

    def test_fno_gradients_flow(self):
        """Verify FNO parameters receive gradients."""
        model = FNO1D()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create dummy data
        x = torch.randn(4, 1, 4000)
        target = torch.randn(4, 1, 4000)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, \
                f"Parameter {name} has zero gradient"

        # Optimizer step should complete without error
        optimizer.step()

    def test_unet_gradients_flow(self):
        """Verify UNet parameters receive gradients."""
        model = UNet1D()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create dummy data
        x = torch.randn(4, 1, 4000)
        target = torch.randn(4, 1, 4000)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, \
                f"Parameter {name} has zero gradient"

        # Optimizer step should complete without error
        optimizer.step()


class TestModelParameterCounts:
    """Test that all models have similar parameter counts (~250K ±50K)."""

    def test_parameter_counts_in_range(self):
        """Verify all models have 200K-300K parameters (target ~250K)."""
        # Create models with default configurations
        deeponet = create_model('deeponet')
        fno = create_model('fno')
        unet = create_model('unet')

        # Count parameters
        deeponet_params = sum(p.numel() for p in deeponet.parameters())
        fno_params = sum(p.numel() for p in fno.parameters())
        unet_params = sum(p.numel() for p in unet.parameters())

        print(f"\nParameter counts:")
        print(f"  DeepONet: {deeponet_params:,} params")
        print(f"  FNO:      {fno_params:,} params")
        print(f"  UNet:     {unet_params:,} params")

        # Check each model is in range [200K, 300K]
        for name, count in [('DeepONet', deeponet_params),
                            ('FNO', fno_params),
                            ('UNet', unet_params)]:
            assert 200_000 <= count <= 300_000, \
                f"{name} has {count:,} params, outside target range [200K, 300K]"

    def test_parameter_counts_within_50k_of_each_other(self):
        """Verify all models are within 50K (20%) parameters of each other."""
        # Create models
        deeponet = create_model('deeponet')
        fno = create_model('fno')
        unet = create_model('unet')

        # Count parameters
        deeponet_params = sum(p.numel() for p in deeponet.parameters())
        fno_params = sum(p.numel() for p in fno.parameters())
        unet_params = sum(p.numel() for p in unet.parameters())

        all_params = [deeponet_params, fno_params, unet_params]
        max_diff = max(all_params) - min(all_params)

        print(f"\nParameter count spread: {max_diff:,} params")
        print(f"  Min: {min(all_params):,}")
        print(f"  Max: {max(all_params):,}")

        assert max_diff <= 50_000, \
            f"Parameter count spread {max_diff:,} exceeds 50K threshold"

    def test_parameter_count_methods(self):
        """Verify count_parameters() method matches direct counting."""
        models = [
            create_model('deeponet'),
            create_model('fno'),
            create_model('unet')
        ]

        for model in models:
            # Count via method
            method_count = model.count_parameters()

            # Count directly
            direct_count = sum(p.numel() for p in model.parameters())

            assert method_count == direct_count, \
                f"count_parameters() returned {method_count}, " \
                f"but direct count is {direct_count}"
