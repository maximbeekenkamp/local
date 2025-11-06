"""
Model factory for creating neural operator models.

Provides centralized model instantiation for DeepONet, FNO, and UNet architectures.
"""

import torch.nn as nn
from typing import Optional, Dict, Any

from .deeponet_1d import DeepONet1D
from .fno_1d import FNO1D
from .unet_1d import UNet1D


def create_model(arch: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Factory function to create neural operator models.

    Args:
        arch: Model architecture name. Options:
              - 'deeponet': DeepONet with branch-trunk architecture
              - 'fno': Fourier Neural Operator
              - 'unet': UNet encoder-decoder
        config: Optional configuration dictionary with model hyperparameters.
                If None, uses default hyperparameters.

    Returns:
        Initialized nn.Module of the specified architecture

    Raises:
        ValueError: If arch is not recognized

    Example:
        >>> # Create model with default hyperparameters
        >>> model = create_model('deeponet')
        >>>
        >>> # Create model with custom hyperparameters
        >>> config = {'latent_dim': 128, 'branch_layers': [64, 128]}
        >>> model = create_model('deeponet', config)
    """
    arch = arch.lower()
    config = config or {}

    if arch == 'deeponet':
        return _create_deeponet(config)
    elif arch == 'fno':
        return _create_fno(config)
    elif arch == 'unet':
        return _create_unet(config)
    else:
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            f"Valid options: 'deeponet', 'fno', 'unet'"
        )


def _create_deeponet(config: Dict[str, Any]) -> DeepONet1D:
    """
    Create DeepONet1D model with configuration.

    Args:
        config: Configuration dictionary. Supported keys:
                - sensor_dim: Input dimension (default 4000)
                - latent_dim: Latent space dimension (default 100)
                - branch_layers: Branch network hidden layers (default [50, 100])
                - trunk_layers: Trunk network hidden layers (default [100, 100])
                - activation: Activation function ('tanh', 'relu', 'siren', default 'siren')

    Returns:
        Initialized DeepONet1D model
    """
    # Default hyperparameters (target ~235K params)
    defaults = {
        'sensor_dim': 4000,
        'latent_dim': 100,
        'branch_layers': [50, 100],
        'trunk_layers': [100, 100],
        'activation': 'siren'  # Default to SIREN activation
    }

    # Merge with user config
    merged_config = {**defaults, **config}

    return DeepONet1D(**merged_config)


def _create_fno(config: Dict[str, Any]) -> FNO1D:
    """
    Create FNO1D model with configuration.

    Args:
        config: Configuration dictionary. Supported keys:
                - n_modes: Number of Fourier modes (default 28)
                - hidden_channels: Hidden channel dimension (default 52)
                - n_layers: Number of FNO layers (default 4)
                - in_channels: Input channels (default 1)
                - out_channels: Output channels (default 1)

    Returns:
        Initialized FNO1D model
    """
    # Default hyperparameters (target ~250K params)
    defaults = {
        'n_modes': 28,
        'hidden_channels': 60,
        'n_layers': 4,
        'in_channels': 1,
        'out_channels': 1
    }

    # Merge with user config
    merged_config = {**defaults, **config}

    return FNO1D(**merged_config)


def _create_unet(config: Dict[str, Any]) -> UNet1D:
    """
    Create UNet1D model with configuration.

    Args:
        config: Configuration dictionary. Supported keys:
                - in_channels: Input channels (default 1)
                - out_channels: Output channels (default 1)
                - base_channels: Base channel count (default 40)
                - num_levels: Encoder/decoder depth (default 3)
                - kernel_size: Convolution kernel size (default 3)
                - num_groups: GroupNorm groups (default 4)

    Returns:
        Initialized UNet1D model
    """
    # Default hyperparameters (target ~250K params)
    defaults = {
        'in_channels': 1,
        'out_channels': 1,
        'base_channels': 40,
        'num_levels': 3,
        'kernel_size': 3,
        'num_groups': 4
    }

    # Merge with user config
    merged_config = {**defaults, **config}

    return UNet1D(**merged_config)


def list_available_models() -> list:
    """
    Get list of available model architectures.

    Returns:
        List of model architecture names
    """
    return ['deeponet', 'fno', 'unet']


def get_model_info(arch: str) -> Dict[str, Any]:
    """
    Get information about a model architecture.

    Args:
        arch: Model architecture name

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If arch is not recognized
    """
    arch = arch.lower()

    info = {
        'deeponet': {
            'name': 'DeepONet1D',
            'description': 'Branch-trunk architecture for operator learning',
            'params_target': '~235K',
            'default_config': {
                'sensor_dim': 4000,
                'latent_dim': 100,
                'branch_layers': [50, 100],
                'trunk_layers': [100, 100]
            }
        },
        'fno': {
            'name': 'FNO1D',
            'description': 'Fourier Neural Operator (spectral methods)',
            'params_target': '~250K',
            'default_config': {
                'n_modes': 28,
                'hidden_channels': 52,
                'n_layers': 4,
                'in_channels': 1,
                'out_channels': 1
            }
        },
        'unet': {
            'name': 'UNet1D',
            'description': 'Encoder-decoder with skip connections',
            'params_target': '~250K',
            'default_config': {
                'in_channels': 1,
                'out_channels': 1,
                'base_channels': 28,
                'num_levels': 3,
                'kernel_size': 3,
                'num_groups': 4
            }
        }
    }

    if arch not in info:
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            f"Valid options: {list(info.keys())}"
        )

    return info[arch]
