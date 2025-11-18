"""
Configuration dataclasses for neural operator models.

Provides structured configuration for DeepONet, FNO, and UNet with
serialization support (JSON save/load).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json
import os


@dataclass
class DeepONetConfig:
    """
    Configuration for DeepONet1D model (matches reference CausalityDeepONet).

    Target parameter count: ~567K

    Attributes:
        sensor_dim: Input dimension (number of timesteps)
        latent_dim: Dimension of latent space for branch-trunk combination
        branch_layers: Hidden layer sizes for branch network
        trunk_layers: Hidden layer sizes for trunk network
        activation: Activation function type ('requ', 'tanh', 'relu', 'siren')
                   'requ' (ReLU²) is the reference implementation default
    """

    sensor_dim: int = 4000
    latent_dim: int = 120  # Reference: 120 (matches CausalityDeepONet)
    branch_layers: List[int] = field(default_factory=lambda: [120, 120, 120])  # Reference: 3 layers of 120
    trunk_layers: List[int] = field(default_factory=lambda: [120, 120, 120])   # Reference: 3 layers of 120
    activation: str = 'requ'  # Reference default: ReQU (ReLU squared)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DeepONetConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            DeepONetConfig instance
        """
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output JSON file path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DeepONetConfig':
        """
        Load configuration from JSON file.

        Args:
            path: Input JSON file path

        Returns:
            DeepONetConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"DeepONetConfig(\n"
            f"  sensor_dim={self.sensor_dim},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  branch_layers={self.branch_layers},\n"
            f"  trunk_layers={self.trunk_layers},\n"
            f"  activation='{self.activation}'\n"
            f")"
        )


@dataclass
class FNOConfig:
    """
    Configuration for FNO1D model.

    Target parameter count: ~500K (matched to DeepONet)

    Attributes:
        n_modes: Number of Fourier modes to keep (low-frequency)
        hidden_channels: Hidden channel dimension
        n_layers: Number of FNO layers
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    n_modes: int = 32
    hidden_channels: int = 80  # Increased from 60 to match ~500K params
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FNOConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            FNOConfig instance
        """
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output JSON file path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FNOConfig':
        """
        Load configuration from JSON file.

        Args:
            path: Input JSON file path

        Returns:
            FNOConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"FNOConfig(\n"
            f"  n_modes={self.n_modes},\n"
            f"  hidden_channels={self.hidden_channels},\n"
            f"  n_layers={self.n_layers},\n"
            f"  in_channels={self.in_channels},\n"
            f"  out_channels={self.out_channels}\n"
            f")"
        )


@dataclass
class UNetConfig:
    """
    Configuration for UNet1D model.

    Target parameter count: ~500K (matched to DeepONet)

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base channel count (doubled at each level)
        num_levels: Number of encoder/decoder levels
        kernel_size: Convolution kernel size
        num_groups: Number of groups for GroupNorm
    """

    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 58  # Increased from 40 to match ~500K params
    num_levels: int = 3
    kernel_size: int = 3
    num_groups: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UNetConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            UNetConfig instance
        """
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output JSON file path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'UNetConfig':
        """
        Load configuration from JSON file.

        Args:
            path: Input JSON file path

        Returns:
            UNetConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"UNetConfig(\n"
            f"  in_channels={self.in_channels},\n"
            f"  out_channels={self.out_channels},\n"
            f"  base_channels={self.base_channels},\n"
            f"  num_levels={self.num_levels},\n"
            f"  kernel_size={self.kernel_size},\n"
            f"  num_groups={self.num_groups}\n"
            f")"
        )


# Type alias for any model config
ModelConfig = DeepONetConfig | FNOConfig | UNetConfig


def create_config(arch: str, **kwargs) -> ModelConfig:
    """
    Factory function to create model configuration.

    Args:
        arch: Model architecture ('deeponet', 'fno', 'unet')
        **kwargs: Configuration parameters

    Returns:
        Configuration instance for the specified architecture

    Raises:
        ValueError: If arch is not recognized

    Example:
        >>> config = create_config('deeponet', latent_dim=128)
        >>> config = create_config('fno', n_modes=32, n_layers=6)
    """
    arch = arch.lower()

    if arch == 'deeponet':
        return DeepONetConfig(**kwargs)
    elif arch == 'fno':
        return FNOConfig(**kwargs)
    elif arch == 'unet':
        return UNetConfig(**kwargs)
    else:
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            f"Valid options: 'deeponet', 'fno', 'unet'"
        )
