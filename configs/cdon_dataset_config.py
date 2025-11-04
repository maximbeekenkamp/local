"""
Configuration dataclass for CDON dataset parameters.

Stores dataset configuration including paths, splits, and normalization settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
import os


@dataclass
class CDONDatasetConfig:
    """
    Configuration for CDON earthquake dataset.

    Stores all parameters needed to load and process the CDON dataset,
    including file paths, split ratios, batch sizes, and normalization settings.
    """

    # Data paths
    data_dir: str = "CDONData"
    dummy_data_dir: str = "data/dummy_cdon"
    stats_path: str = "configs/cdon_stats.json"

    # Dataset dimensions (from CDON dataset)
    n_timesteps: int = 4000
    n_channels: int = 1  # Single channel (1D time series)
    n_train_samples: int = 100  # Total training samples before split
    n_test_samples: int = 44

    # Train/val split
    val_split_ratio: float = 0.2
    val_split_seed: int = 42

    # DataLoader parameters
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    # Normalization settings
    normalize: bool = True
    normalization_mode: str = "z-score"  # Only z-score supported for now

    # Computed properties (derived from split ratio)
    @property
    def n_train_split(self) -> int:
        """Number of samples in training split after train/val division."""
        return int(self.n_train_samples * (1 - self.val_split_ratio))

    @property
    def n_val_split(self) -> int:
        """Number of samples in validation split."""
        return self.n_train_samples - self.n_train_split

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of config (excludes computed properties)
        """
        return asdict(self)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output JSON file path

        Example:
            >>> config = CDONDatasetConfig(batch_size=64)
            >>> config.save('configs/my_dataset_config.json')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CDONDatasetConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            CDONDatasetConfig instance

        Example:
            >>> config_dict = {'batch_size': 64, 'val_split_ratio': 0.3}
            >>> config = CDONDatasetConfig.from_dict(config_dict)
        """
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> 'CDONDatasetConfig':
        """
        Load configuration from JSON file.

        Args:
            path: Input JSON file path

        Returns:
            CDONDatasetConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist

        Example:
            >>> config = CDONDatasetConfig.load('configs/my_dataset_config.json')
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"CDONDatasetConfig(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  batch_size={self.batch_size},\n"
            f"  train_split={self.n_train_split}, "
            f"val_split={self.n_val_split}, "
            f"test={self.n_test_samples},\n"
            f"  shape=[{self.n_channels}, {self.n_timesteps}],\n"
            f"  normalize={self.normalize}\n"
            f")"
        )
