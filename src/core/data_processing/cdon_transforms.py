"""
Normalization transforms for CDON earthquake dataset.

Implements z-score normalization using pre-computed statistics from Phase 1.
"""

import json
import torch
from typing import Dict, Union
import numpy as np


class CDONNormalization:
    """
    Normalization transform for CDON earthquake data using z-score normalization.

    Applies separate normalization to loads (inputs) and responses (outputs)
    using pre-computed global statistics from Phase 1 analysis.

    Normalization: x_norm = (x - mean) / std
    Denormalization: x = x_norm * std + mean
    """

    def __init__(self, stats_path: str = "configs/cdon_stats.json"):
        """
        Initialize normalization transform with statistics from JSON file.

        Args:
            stats_path: Path to JSON file containing normalization statistics.
                       Must contain: load_mean, load_std, response_mean, response_std

        Raises:
            FileNotFoundError: If stats file doesn't exist
            KeyError: If required statistics are missing
        """
        # Load statistics from JSON
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Extract and validate required statistics
        required_keys = ['load_mean', 'load_std', 'response_mean', 'response_std']
        for key in required_keys:
            if key not in stats:
                raise KeyError(f"Missing required statistic '{key}' in {stats_path}")

        # Store normalization parameters as tensors for efficient computation
        self.load_mean = torch.tensor(stats['load_mean'], dtype=torch.float32)
        self.load_std = torch.tensor(stats['load_std'], dtype=torch.float32)
        self.response_mean = torch.tensor(stats['response_mean'], dtype=torch.float32)
        self.response_std = torch.tensor(stats['response_std'], dtype=torch.float32)

        # Store raw stats dict for reference
        self.stats = stats

    def normalize_loads(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Normalize load (input) data using z-score normalization.

        Args:
            x: Load data as tensor or array

        Returns:
            Normalized load tensor: (x - load_mean) / load_std
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Move normalization params to same device as input
        mean = self.load_mean.to(x.device)
        std = self.load_std.to(x.device)

        return (x - mean) / std

    def normalize_responses(self, y: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Normalize response (output) data using z-score normalization.

        Args:
            y: Response data as tensor or array

        Returns:
            Normalized response tensor: (y - response_mean) / response_std
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # Move normalization params to same device as input
        mean = self.response_mean.to(y.device)
        std = self.response_std.to(y.device)

        return (y - mean) / std

    def denormalize_loads(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize load data (inverse of normalize_loads).

        Args:
            x_norm: Normalized load tensor

        Returns:
            Original scale tensor: x_norm * load_std + load_mean
        """
        mean = self.load_mean.to(x_norm.device)
        std = self.load_std.to(x_norm.device)

        return x_norm * std + mean

    def denormalize_responses(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize response data (inverse of normalize_responses).

        Args:
            y_norm: Normalized response tensor

        Returns:
            Original scale tensor: y_norm * response_std + response_mean
        """
        mean = self.response_mean.to(y_norm.device)
        std = self.response_std.to(y_norm.device)

        return y_norm * std + mean

    def __call__(self, loads: Union[torch.Tensor, np.ndarray],
                 responses: Union[torch.Tensor, np.ndarray]) -> tuple:
        """
        Normalize both loads and responses (convenience method).

        Args:
            loads: Input load data
            responses: Target response data

        Returns:
            Tuple of (normalized_loads, normalized_responses)
        """
        return self.normalize_loads(loads), self.normalize_responses(responses)

    def get_stats_dict(self) -> Dict[str, float]:
        """
        Get dictionary of normalization statistics.

        Returns:
            Dictionary containing all statistics from JSON file
        """
        return self.stats.copy()
