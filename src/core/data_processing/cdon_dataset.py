"""
PyTorch Dataset class for CDON earthquake data.

Loads CDON data (earthquake accelerations → structural displacements),
applies normalization, and provides data in format expected by neural operators.

Supports causal zero-padding preprocessing to enforce physical causality at the data level,
matching the reference CausalityDeepONet implementation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from .cdon_transforms import CDONNormalization
from src.data.preprocessing_utils import prepare_causal_sequence_data


class CDONDataset(Dataset):
    """
    PyTorch Dataset for CDON earthquake time-series data.

    Loads acceleration (loads) and displacement (responses) from .npy files,
    applies normalization, and returns tensors with shape [channels=1, timesteps].

    Supports train/val/test splits with deterministic validation split from training data.

    Causal Mode:
        When use_causal_padding=True, applies zero-padding preprocessing to enforce
        physical causality at the data level (matches reference CausalityDeepONet).
        Inputs are left-padded with (signal_length - 1) zeros, so output at time t
        only depends on input up to time t.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        normalize: Optional[CDONNormalization] = None,
        use_dummy: bool = False,
        val_split_ratio: float = 0.2,
        val_split_seed: int = 42,
        use_causal_padding: bool = True,  # ENABLED BY DEFAULT (matches reference)
        signal_length: int = 4000
    ):
        """
        Initialize CDON dataset.

        Args:
            data_dir: Path to directory containing .npy files.
                     For real data: 'CDONData'
                     For dummy data: 'data/dummy_cdon'
            split: Dataset split - 'train', 'val', or 'test'
            normalize: CDONNormalization object for z-score normalization (optional)
            use_dummy: If True, expect dummy data directory structure (informational only)
            val_split_ratio: Fraction of training data to use for validation (default 0.2)
            val_split_seed: Random seed for reproducible train/val split (default 42)
            use_causal_padding: If True, apply zero-padding for causality (default True)
                               Matches reference CausalityDeepONet implementation
                               Suitable for UNet and FNO models
                               For DeepONet, use prepare_causal_deeponet_data() separately
                               Set to False to disable (standard preprocessing)
            signal_length: Original signal length (default 4000)
                          Used to determine padding amount

        Raises:
            FileNotFoundError: If required .npy files don't exist
            ValueError: If split is invalid or data shapes are inconsistent

        Note:
            When use_causal_padding=True:
            - Input shape becomes [1, signal_length + (signal_length - 1)]
            - Output shape remains [1, signal_length]
            - For signal_length=4000: inputs become [1, 7999], outputs stay [1, 4000]
        """
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.use_dummy = use_dummy
        self.val_split_ratio = val_split_ratio
        self.val_split_seed = val_split_seed
        self.use_causal_padding = use_causal_padding
        self.signal_length = signal_length

        # Validate split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        # Load data based on split
        self.loads, self.responses = self._load_data()

        # Validate shapes
        if self.loads.shape != self.responses.shape:
            raise ValueError(
                f"Shape mismatch: loads {self.loads.shape} != responses {self.responses.shape}"
            )

        # Store dataset dimensions
        self.n_samples, self.n_timesteps = self.loads.shape
        self.n_channels = 1  # Single channel for 1D time series

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load appropriate data based on split.

        For 'train' and 'val': Load train_*.npy files, then split
        For 'test': Load test_*.npy files

        Returns:
            Tuple of (loads, responses) arrays with shape [n_samples, n_timesteps]
        """
        if self.split in ['train', 'val']:
            # Load full training data
            loads_path = os.path.join(self.data_dir, 'train_Loads.npy')
            responses_path = os.path.join(self.data_dir, 'train_Responses.npy')

            if not os.path.exists(loads_path):
                raise FileNotFoundError(f"Train loads not found: {loads_path}")
            if not os.path.exists(responses_path):
                raise FileNotFoundError(f"Train responses not found: {responses_path}")

            loads = np.load(loads_path)
            responses = np.load(responses_path)

            # Split into train/val
            n_samples = loads.shape[0]
            n_val = int(n_samples * self.val_split_ratio)
            n_train = n_samples - n_val

            # Use deterministic shuffle for reproducible split
            rng = np.random.RandomState(self.val_split_seed)
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            if self.split == 'train':
                # First n_train samples after shuffle
                selected_indices = indices[:n_train]
            else:  # val
                # Last n_val samples after shuffle
                selected_indices = indices[n_train:]

            loads = loads[selected_indices]
            responses = responses[selected_indices]

        else:  # test
            loads_path = os.path.join(self.data_dir, 'test_Loads.npy')
            responses_path = os.path.join(self.data_dir, 'test_Responses.npy')

            if not os.path.exists(loads_path):
                raise FileNotFoundError(f"Test loads not found: {loads_path}")
            if not os.path.exists(responses_path):
                raise FileNotFoundError(f"Test responses not found: {responses_path}")

            loads = np.load(loads_path)
            responses = np.load(responses_path)

        return loads, responses

    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            Number of samples (80 for train, 20 for val, 44 for test by default)
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_tensor, target_tensor)

            Without causal padding:
                - input_tensor: [1, signal_length]
                - target_tensor: [1, signal_length]

            With causal padding (use_causal_padding=True):
                - input_tensor: [1, signal_length + (signal_length - 1)]
                - target_tensor: [1, signal_length]

            Example for signal_length=4000:
                - Without padding: inputs [1, 4000], targets [1, 4000]
                - With padding: inputs [1, 7999], targets [1, 4000]
        """
        # Get sample from numpy arrays
        load = self.loads[idx]  # Shape: [n_timesteps]
        response = self.responses[idx]  # Shape: [n_timesteps]

        # Convert to tensors
        load_tensor = torch.from_numpy(load).float()
        response_tensor = torch.from_numpy(response).float()

        # Apply normalization if provided
        if self.normalize is not None:
            load_tensor = self.normalize.normalize_loads(load_tensor)
            response_tensor = self.normalize.normalize_responses(response_tensor)

        # Add channel dimension: [n_timesteps] -> [1, n_timesteps]
        load_tensor = load_tensor.unsqueeze(0)
        response_tensor = response_tensor.unsqueeze(0)

        # Apply causal zero-padding if enabled
        if self.use_causal_padding:
            # Left-pad input with (signal_length - 1) zeros for causality
            # Uses prepare_causal_sequence_data() which handles the padding
            load_tensor = prepare_causal_sequence_data(
                load_tensor,
                signal_length=self.signal_length
            )
            # Note: response_tensor stays as-is (not padded)

        return load_tensor, response_tensor

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get raw sample without normalization (for debugging/visualization).

        Args:
            idx: Sample index

        Returns:
            Tuple of (load, response) as numpy arrays with shape [n_timesteps]
        """
        return self.loads[idx].copy(), self.responses[idx].copy()


def create_cdon_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    use_dummy: bool = False,
    val_split_ratio: float = 0.2,
    val_split_seed: int = 42,
    stats_path: str = "configs/cdon_stats.json",
    num_workers: int = 0,
    pin_memory: bool = True,
    use_causal_padding: bool = True,  # ENABLED BY DEFAULT (matches reference)
    signal_length: int = 4000
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train, validation, and test DataLoaders.

    Creates datasets with proper normalization and returns DataLoaders
    configured for training.

    Args:
        data_dir: Path to data directory ('CDONData' or 'data/dummy_cdon')
        batch_size: Batch size for DataLoader (default 32)
        use_dummy: Whether using dummy data (informational)
        val_split_ratio: Fraction of training data for validation (default 0.2)
        val_split_seed: Seed for reproducible train/val split (default 42)
        stats_path: Path to normalization statistics JSON (default 'configs/cdon_stats.json')
        num_workers: Number of DataLoader workers (default 0 for single-process)
        pin_memory: Whether to pin memory for faster GPU transfer (default True)
        use_causal_padding: Apply zero-padding for causality (default True)
                           Matches reference CausalityDeepONet implementation
                           Set to False to disable (standard preprocessing)
        signal_length: Original signal length (default 4000)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> # Without causal padding (standard)
        >>> train_loader, val_loader, test_loader = create_cdon_dataloaders(
        ...     data_dir='data/dummy_cdon',
        ...     batch_size=16,
        ...     use_dummy=True
        ... )
        >>> for inputs, targets in train_loader:
        ...     # inputs shape: [16, 1, 4000]
        ...     # targets shape: [16, 1, 4000]
        ...     pass
        >>>
        >>> # With causal padding (for UNet/FNO)
        >>> train_loader, val_loader, test_loader = create_cdon_dataloaders(
        ...     data_dir='data/dummy_cdon',
        ...     batch_size=16,
        ...     use_dummy=True,
        ...     use_causal_padding=True
        ... )
        >>> for inputs, targets in train_loader:
        ...     # inputs shape: [16, 1, 7999]  # Zero-padded
        ...     # targets shape: [16, 1, 4000]
        ...     pass
    """
    # Create normalization
    normalizer = CDONNormalization(stats_path=stats_path)

    # Create datasets for each split
    train_dataset = CDONDataset(
        data_dir=data_dir,
        split='train',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        use_causal_padding=use_causal_padding,
        signal_length=signal_length
    )

    val_dataset = CDONDataset(
        data_dir=data_dir,
        split='val',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        use_causal_padding=use_causal_padding,
        signal_length=signal_length
    )

    test_dataset = CDONDataset(
        data_dir=data_dir,
        split='test',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        use_causal_padding=use_causal_padding,
        signal_length=signal_length
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
