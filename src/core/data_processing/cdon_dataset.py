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
from typing import Optional, Tuple, Dict
from .cdon_transforms import CDONNormalization
from src.data.preprocessing_utils import prepare_causal_deeponet_data, prepare_causal_sequence_data, create_penalty_weights


class CDONDataset(Dataset):
    """
    PyTorch Dataset for CDON earthquake time-series data with dual-mode support.

    Supports two modes:
    1. 'per_timestep': Sliding windows with zero-padding for DeepONet MSE loss
       - Returns: {'input': [4000], 'target': [], 'time_coord': [], 'penalty': []}
       - Total samples: n_earthquakes × signal_length (400K samples)

    2. 'sequence': Full sequences without padding for BSP loss and FNO/UNet MSE
       - Returns: (input [1, 4000], target [1, 4000])
       - Total samples: n_earthquakes (100 samples)

    Causality in 'per_timestep' mode:
        Prediction at timestep t only uses input [0, ..., t] via zero-padding.

    Reference:
        Penwarden et al. "A metalearning approach for physics-informed neural networks" (2023)
        Custom_dataset.py: reshapeTraining class
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        normalize: Optional[CDONNormalization] = None,
        use_dummy: bool = False,
        val_split_ratio: float = 0.2,
        val_split_seed: int = 42,
        signal_length: int = 4000,
        mode: str = 'per_timestep',
        use_causal_sequence: bool = False
    ):
        """
        Initialize CDON dataset with dual-mode support.

        Args:
            data_dir: Path to directory containing .npy files.
                     For real data: 'CDONData'
                     For dummy data: 'data/dummy_cdon'
            split: Dataset split - 'train', 'val', or 'test'
            normalize: CDONNormalization object for z-score normalization (optional)
            use_dummy: If True, expect dummy data directory structure (informational only)
            val_split_ratio: Fraction of training data to use for validation (default 0.2)
            val_split_seed: Random seed for reproducible train/val split (default 42)
            signal_length: Original signal length (default 4000)
            mode: Dataset mode - 'per_timestep' or 'sequence' (default: 'per_timestep')
            use_causal_sequence: If True and mode='sequence', apply zero-padding for causality.
                                Only applicable for causal architectures like DeepONet.
                                Should be False for non-causal models like FNO/UNet. (default: False)

        Raises:
            FileNotFoundError: If required .npy files don't exist
            ValueError: If split or mode is invalid, or data shapes are inconsistent
        """
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.use_dummy = use_dummy
        self.val_split_ratio = val_split_ratio
        self.val_split_seed = val_split_seed
        self.signal_length = signal_length
        self.mode = mode
        self.use_causal_sequence = use_causal_sequence

        # Validate arguments
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")
        if mode not in ['per_timestep', 'sequence']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'per_timestep' or 'sequence'.")

        # Load raw data based on split
        self.loads, self.responses = self._load_data()

        # Validate shapes
        if self.loads.shape != self.responses.shape:
            raise ValueError(
                f"Shape mismatch: loads {self.loads.shape} != responses {self.responses.shape}"
            )

        # Store original dimensions
        self.n_earthquakes, self.n_timesteps = self.loads.shape

        # Prepare data based on mode
        if mode == 'per_timestep':
            self._prepare_windowed_data()
        else:  # mode == 'sequence'
            self._prepare_sequence_data()

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

    def _prepare_windowed_data(self):
        """
        Convert raw earthquake data to per-timestep windowed samples.

        Applies normalization, then uses prepare_causal_deeponet_data() to create
        sliding windows with zero-padding. Also computes penalty weights and time coordinates.

        Sets:
            self.windowed_inputs: [N*T, signal_length] - zero-padded windows
            self.windowed_targets: [N*T] - scalar outputs
            self.time_coords: [N*T] - time coordinates in [0, 1]
            self.penalty_weights: [N] - penalty per earthquake
        """
        # Convert to tensors
        loads_tensor = torch.from_numpy(self.loads).float()  # [N, T]
        responses_tensor = torch.from_numpy(self.responses).float()  # [N, T]

        # Apply normalization if provided
        if self.normalize is not None:
            # Normalize each sample
            for i in range(self.n_earthquakes):
                loads_tensor[i] = self.normalize.normalize_loads(loads_tensor[i])
                responses_tensor[i] = self.normalize.normalize_responses(responses_tensor[i])

        # Compute penalty weights (one per earthquake, applied to all timesteps)
        # Penalty = 1 / max(|response|)²
        self.penalty_weights = create_penalty_weights(responses_tensor)  # [N]

        # Create windowed causal data
        # prepare_causal_deeponet_data converts [N, T] → [N*T, T] inputs and [N*T] outputs
        self.windowed_inputs, self.windowed_targets = prepare_causal_deeponet_data(
            loads_tensor,
            responses_tensor,
            signal_length=self.signal_length
        )

        # Create time coordinates [0, 1] for each timestep
        # Shape: [T] repeated for each earthquake
        time_grid = torch.linspace(0, 1, self.signal_length)  # [T]
        # Tile for all earthquakes: [N, T] → [N*T]
        self.time_coords = time_grid.repeat(self.n_earthquakes)  # [N*T]

        # Expand penalty weights to match per-timestep samples
        # [N] → [N*T] by repeating each penalty signal_length times
        self.penalty_weights_expanded = self.penalty_weights.repeat_interleave(self.signal_length)  # [N*T]

        # Total samples after windowing
        self.n_samples = self.windowed_inputs.shape[0]  # N * T

    def _prepare_sequence_data(self):
        """
        Prepare full sequences with optional causal zero-padding for BSP loss.

        If use_causal_sequence=True:
            Creates per-timestep zero-padded windows organized by trajectory.
            Used for causal models (DeepONet) with BSP loss.
            Shape: [N, T, signal_length] where each timestep has appropriate padding

        If use_causal_sequence=False:
            Creates full sequences without padding.
            Used for non-causal models (FNO, UNet) with BSP or MSE loss.
            Shape: [N, 1, signal_length]

        Sets:
            self.sequence_inputs: Sequence inputs (shape depends on use_causal_sequence)
            self.sequence_targets: Sequence targets (shape depends on use_causal_sequence)
            self.n_samples: N - number of earthquakes
        """
        # Convert to tensors
        loads_tensor = torch.from_numpy(self.loads).float()  # [N, T]
        responses_tensor = torch.from_numpy(self.responses).float()  # [N, T]

        # Apply normalization if provided
        if self.normalize is not None:
            # Normalize each sample
            for i in range(self.n_earthquakes):
                loads_tensor[i] = self.normalize.normalize_loads(loads_tensor[i])
                responses_tensor[i] = self.normalize.normalize_responses(responses_tensor[i])

        if self.use_causal_sequence:
            # Use causal zero-padding for DeepONet BSP loss
            # Returns: [N, T, signal_length] inputs and [N, T] targets
            self.sequence_inputs, self.sequence_targets = prepare_causal_sequence_data(
                loads_tensor,
                responses_tensor,
                signal_length=self.signal_length
            )
            # Targets shape: [N, T] - no channel dimension needed for causal sequence
        else:
            # Use full sequences without padding for FNO/UNet
            # Add channel dimension: [N, T] → [N, 1, T]
            self.sequence_inputs = loads_tensor.unsqueeze(1)   # [N, 1, signal_length]
            self.sequence_targets = responses_tensor.unsqueeze(1)  # [N, 1, signal_length]

        # Total samples: number of earthquakes
        self.n_samples = self.n_earthquakes

    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            For 'per_timestep' mode: n_earthquakes × signal_length (e.g., 320,000 for 80 earthquakes)
            For 'sequence' mode: n_earthquakes (e.g., 100)
        """
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        Get a single sample from dataset.

        Args:
            idx: Sample index

        Returns:
            For 'per_timestep' mode:
                Dictionary with keys:
                - 'input': [signal_length] - windowed input with zero-padding
                - 'target': [] - scalar response at timestep
                - 'time_coord': [] - scalar time in [0, 1]
                - 'penalty': [] - scalar weight = 1/max(|response|)²

            For 'sequence' mode:
                Tuple: (input [1, 4000], target [1, 4000]) - full sequences without padding
        """
        if self.mode == 'per_timestep':
            return {
                'input': self.windowed_inputs[idx],      # [signal_length]
                'target': self.windowed_targets[idx],    # [] scalar
                'time_coord': self.time_coords[idx],     # [] scalar
                'penalty': self.penalty_weights_expanded[idx]  # [] scalar
            }
        else:  # mode == 'sequence'
            return (self.sequence_inputs[idx], self.sequence_targets[idx])

    def get_raw_earthquake(self, earthquake_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get raw earthquake data without normalization or windowing (for debugging/visualization).

        Args:
            earthquake_idx: Earthquake index in range [0, n_earthquakes)

        Returns:
            Tuple of (load, response) as numpy arrays with shape [signal_length]

        Note:
            This returns the original earthquake data, NOT the per-timestep windowed samples.
        """
        if earthquake_idx >= self.n_earthquakes:
            raise IndexError(
                f"Earthquake index {earthquake_idx} out of range [0, {self.n_earthquakes})"
            )
        return self.loads[earthquake_idx].copy(), self.responses[earthquake_idx].copy()


def create_cdon_dataloaders(
    data_dir: str,
    batch_size_per_timestep: int = 32,
    batch_size_sequence: int = 4,
    use_dummy: bool = False,
    val_split_ratio: float = 0.2,
    val_split_seed: int = 42,
    stats_path: str = "configs/cdon_stats.json",
    num_workers: int = 0,
    pin_memory: bool = True,
    signal_length: int = 4000,
    use_causal_sequence: bool = False
):
    """
    Factory function to create dual DataLoaders for dual-batch training.

    Creates both per-timestep and sequence loaders for combined MSE+BSP training.
    Matches the dual-batch paradigm for:
    - DeepONet: MSE on per-timestep, BSP on causal sequences (zero-padded)
    - FNO/UNet: Both MSE and BSP on full sequences (non-causal)

    Args:
        data_dir: Path to data directory ('CDONData' or 'data/dummy_cdon')
        batch_size_per_timestep: Batch size for per-timestep loader (default 32)
        batch_size_sequence: Batch size for sequence loader (default 4)
        use_dummy: Whether using dummy data (informational)
        val_split_ratio: Fraction of training data for validation (default 0.2)
        val_split_seed: Seed for reproducible train/val split (default 42)
        stats_path: Path to normalization statistics JSON (default 'configs/cdon_stats.json')
        num_workers: Number of DataLoader workers (default 0 for single-process)
        pin_memory: Whether to pin memory for faster GPU transfer (default True)
        signal_length: Original signal length (default 4000)
        use_causal_sequence: Whether to use causal zero-padding for sequence mode.
                            Set to True for DeepONet, False for FNO/UNet (default False)

    Returns:
        Tuple of (per_timestep_train_loader, sequence_train_loader,
                  per_timestep_val_loader, sequence_val_loader)

        Or for simple sequence-only access:
        Can use sequence loaders directly for non-causal models.

    Note:
        Per-timestep loaders (for DeepONet MSE):
        - Dict format with 'input', 'target', 'time_coord', 'penalty'
        - Large batch size (32) due to many samples available
        - SHUFFLED for better training

        Sequence loaders (for BSP and FNO/UNet):
        - For DeepONet (use_causal_sequence=True): [N, T, signal_length] with zero-padding
        - For FNO/UNet (use_causal_sequence=False): [N, 1, signal_length] full sequences
        - Small batch size (4) due to memory constraints
        - NOT SHUFFLED (important for BSP consistency)
    """
    # Create normalization
    normalizer = CDONNormalization(stats_path=stats_path)

    # ===== Per-timestep mode (for DeepONet MSE) =====
    per_timestep_train_dataset = CDONDataset(
        data_dir=data_dir,
        split='train',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        signal_length=signal_length,
        mode='per_timestep'
    )

    per_timestep_val_dataset = CDONDataset(
        data_dir=data_dir,
        split='val',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        signal_length=signal_length,
        mode='per_timestep'
    )

    # ===== Sequence mode (for BSP and FNO/UNet) =====
    sequence_train_dataset = CDONDataset(
        data_dir=data_dir,
        split='train',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        signal_length=signal_length,
        mode='sequence',
        use_causal_sequence=use_causal_sequence
    )

    sequence_val_dataset = CDONDataset(
        data_dir=data_dir,
        split='val',
        normalize=normalizer,
        use_dummy=use_dummy,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        signal_length=signal_length,
        mode='sequence',
        use_causal_sequence=use_causal_sequence
    )

    # ===== Create DataLoaders =====

    # Per-timestep loaders: SHUFFLED, larger batch size
    per_timestep_train_loader = DataLoader(
        per_timestep_train_dataset,
        batch_size=batch_size_per_timestep,
        shuffle=True,  # Shuffle per-timestep samples
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    per_timestep_val_loader = DataLoader(
        per_timestep_val_dataset,
        batch_size=batch_size_per_timestep,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Sequence loaders: NOT SHUFFLED (critical for BSP), smaller batch size
    sequence_train_loader = DataLoader(
        sequence_train_dataset,
        batch_size=batch_size_sequence,
        shuffle=False,  # MUST NOT shuffle for BSP consistency
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    sequence_val_loader = DataLoader(
        sequence_val_dataset,
        batch_size=batch_size_sequence,
        shuffle=False,  # MUST NOT shuffle for BSP consistency
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return (
        per_timestep_train_loader,
        per_timestep_val_loader,
        sequence_train_loader,
        sequence_val_loader
    )
