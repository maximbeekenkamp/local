"""
Unit tests for CDON dataset classes and transforms.

Tests CDONDataset, CDONNormalization, and create_cdon_dataloaders function.
"""

import pytest
import numpy as np
import torch
import os
import tempfile
import json
from pathlib import Path

# Import classes to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.data_processing import CDONDataset, CDONNormalization, create_cdon_dataloaders


class TestCDONDatasetLength:
    """Test that dataset length is correct for different splits."""

    def test_train_split_length_dummy(self):
        """Verify training split has 80 samples (80% of 100) with dummy data."""
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='train',
            normalize=None,
            val_split_ratio=0.2,
            val_split_seed=42
        )
        assert len(dataset) == 80, f"Expected 80 train samples, got {len(dataset)}"

    def test_val_split_length_dummy(self):
        """Verify validation split has 20 samples (20% of 100) with dummy data."""
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='val',
            normalize=None,
            val_split_ratio=0.2,
            val_split_seed=42
        )
        assert len(dataset) == 20, f"Expected 20 val samples, got {len(dataset)}"

    def test_test_split_length_dummy(self):
        """Verify test split has 44 samples with dummy data."""
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='test',
            normalize=None
        )
        assert len(dataset) == 44, f"Expected 44 test samples, got {len(dataset)}"

    def test_train_split_length_real(self):
        """Verify training split has 80 samples with real data."""
        if not os.path.exists('CDONData/train_Loads.npy'):
            pytest.skip("Real CDON data not available")

        dataset = CDONDataset(
            data_dir='CDONData',
            split='train',
            normalize=None,
            val_split_ratio=0.2,
            val_split_seed=42
        )
        assert len(dataset) == 80, f"Expected 80 train samples, got {len(dataset)}"


class TestCDONDatasetGetItem:
    """Test that __getitem__ returns correct shapes and types."""

    def test_getitem_shape_without_transform(self):
        """Verify __getitem__ returns correct shapes without normalization (causal padding enabled by default)."""
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='train',
            normalize=None
            # use_causal_padding=True by default
        )

        input_tensor, target_tensor = dataset[0]

        # Check types
        assert isinstance(input_tensor, torch.Tensor), "Input should be torch.Tensor"
        assert isinstance(target_tensor, torch.Tensor), "Target should be torch.Tensor"

        # Check shapes (with causal padding: input is padded, target is not)
        expected_input_len = 4000 + (4000 - 1)  # signal_length + padding = 7999
        assert input_tensor.shape == (1, expected_input_len), \
            f"Expected shape [1, {expected_input_len}] (causal padding), got {input_tensor.shape}"
        assert target_tensor.shape == (1, 4000), f"Expected shape [1, 4000], got {target_tensor.shape}"

        # Check dtype
        assert input_tensor.dtype == torch.float32, "Input should be float32"
        assert target_tensor.dtype == torch.float32, "Target should be float32"

    def test_getitem_shape_with_transform(self):
        """Verify __getitem__ returns correct shapes with normalization (causal padding enabled by default)."""
        normalizer = CDONNormalization(stats_path='configs/cdon_stats.json')
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='train',
            normalize=normalizer
            # use_causal_padding=True by default
        )

        input_tensor, target_tensor = dataset[0]

        # Check shapes (with causal padding)
        expected_input_len = 4000 + (4000 - 1)  # signal_length + padding = 7999
        assert input_tensor.shape == (1, expected_input_len), \
            f"Expected shape [1, {expected_input_len}] (causal padding), got {input_tensor.shape}"
        assert target_tensor.shape == (1, 4000), f"Expected shape [1, 4000], got {target_tensor.shape}"

    def test_multiple_samples_access(self):
        """Verify can access multiple samples sequentially (causal padding enabled by default)."""
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='test',
            normalize=None
            # use_causal_padding=True by default
        )

        # Access first 5 samples
        expected_input_len = 4000 + (4000 - 1)  # signal_length + padding = 7999
        for i in range(5):
            input_tensor, target_tensor = dataset[i]
            assert input_tensor.shape == (1, expected_input_len), \
                f"Expected input shape [1, {expected_input_len}] (causal padding)"
            assert target_tensor.shape == (1, 4000), "Expected target shape [1, 4000]"


class TestCDONNormalization:
    """Test normalization transform and inverse operations."""

    def test_normalization_inverts_correctly(self):
        """Verify normalize then denormalize returns original values."""
        transform = CDONNormalization(stats_path='configs/cdon_stats.json')

        # Create random test data
        original_loads = torch.randn(10, 4000) * 0.014  # Approximate load scale
        original_responses = torch.randn(10, 4000) * 0.038  # Approximate response scale

        # Normalize
        normalized_loads = transform.normalize_loads(original_loads)
        normalized_responses = transform.normalize_responses(original_responses)

        # Denormalize
        recovered_loads = transform.denormalize_loads(normalized_loads)
        recovered_responses = transform.denormalize_responses(normalized_responses)

        # Check recovery within numerical precision
        assert torch.allclose(original_loads, recovered_loads, rtol=1e-5, atol=1e-7), \
            "Load normalization should be invertible"
        assert torch.allclose(original_responses, recovered_responses, rtol=1e-5, atol=1e-7), \
            "Response normalization should be invertible"

    def test_normalization_with_numpy_input(self):
        """Verify normalization works with numpy arrays."""
        transform = CDONNormalization(stats_path='configs/cdon_stats.json')

        # Create numpy test data
        loads_np = np.random.randn(5, 4000).astype(np.float32) * 0.014
        responses_np = np.random.randn(5, 4000).astype(np.float32) * 0.038

        # Normalize numpy arrays
        normalized_loads = transform.normalize_loads(loads_np)
        normalized_responses = transform.normalize_responses(responses_np)

        # Check output is tensor
        assert isinstance(normalized_loads, torch.Tensor), "Should convert numpy to tensor"
        assert isinstance(normalized_responses, torch.Tensor), "Should convert numpy to tensor"

    def test_stats_loading(self):
        """Verify stats are loaded correctly from JSON."""
        transform = CDONNormalization(stats_path='configs/cdon_stats.json')

        # Check stats dict is available
        stats = transform.get_stats_dict()
        assert 'load_mean' in stats
        assert 'load_std' in stats
        assert 'response_mean' in stats
        assert 'response_std' in stats

        # Check values are reasonable
        assert abs(stats['load_mean']) < 1e-3, "Load mean should be near zero"
        assert 0.01 < stats['load_std'] < 0.02, "Load std should be ~0.014"
        assert abs(stats['response_mean']) < 1e-3, "Response mean should be near zero"
        assert 0.03 < stats['response_std'] < 0.05, "Response std should be ~0.038"


class TestNormalizedDataStatistics:
    """Test that normalized data has mean≈0 and std≈1."""

    def test_normalized_data_statistics_dummy(self):
        """Verify normalized dummy data has mean≈0, std≈1."""
        normalizer = CDONNormalization(stats_path='configs/cdon_stats.json')
        dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='train',
            normalize=normalizer,
            use_causal_padding=False  # Disable to test pure normalization
        )

        # Collect all inputs and targets
        all_inputs = []
        all_targets = []
        for i in range(len(dataset)):
            input_tensor, target_tensor = dataset[i]
            all_inputs.append(input_tensor)
            all_targets.append(target_tensor)

        # Stack into single tensor
        all_inputs = torch.stack(all_inputs)  # Shape: [n_samples, 1, 4000]
        all_targets = torch.stack(all_targets)  # Shape: [n_samples, 1, 4000]

        # Compute statistics
        input_mean = all_inputs.mean().item()
        input_std = all_inputs.std().item()
        target_mean = all_targets.mean().item()
        target_std = all_targets.std().item()

        # Check mean is close to 0 (within 0.1)
        assert abs(input_mean) < 0.1, f"Input mean should be ~0, got {input_mean}"
        assert abs(target_mean) < 0.1, f"Target mean should be ~0, got {target_mean}"

        # Check std is close to 1 (within 0.2)
        assert abs(input_std - 1.0) < 0.2, f"Input std should be ~1, got {input_std}"
        assert abs(target_std - 1.0) < 0.2, f"Target std should be ~1, got {target_std}"

    def test_normalized_data_statistics_real(self):
        """Verify normalized real data has mean≈0, std≈1."""
        if not os.path.exists('CDONData/train_Loads.npy'):
            pytest.skip("Real CDON data not available")

        normalizer = CDONNormalization(stats_path='configs/cdon_stats.json')
        dataset = CDONDataset(
            data_dir='CDONData',
            split='train',
            normalize=normalizer,
            use_causal_padding=False  # Disable to test pure normalization
        )

        # Collect all data
        all_inputs = []
        all_targets = []
        for i in range(len(dataset)):
            input_tensor, target_tensor = dataset[i]
            all_inputs.append(input_tensor)
            all_targets.append(target_tensor)

        all_inputs = torch.stack(all_inputs)
        all_targets = torch.stack(all_targets)

        # Compute statistics
        input_mean = all_inputs.mean().item()
        input_std = all_inputs.std().item()
        target_mean = all_targets.mean().item()
        target_std = all_targets.std().item()

        # Real data should normalize very precisely (stats computed from this data)
        assert abs(input_mean) < 0.05, f"Input mean should be ~0, got {input_mean}"
        assert abs(target_mean) < 0.05, f"Target mean should be ~0, got {target_mean}"
        assert abs(input_std - 1.0) < 0.05, f"Input std should be ~1, got {input_std}"
        assert abs(target_std - 1.0) < 0.05, f"Target std should be ~1, got {target_std}"


class TestDataLoaderBatching:
    """Test that DataLoader produces correct batch shapes."""

    def test_dataloader_batch_shapes(self):
        """Verify DataLoader produces batches with correct shapes (causal padding enabled by default)."""
        train_loader, val_loader, test_loader = create_cdon_dataloaders(
            data_dir='data/dummy_cdon',
            batch_size=8,
            use_dummy=True,
            num_workers=0
            # use_causal_padding=True by default
        )

        # Expected shapes with causal padding
        expected_input_len = 4000 + (4000 - 1)  # signal_length + padding = 7999

        # Test train loader
        inputs, targets = next(iter(train_loader))
        assert inputs.shape == (8, 1, expected_input_len), \
            f"Expected [8, 1, {expected_input_len}] (causal padding), got {inputs.shape}"
        assert targets.shape == (8, 1, 4000), f"Expected [8, 1, 4000], got {targets.shape}"

        # Test val loader
        inputs, targets = next(iter(val_loader))
        assert inputs.shape == (8, 1, expected_input_len), \
            f"Expected [8, 1, {expected_input_len}] (causal padding), got {inputs.shape}"
        assert targets.shape == (8, 1, 4000), f"Expected [8, 1, 4000], got {targets.shape}"

        # Test test loader
        inputs, targets = next(iter(test_loader))
        assert inputs.shape == (8, 1, expected_input_len), \
            f"Expected [8, 1, {expected_input_len}] (causal padding), got {inputs.shape}"
        assert targets.shape == (8, 1, 4000), f"Expected [8, 1, 4000], got {targets.shape}"

    def test_dataloader_different_batch_sizes(self):
        """Verify DataLoader works with different batch sizes (causal padding enabled by default)."""
        expected_input_len = 4000 + (4000 - 1)  # signal_length + padding = 7999

        for batch_size in [4, 16, 32]:
            train_loader, _, _ = create_cdon_dataloaders(
                data_dir='data/dummy_cdon',
                batch_size=batch_size,
                use_dummy=True,
                num_workers=0
                # use_causal_padding=True by default
            )

            inputs, targets = next(iter(train_loader))
            assert inputs.shape[0] == batch_size, f"Batch size should be {batch_size}"
            assert inputs.shape == (batch_size, 1, expected_input_len), \
                f"Expected [{batch_size}, 1, {expected_input_len}] (causal padding)"

    def test_dataloader_iteration_complete(self):
        """Verify can iterate through entire DataLoader."""
        train_loader, val_loader, test_loader = create_cdon_dataloaders(
            data_dir='data/dummy_cdon',
            batch_size=16,
            use_dummy=True,
            num_workers=0
        )

        # Count samples in train loader
        train_samples = 0
        for inputs, targets in train_loader:
            train_samples += inputs.shape[0]
        assert train_samples == 80, f"Should have 80 train samples, got {train_samples}"

        # Count samples in val loader
        val_samples = 0
        for inputs, targets in val_loader:
            val_samples += inputs.shape[0]
        assert val_samples == 20, f"Should have 20 val samples, got {val_samples}"

        # Count samples in test loader
        test_samples = 0
        for inputs, targets in test_loader:
            test_samples += inputs.shape[0]
        assert test_samples == 44, f"Should have 44 test samples, got {test_samples}"


class TestDummyAndRealDataCompatibility:
    """Test that same code works for both dummy and real data."""

    def test_dummy_and_real_have_same_interface(self):
        """Verify dummy and real datasets have identical interface."""
        # Create dummy dataset
        dummy_dataset = CDONDataset(
            data_dir='data/dummy_cdon',
            split='train',
            normalize=None
        )

        # Check if real data exists
        if not os.path.exists('CDONData/train_Loads.npy'):
            pytest.skip("Real CDON data not available")

        # Create real dataset
        real_dataset = CDONDataset(
            data_dir='CDONData',
            split='train',
            normalize=None
        )

        # Both should have same length (80 after split)
        assert len(dummy_dataset) == len(real_dataset), \
            f"Dummy and real datasets should have same length"

        # Both should return same shapes
        dummy_input, dummy_target = dummy_dataset[0]
        real_input, real_target = real_dataset[0]

        assert dummy_input.shape == real_input.shape, "Shapes should match"
        assert dummy_target.shape == real_target.shape, "Shapes should match"

    def test_dataloaders_work_with_both_sources(self):
        """Verify DataLoader factory works with both dummy and real data (causal padding enabled by default)."""
        # Test with dummy data
        dummy_train, dummy_val, dummy_test = create_cdon_dataloaders(
            data_dir='data/dummy_cdon',
            batch_size=8,
            use_dummy=True,
            num_workers=0
            # use_causal_padding=True by default
        )

        dummy_inputs, dummy_targets = next(iter(dummy_train))
        expected_input_len = 4000 + (4000 - 1)  # signal_length + padding = 7999
        assert dummy_inputs.shape == (8, 1, expected_input_len), \
            f"Expected [8, 1, {expected_input_len}] (causal padding)"

        # Test with real data if available
        if os.path.exists('CDONData/train_Loads.npy'):
            real_train, real_val, real_test = create_cdon_dataloaders(
                data_dir='CDONData',
                batch_size=8,
                use_dummy=False,
                num_workers=0
            )

            real_inputs, real_targets = next(iter(real_train))
            assert real_inputs.shape == dummy_inputs.shape, \
                "Dummy and real should produce same shapes"

    def test_switching_between_dummy_and_real(self):
        """Verify seamless switching between dummy and real data."""
        def run_pipeline(data_dir, use_dummy):
            """Helper to run same pipeline on different data sources."""
            train_loader, val_loader, test_loader = create_cdon_dataloaders(
                data_dir=data_dir,
                batch_size=16,
                use_dummy=use_dummy,
                num_workers=0
            )

            # Get batch from each loader
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            test_batch = next(iter(test_loader))

            return {
                'train_shape': train_batch[0].shape,
                'val_shape': val_batch[0].shape,
                'test_shape': test_batch[0].shape,
            }

        # Run on dummy data
        dummy_results = run_pipeline('data/dummy_cdon', use_dummy=True)

        # Run on real data if available
        if os.path.exists('CDONData/train_Loads.npy'):
            real_results = run_pipeline('CDONData', use_dummy=False)

            # Verify all shapes match
            assert dummy_results['train_shape'] == real_results['train_shape']
            assert dummy_results['val_shape'] == real_results['val_shape']
            assert dummy_results['test_shape'] == real_results['test_shape']
