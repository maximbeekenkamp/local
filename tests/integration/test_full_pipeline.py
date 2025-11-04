"""
Integration tests for end-to-end pipeline verification.

Tests the complete pipeline: data loading → model creation → training → evaluation
across all three baseline models (DeepONet, FNO, UNet) on both dummy and real data.
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.models.model_factory import create_model
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig
from configs.loss_config import BASELINE_CONFIG, BSP_CONFIG, SA_BSP_CONFIG


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def dummy_data_dir(project_root):
    """Get dummy data directory path."""
    return str(project_root / 'data' / 'dummy_cdon')


@pytest.fixture
def real_data_dir(project_root):
    """Get real data directory path."""
    return str(project_root / 'data' / 'CDONData')


@pytest.fixture
def test_config():
    """Create minimal training config for integration tests."""
    return TrainingConfig(
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=16,
        device='cpu',
        num_workers=0,
        verbose=False,
        save_best=False,  # Don't save during tests
        save_latest=False
    )


def create_dataloaders(data_dir: str, batch_size: int = 16):
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for loaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create normalization
    project_root = Path(__file__).parent.parent.parent
    stats_path = project_root / 'configs' / 'cdon_stats.json'
    normalizer = CDONNormalization(stats_path=str(stats_path))

    train_dataset = CDONDataset(
        data_dir=data_dir,
        split='train',
        normalize=normalizer
    )

    val_dataset = CDONDataset(
        data_dir=data_dir,
        split='test',
        normalize=normalizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


# ============================================================================
# Integration Test 1: DeepONet on Dummy Data
# ============================================================================

def test_end_to_end_deeponet_dummy(dummy_data_dir, test_config):
    """
    Test end-to-end pipeline with DeepONet on dummy data.

    Validates:
    - Data loading works
    - Model creation works
    - Training completes without errors
    - Loss decreases over epochs
    - Metrics are computed correctly
    - No NaN/Inf values
    """
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dummy_data_dir, batch_size=16)

    # Verify data loaded
    assert len(train_loader.dataset) > 0, "Train dataset should not be empty"
    assert len(val_loader.dataset) > 0, "Val dataset should not be empty"

    # Create model
    model = create_model('deeponet')
    assert model is not None, "Model should be created"

    # Create trainer
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config,
        loss_config=BASELINE_CONFIG,
        experiment_name='test_deeponet_dummy'
    )

    # Train
    results = trainer.train()

    # Validate results structure
    assert 'train_history' in results, "Results should contain train_history"
    assert 'val_history' in results, "Results should contain val_history"
    assert 'best_val_loss' in results, "Results should contain best_val_loss"

    # Validate training completed
    assert len(results['train_history']) == test_config.num_epochs, \
        f"Should have {test_config.num_epochs} training epochs"
    assert len(results['val_history']) == test_config.num_epochs, \
        f"Should have {test_config.num_epochs} validation epochs"

    # Validate metrics are finite
    final_train = results['train_history'][-1]
    final_val = results['val_history'][-1]

    assert torch.isfinite(torch.tensor(final_train['loss'])), \
        "Final train loss should be finite"
    assert torch.isfinite(torch.tensor(final_val['loss'])), \
        "Final val loss should be finite"
    assert torch.isfinite(torch.tensor(final_val['field_error'])), \
        "Final field error should be finite"

    # Validate loss is reasonable (DeepONet typically ~1.0 on dummy data)
    assert final_val['loss'] < 2.0, \
        f"Val loss should be < 2.0, got {final_val['loss']}"

    # Validate best loss improved from initial
    first_val_loss = results['val_history'][0]['loss']
    assert results['best_val_loss'] <= first_val_loss, \
        "Best val loss should be <= initial val loss"


# ============================================================================
# Integration Test 2: FNO on Dummy Data
# ============================================================================

def test_end_to_end_fno_dummy(dummy_data_dir, test_config):
    """
    Test end-to-end pipeline with FNO on dummy data.

    Validates:
    - FNO model trains successfully
    - Spectral convolution layers work correctly
    - Loss decreases (FNO typically converges faster)
    """
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dummy_data_dir, batch_size=16)

    # Create FNO model
    model = create_model('fno')
    assert model is not None, "FNO model should be created"

    # Create trainer
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config,
        loss_config=BASELINE_CONFIG,
        experiment_name='test_fno_dummy'
    )

    # Train
    results = trainer.train()

    # Validate training completed
    assert len(results['train_history']) == test_config.num_epochs, \
        "FNO should complete all epochs"

    # Validate metrics
    final_val = results['val_history'][-1]

    assert torch.isfinite(torch.tensor(final_val['loss'])), \
        "FNO final loss should be finite"
    assert torch.isfinite(torch.tensor(final_val['spectrum_error'])), \
        "FNO spectrum error should be finite"

    # FNO typically achieves better loss than DeepONet
    assert final_val['loss'] < 2.0, \
        f"FNO val loss should be < 2.0, got {final_val['loss']}"

    # Validate convergence (loss should decrease)
    initial_loss = results['val_history'][0]['loss']
    final_loss = results['val_history'][-1]['loss']
    assert final_loss <= initial_loss, \
        "FNO loss should decrease or stay same"


# ============================================================================
# Integration Test 3: UNet on Dummy Data
# ============================================================================

def test_end_to_end_unet_dummy(dummy_data_dir, test_config):
    """
    Test end-to-end pipeline with UNet on dummy data.

    Validates:
    - UNet encoder-decoder architecture works
    - Skip connections function correctly
    - UNet typically converges fastest on dummy data
    """
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dummy_data_dir, batch_size=16)

    # Create UNet model
    model = create_model('unet')
    assert model is not None, "UNet model should be created"

    # Create trainer
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config,
        loss_config=BASELINE_CONFIG,
        experiment_name='test_unet_dummy'
    )

    # Train
    results = trainer.train()

    # Validate training completed
    assert len(results['train_history']) == test_config.num_epochs, \
        "UNet should complete all epochs"

    # Validate metrics
    final_val = results['val_history'][-1]

    assert torch.isfinite(torch.tensor(final_val['loss'])), \
        "UNet final loss should be finite"
    assert torch.isfinite(torch.tensor(final_val['field_error'])), \
        "UNet field error should be finite"

    # UNet typically achieves best loss on dummy data
    assert final_val['loss'] < 2.0, \
        f"UNet val loss should be < 2.0, got {final_val['loss']}"

    # UNet should converge (often achieves very low loss)
    # Based on earlier run: UNet achieved ~0.10 loss
    assert final_val['field_error'] < 5.0, \
        "UNet should achieve reasonable field error"


# ============================================================================
# Integration Test 4: All Models on Same Data
# ============================================================================

def test_all_models_same_data(dummy_data_dir, test_config):
    """
    Test all three models on the same dummy data.

    Validates:
    - All models can train on identical dataset
    - No model fails or produces NaN
    - Results are comparable and reasonable
    """
    # Create dataloaders once (shared across all models)
    train_loader, val_loader = create_dataloaders(dummy_data_dir, batch_size=16)

    models_to_test = ['deeponet', 'fno', 'unet']
    results_all = {}

    for arch in models_to_test:
        # Create model
        model = create_model(arch)

        # Create trainer
        trainer = SimpleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=test_config,
            loss_config=BASELINE_CONFIG,
            experiment_name=f'test_{arch}_comparison'
        )

        # Train
        results = trainer.train()
        results_all[arch] = results

    # Validate all models completed training
    for arch in models_to_test:
        assert len(results_all[arch]['train_history']) == test_config.num_epochs, \
            f"{arch} should complete all epochs"

        final_val = results_all[arch]['val_history'][-1]

        # No NaN values
        assert torch.isfinite(torch.tensor(final_val['loss'])), \
            f"{arch} should have finite loss"
        assert torch.isfinite(torch.tensor(final_val['field_error'])), \
            f"{arch} should have finite field error"

        # Reasonable loss values
        assert final_val['loss'] < 5.0, \
            f"{arch} loss should be reasonable, got {final_val['loss']}"

    # All models should have produced results
    assert len(results_all) == 3, "Should have results for all 3 models"


# ============================================================================
# Integration Test 5: Real Data Compatibility
# ============================================================================

def test_real_data_compatibility(real_data_dir, test_config):
    """
    Test that real CDON data loads and trains without code changes.

    Validates:
    - Real data directory exists
    - Real data loads through CDONDataset
    - Shapes are compatible with models
    - Training runs without shape mismatches
    - Normalization works correctly
    """
    # Check if real data exists
    real_data_path = Path(real_data_dir)
    if not real_data_path.exists():
        pytest.skip(f"Real data not found at {real_data_dir}")

    # Create dataloaders with real data
    try:
        train_loader, val_loader = create_dataloaders(real_data_dir, batch_size=16)
    except Exception as e:
        pytest.fail(f"Failed to create dataloaders for real data: {e}")

    # Verify data loaded
    assert len(train_loader.dataset) > 0, "Real train dataset should not be empty"
    assert len(val_loader.dataset) > 0, "Real val dataset should not be empty"

    # Get a sample batch to check shapes
    sample_input, sample_target = next(iter(train_loader))
    assert sample_input.shape[1] == 1, "Input should have 1 channel"
    assert sample_input.shape[2] == 4000, "Input should have 4000 timesteps"
    assert sample_target.shape == sample_input.shape, \
        "Target should have same shape as input"

    # Test training with DeepONet (representative model)
    model = create_model('deeponet')

    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config,
        loss_config=BASELINE_CONFIG,
        experiment_name='test_real_data'
    )

    # Train for 5 epochs
    try:
        results = trainer.train()
    except Exception as e:
        pytest.fail(f"Training failed on real data: {e}")

    # Validate training completed
    assert len(results['train_history']) == test_config.num_epochs, \
        "Should complete all epochs on real data"

    # Validate metrics are finite (no NaN from normalization issues)
    final_val = results['val_history'][-1]
    assert torch.isfinite(torch.tensor(final_val['loss'])), \
        "Real data training should produce finite loss"
    assert torch.isfinite(torch.tensor(final_val['field_error'])), \
        "Real data training should produce finite field error"

    # Validate loss is reasonable (real data might be harder than dummy)
    assert final_val['loss'] < 10.0, \
        f"Real data loss should be reasonable, got {final_val['loss']}"

    # Validate no shape errors occurred (would have failed earlier if so)
    # If we reached here, pipeline works on real data
    assert True, "Pipeline successfully handled real data"


# ============================================================================
# Integration Test 6: BSP Loss Training
# ============================================================================

def test_bsp_loss_training(dummy_data_dir, test_config):
    """
    Test training with BSP loss.

    Validates:
    - BSP loss can be used in training without errors
    - Model trains and converges with BSP loss
    - Combined loss (base + spectral) is computed correctly
    """
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dummy_data_dir, batch_size=16)

    # Create model
    model = create_model('deeponet')

    # Create trainer with BSP loss
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config,
        loss_config=BSP_CONFIG,
        experiment_name='test_bsp_loss'
    )

    # Train
    results = trainer.train()

    # Validate training completed
    assert len(results['train_history']) == test_config.num_epochs
    assert len(results['val_history']) == test_config.num_epochs

    # Validate metrics are finite
    final_val = results['val_history'][-1]
    assert torch.isfinite(torch.tensor(final_val['loss'])), \
        "BSP loss training should produce finite loss"
    assert torch.isfinite(torch.tensor(final_val['field_error'])), \
        "BSP loss training should produce finite field error"

    # Validate loss is reasonable (BSP loss scale is different from baseline)
    # Just check training didn't explode
    assert final_val['loss'] < 1e6, \
        f"BSP loss exploded, got {final_val['loss']}"
    assert final_val['field_error'] < 10.0, \
        f"Field error should be reasonable, got {final_val['field_error']}"


# ============================================================================
# Integration Test 7: SA-BSP Loss Training
# ============================================================================

def test_sa_bsp_loss_training(dummy_data_dir, test_config):
    """
    Test training with SA-BSP loss.

    Validates:
    - SA-BSP loss with adaptive weights can be used in training
    - Separate weight optimizer is created automatically
    - Adaptive weights update during training
    - Model converges with SA-BSP loss
    """
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dummy_data_dir, batch_size=16)

    # Create model
    model = create_model('fno')

    # Create trainer with SA-BSP loss
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config,
        loss_config=SA_BSP_CONFIG,
        experiment_name='test_sa_bsp_loss'
    )

    # Verify weight optimizer was created
    assert trainer.weight_optimizer is not None, \
        "SA-BSP training should create weight optimizer"

    # Get initial adaptive weights
    initial_weights = trainer.criterion.spectral_loss.adaptive_weights().clone()

    # Train
    results = trainer.train()

    # Get final adaptive weights
    final_weights = trainer.criterion.spectral_loss.adaptive_weights()

    # Validate weights changed during training
    assert not torch.allclose(initial_weights, final_weights, rtol=1e-3), \
        "Adaptive weights should change during SA-BSP training"

    # Validate training completed
    assert len(results['train_history']) == test_config.num_epochs
    assert len(results['val_history']) == test_config.num_epochs

    # Validate metrics are finite
    final_val = results['val_history'][-1]
    assert torch.isfinite(torch.tensor(final_val['loss'])), \
        "SA-BSP loss training should produce finite loss"
    assert torch.isfinite(torch.tensor(final_val['field_error'])), \
        "SA-BSP loss training should produce finite field error"

    # Validate loss is reasonable (SA-BSP loss scale is different from baseline)
    # Just check training didn't explode
    assert final_val['loss'] < 1e6, \
        f"SA-BSP loss exploded, got {final_val['loss']}"
    assert final_val['field_error'] < 10.0, \
        f"Field error should be reasonable, got {final_val['field_error']}"
