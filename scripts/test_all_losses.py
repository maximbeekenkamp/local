#!/usr/bin/env python3
"""
Comprehensive test script for all loss types.
Tests BASELINE, BSP, Log-BSP, and all SA-BSP variants.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.models.model_factory import create_model
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig
from configs.loss_config import (
    BASELINE_CONFIG,
    BSP_CONFIG,
    LOG_BSP_CONFIG,
    SA_BSP_PERBIN_CONFIG,
    SA_BSP_GLOBAL_CONFIG,
    SA_BSP_COMBINED_CONFIG
)
from src.core.evaluation.loss_factory import create_loss


def test_loss_type(loss_name, loss_config, model_arch='deeponet', num_epochs=2):
    """
    Test a single loss configuration.

    Args:
        loss_name: Name of the loss type
        loss_config: LossConfig object
        model_arch: Model architecture to use
        num_epochs: Number of epochs to train

    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Testing {loss_name.upper()}")
    print(f"{'='*70}")

    try:
        # Data setup
        data_dir = project_root / 'CDONData'
        stats_path = project_root / 'configs' / 'cdon_stats.json'

        # Create normalizer
        normalizer = CDONNormalization(stats_path=str(stats_path))

        # Check if this is a combined loss (needs dual-batch for DeepONet)
        is_combined = loss_config.loss_type == 'combined'

        if model_arch == 'deeponet' and is_combined:
            print("📊 Creating dual-batch loaders (per-timestep + sequence)...")

            # Per-timestep datasets
            per_ts_train_dataset = CDONDataset(
                data_dir=str(data_dir),
                split='train',
                normalize=normalizer,
                mode='per_timestep',
                use_causal_sequence=True,
                signal_length=4000
            )
            per_ts_val_dataset = CDONDataset(
                data_dir=str(data_dir),
                split='test',
                normalize=normalizer,
                mode='per_timestep',
                use_causal_sequence=True,
                signal_length=4000
            )

            # Sequence datasets
            seq_train_dataset = CDONDataset(
                data_dir=str(data_dir),
                split='train',
                normalize=normalizer,
                mode='sequence',
                use_causal_sequence=False,
                signal_length=4000
            )
            seq_val_dataset = CDONDataset(
                data_dir=str(data_dir),
                split='test',
                normalize=normalizer,
                mode='sequence',
                use_causal_sequence=False,
                signal_length=4000
            )

            # Create loaders
            per_ts_train_loader = DataLoader(
                per_ts_train_dataset, batch_size=32, shuffle=True, num_workers=0
            )
            per_ts_val_loader = DataLoader(
                per_ts_val_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            seq_train_loader = DataLoader(
                seq_train_dataset, batch_size=2, shuffle=True, num_workers=0
            )
            seq_val_loader = DataLoader(
                seq_val_dataset, batch_size=2, shuffle=False, num_workers=0
            )

            print(f"  ✓ Per-timestep train: {len(per_ts_train_dataset):,} samples")
            print(f"  ✓ Sequence train: {len(seq_train_dataset)} samples")

        else:
            print("📊 Creating sequence-only loaders...")

            # Sequence datasets only
            seq_train_dataset = CDONDataset(
                data_dir=str(data_dir),
                split='train',
                normalize=normalizer,
                mode='sequence',
                use_causal_sequence=False if model_arch != 'deeponet' else True,
                signal_length=4000
            )
            seq_val_dataset = CDONDataset(
                data_dir=str(data_dir),
                split='test',
                normalize=normalizer,
                mode='sequence',
                use_causal_sequence=False if model_arch != 'deeponet' else True,
                signal_length=4000
            )

            seq_train_loader = DataLoader(
                seq_train_dataset, batch_size=2, shuffle=True, num_workers=0
            )
            seq_val_loader = DataLoader(
                seq_val_dataset, batch_size=2, shuffle=False, num_workers=0
            )

            print(f"  ✓ Sequence train: {len(seq_train_dataset)} samples")

        # Create model
        model = create_model(model_arch)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created ({num_params:,} parameters)")

        # Create loss
        criterion = create_loss(loss_config)
        print(f"✓ Loss created: {type(criterion).__name__}")

        # Create training config
        config = TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=1e-3,
            optimizer_type='soap',
            batch_size=2,
            weight_decay=1e-4,
            scheduler_type='cosine',
            cosine_eta_min=1e-6,
            eval_metrics=['mse', 'spectrum_error'],
            eval_frequency=1,
            checkpoint_dir=f'test_checkpoints/{model_arch}_{loss_name}',
            save_best=False,
            save_latest=False,
            device='cpu',
            num_workers=0,
            verbose=False  # Reduce output
        )

        # Create trainer
        if model_arch == 'deeponet' and is_combined:
            trainer = SimpleTrainer(
                model=model,
                per_timestep_train_loader=per_ts_train_loader,
                sequence_train_loader=seq_train_loader,
                per_timestep_val_loader=per_ts_val_loader,
                sequence_val_loader=seq_val_loader,
                config=config,
                loss_config=loss_config,
                experiment_name=f'{model_arch}_{loss_name}'
            )
            print("✓ Trainer initialized with DUAL-BATCH mode")
        else:
            trainer = SimpleTrainer(
                model=model,
                train_loader=seq_train_loader,
                val_loader=seq_val_loader,
                config=config,
                loss_config=loss_config,
                experiment_name=f'{model_arch}_{loss_name}'
            )
            print("✓ Trainer initialized with SEQUENCE mode")

        # Train
        print(f"\n🚀 Training for {num_epochs} epochs...")
        results = trainer.train()

        # Verify results
        assert 'best_val_loss' in results, "Missing best_val_loss in results"
        assert 'val_history' in results, "Missing val_history in results"
        assert len(results['val_history']) == num_epochs, f"Expected {num_epochs} epochs, got {len(results['val_history'])}"

        final_loss = results['val_history'][-1]['loss']
        print(f"\n✅ {loss_name.upper()} test PASSED!")
        print(f"   Final val loss: {final_loss:.6f}")

        return True

    except Exception as e:
        print(f"\n❌ {loss_name.upper()} test FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive tests for all loss types."""
    print("="*70)
    print("COMPREHENSIVE LOSS FUNCTION TEST SUITE")
    print("="*70)
    print(f"Project root: {project_root}")
    print(f"Device: CPU (test mode)")
    print(f"Epochs per test: 2")
    print("="*70)

    # Test configurations
    tests = [
        ('baseline', BASELINE_CONFIG),
        ('bsp', BSP_CONFIG),
        ('log-bsp', LOG_BSP_CONFIG),
        ('sa-bsp-perbin', SA_BSP_PERBIN_CONFIG),
        ('sa-bsp-global', SA_BSP_GLOBAL_CONFIG),
        ('sa-bsp-combined', SA_BSP_COMBINED_CONFIG),
    ]

    results = {}

    for loss_name, loss_config in tests:
        passed = test_loss_type(loss_name, loss_config, num_epochs=2)
        results[loss_name] = passed

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for loss_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{loss_name:<20} {status}")

    all_passed = all(results.values())

    print("="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("   The codebase is working correctly.")
        print("   Notebook should work in Colab.")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED!")
        failed = [name for name, passed in results.items() if not passed]
        print(f"   Failed tests: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    exit(main())
