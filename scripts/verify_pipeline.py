"""
Pipeline verification script for quick validation of baseline models.

Runs quick training tests on all models and reports pass/fail status with metrics.

Usage:
    # Verify all models on dummy data (quick)
    python scripts/verify_pipeline.py --all --dummy --epochs 5

    # Verify specific model
    python scripts/verify_pipeline.py --arch fno --dummy --epochs 5

    # Verify on real data
    python scripts/verify_pipeline.py --all --real --epochs 10

    # Quick check (3 epochs)
    python scripts/verify_pipeline.py --all --dummy --quick
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import traceback

import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.models.model_factory import create_model
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig
from configs.loss_config import BASELINE_CONFIG


console = Console()


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


def verify_single_model(
    arch: str,
    data_dir: str,
    use_dummy: bool,
    epochs: int
) -> Dict[str, Any]:
    """
    Verify a single model architecture.

    Args:
        arch: Model architecture ('deeponet', 'fno', 'unet')
        data_dir: Path to data directory
        use_dummy: Whether using dummy data
        epochs: Number of training epochs

    Returns:
        Dictionary with verification results:
        - status: 'PASS' or 'FAIL'
        - val_loss: Final validation loss (or None if failed)
        - mse: Final MSE (or None if failed)
        - training_time: Time taken in seconds
        - error_message: Error message if failed (or None if passed)
    """
    result = {
        'arch': arch,
        'status': 'FAIL',
        'val_loss': None,
        'mse': None,
        'spectrum_error': None,
        'training_time': 0.0,
        'error_message': None
    }

    try:
        start_time = time.time()

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(data_dir, batch_size=16)

        # Create model
        model = create_model(arch)

        # Create config
        config = TrainingConfig(
            num_epochs=epochs,
            learning_rate=1e-3,
            batch_size=16,
            device='cpu',
            num_workers=0,
            verbose=False,
            save_best=False,
            save_latest=False
        )

        # Create trainer
        trainer = SimpleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            loss_config=BASELINE_CONFIG,
            experiment_name=f'verify_{arch}'
        )

        # Train
        training_results = trainer.train()

        end_time = time.time()

        # Extract final metrics
        final_val = training_results['val_history'][-1]

        result['status'] = 'PASS'
        result['val_loss'] = final_val['loss']
        result['mse'] = final_val['mse']
        result['spectrum_error'] = final_val.get('spectrum_error', None)
        result['training_time'] = end_time - start_time

        # Additional validation checks
        if not torch.isfinite(torch.tensor(final_val['loss'])):
            result['status'] = 'FAIL'
            result['error_message'] = 'Non-finite loss value (NaN or Inf)'

        if final_val['loss'] > 10.0:
            result['status'] = 'FAIL'
            result['error_message'] = f'Loss too high: {final_val["loss"]:.4f}'

    except Exception as e:
        result['status'] = 'FAIL'
        result['error_message'] = str(e)
        console.print(f"[red]Error in {arch}:[/red] {e}")
        if console.is_terminal:
            traceback.print_exc()

    return result


def verify_all_models(
    data_dir: str,
    use_dummy: bool,
    epochs: int
) -> List[Dict[str, Any]]:
    """
    Verify all three baseline models.

    Args:
        data_dir: Path to data directory
        use_dummy: Whether using dummy data
        epochs: Number of training epochs

    Returns:
        List of verification results for each model
    """
    models = ['deeponet', 'fno', 'unet']
    results = []

    for arch in models:
        console.print(f"\n[bold cyan]Verifying {arch.upper()}...[/bold cyan]")
        result = verify_single_model(arch, data_dir, use_dummy, epochs)
        results.append(result)

    return results


def print_verification_summary(results: List[Dict[str, Any]]):
    """
    Print verification results in a nice table.

    Args:
        results: List of verification results
    """
    # Create table
    table = Table(title="Pipeline Verification Results", show_header=True)

    table.add_column("Model", style="cyan", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Val Loss", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("Spectrum Error", justify="right")
    table.add_column("Time (s)", justify="right")

    for result in results:
        # Format status with color
        if result['status'] == 'PASS':
            status = "[green]✓ PASS[/green]"
        else:
            status = "[red]✗ FAIL[/red]"

        # Format metrics
        if result['val_loss'] is not None:
            val_loss = f"{result['val_loss']:.4f}"
        else:
            val_loss = "N/A"

        if result['mse'] is not None:
            mse = f"{result['mse']:.4f}"
        else:
            mse = "N/A"

        if result['spectrum_error'] is not None:
            spectrum_error = f"{result['spectrum_error']:.4f}"
        else:
            spectrum_error = "N/A"

        training_time = f"{result['training_time']:.1f}"

        table.add_row(
            result['arch'].upper(),
            status,
            val_loss,
            mse,
            spectrum_error,
            training_time
        )

    console.print("\n")
    console.print(table)

    # Print error messages if any
    failures = [r for r in results if r['status'] == 'FAIL']
    if failures:
        console.print("\n[bold red]Failures:[/bold red]")
        for result in failures:
            if result['error_message']:
                console.print(
                    f"  • {result['arch'].upper()}: {result['error_message']}"
                )

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    console.print(f"  Passed: {passed}/{total}")

    if passed == total:
        console.print("[bold green]All verifications passed! ✓[/bold green]")
    else:
        console.print(f"[bold red]{total - passed} verification(s) failed[/bold red]")


def main():
    """Main entry point for verification script."""
    parser = argparse.ArgumentParser(
        description='Verify neural operator pipeline with quick training tests'
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--arch',
        type=str,
        choices=['deeponet', 'fno', 'unet'],
        help='Verify specific architecture'
    )
    model_group.add_argument(
        '--all',
        action='store_true',
        help='Verify all architectures'
    )

    # Data selection
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--dummy',
        action='store_true',
        help='Use dummy data'
    )
    data_group.add_argument(
        '--real',
        action='store_true',
        help='Use real CDON data'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick check with 3 epochs'
    )

    # Data directory
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Custom data directory path'
    )

    args = parser.parse_args()

    # Determine epochs
    epochs = 3 if args.quick else args.epochs

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif args.dummy:
        data_dir = str(project_root / 'data' / 'dummy_cdon')
    else:  # args.real
        data_dir = str(project_root / 'data' / 'CDONData')

    # Print header
    console.print(Panel.fit(
        "[bold]Neural Operator Pipeline Verification[/bold]\n"
        f"Data: {'Dummy' if args.dummy else 'Real'}\n"
        f"Epochs: {epochs}",
        border_style="cyan"
    ))

    # Verify data directory exists
    if not Path(data_dir).exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        return 1

    # Run verification
    if args.all:
        results = verify_all_models(data_dir, args.dummy, epochs)
    else:
        console.print(f"\n[bold cyan]Verifying {args.arch.upper()}...[/bold cyan]")
        result = verify_single_model(args.arch, data_dir, args.dummy, epochs)
        results = [result]

    # Print summary
    print_verification_summary(results)

    # Return exit code
    all_passed = all(r['status'] == 'PASS' for r in results)
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
