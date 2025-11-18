"""
Main training script for baseline neural operator models.

Trains DeepONet, FNO, or UNet on CDON dataset with configurable hyperparameters.

Usage:
    # Train on dummy data
    python scripts/train_baseline.py --arch deeponet --use-dummy --epochs 100

    # Train on real data
    python scripts/train_baseline.py --arch fno --data-dir data/cdon --epochs 200

    # Use custom config
    python scripts/train_baseline.py --arch unet --config configs/my_training_config.json
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models.model_factory import create_model
from src.core.data_processing.cdon_dataset import create_cdon_dataloaders
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig
from configs.loss_config import BASELINE_CONFIG, BSP_CONFIG, SA_BSP_CONFIG


def create_dual_dataloaders(
    data_dir: str,
    arch: str,
    batch_size_per_timestep: int = 32,
    batch_size_sequence: int = 4,
    num_workers: int = 4,
    use_dummy: bool = False
) -> tuple:
    """
    Create dual dataloaders for dual-batch training.

    Uses create_cdon_dataloaders() factory to create per-timestep and sequence loaders.
    - Per-timestep loader: [N*T, 4000] samples with time coordinates (shuffled)
    - Sequence loader:
      * DeepONet: [N, T, 4000] causal zero-padded sequences (NOT shuffled)
      * FNO/UNet: [N, 1, 4000] full sequences (NOT shuffled)

    Args:
        data_dir: Path to data directory
        arch: Model architecture ('deeponet', 'fno', or 'unet')
        batch_size_per_timestep: Batch size for per-timestep loader
        batch_size_sequence: Batch size for sequence loader
        num_workers: Number of dataloader workers
        use_dummy: Whether to use dummy data

    Returns:
        Tuple of (per_timestep_train_loader, per_timestep_val_loader,
                  sequence_train_loader, sequence_val_loader)
    """
    # Determine data path
    if use_dummy:
        data_path = project_root / 'data' / 'dummy_cdon'
    else:
        data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Use factory function from dataset module
    stats_path = project_root / 'configs' / 'cdon_stats.json'

    # Enable causal sequences for DeepONet, disable for FNO/UNet
    use_causal_sequence = (arch.lower() == 'deeponet')

    per_timestep_train, per_timestep_val, sequence_train, sequence_val = (
        create_cdon_dataloaders(
            data_dir=str(data_path),
            batch_size_per_timestep=batch_size_per_timestep,
            batch_size_sequence=batch_size_sequence,
            use_dummy=use_dummy,
            stats_path=str(stats_path),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            use_causal_sequence=use_causal_sequence
        )
    )

    return per_timestep_train, per_timestep_val, sequence_train, sequence_val


def train_baseline(
    arch: str,
    data_dir: str = 'data/cdon',
    use_dummy: bool = False,
    config: TrainingConfig = None,
    loss_type: str = 'baseline',
    experiment_name: str = None
) -> dict:
    """
    Train a baseline neural operator model with dual-batch training.

    For DeepONet:
    - Uses per-timestep loader for MSE loss (shuffled, ~320K samples)
    - Uses sequence loader for BSP loss (not shuffled, ~100 samples)
    - Combines: loss = lambda_mse * mse_loss + lambda_bsp * bsp_loss

    For FNO/UNet:
    - Uses only sequence loader (not shuffled, ~100 samples)
    - Computes loss from full-sequence predictions

    Args:
        arch: Model architecture ('deeponet', 'fno', 'unet')
        data_dir: Path to data directory
        use_dummy: Whether to use dummy data
        config: Training configuration (if None, uses defaults)
        loss_type: Loss function type ('baseline', 'bsp', 'sa-bsp')
        experiment_name: Name for this experiment (if None, uses arch name)

    Returns:
        Dictionary with training results and history
    """
    console = Console()

    # Create config if not provided
    if config is None:
        config = TrainingConfig()

    # Set experiment name
    if experiment_name is None:
        experiment_name = f"{arch}_baseline"

    # Print setup information
    console.print("\n[bold cyan]═══ Training Setup ═══[/bold cyan]")
    console.print(f"Architecture: [bold]{arch}[/bold]")
    console.print(f"Data: [bold]{'Dummy' if use_dummy else 'Real'}[/bold]")
    console.print(f"Experiment: [bold]{experiment_name}[/bold]")
    console.print(f"Epochs: [bold]{config.num_epochs}[/bold]")
    console.print(f"Learning rate: [bold]{config.learning_rate}[/bold]")
    console.print(f"Batch size: [bold]{config.batch_size}[/bold]")
    console.print(f"Scheduler: [bold]{config.scheduler_type}[/bold]")

    # Create dual dataloaders
    console.print("\n[bold cyan]Loading data (dual-batch)...[/bold cyan]")
    per_timestep_train, per_timestep_val, sequence_train, sequence_val = create_dual_dataloaders(
        data_dir=data_dir,
        arch=arch,  # Pass architecture to determine causality
        batch_size_per_timestep=config.batch_size,
        batch_size_sequence=4,  # Keep sequence batch size small
        num_workers=config.num_workers,
        use_dummy=use_dummy
    )
    console.print(f"✓ Per-timestep train samples: {len(per_timestep_train.dataset):,}")
    console.print(f"✓ Per-timestep val samples: {len(per_timestep_val.dataset):,}")
    console.print(f"✓ Sequence train samples: {len(sequence_train.dataset)}")
    console.print(f"✓ Sequence val samples: {len(sequence_val.dataset)}")

    # Create model
    console.print(f"\n[bold cyan]Creating {arch} model...[/bold cyan]")
    model = create_model(arch)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"✓ Model created ({num_params:,} parameters)")

    # Select loss configuration
    loss_config_map = {
        'baseline': BASELINE_CONFIG,
        'bsp': BSP_CONFIG,
        'sa-bsp': SA_BSP_CONFIG
    }
    loss_config = loss_config_map[loss_type]

    # Create trainer with dual loaders
    console.print("\n[bold cyan]Initializing trainer...[/bold cyan]")
    console.print(f"Loss function: [bold]{loss_type}[/bold] - {loss_config.description}")

    # Detect if DeepONet for conditional per-timestep loader
    is_deeponet = arch.lower() == 'deeponet'

    if is_deeponet:
        # DeepONet: use both per-timestep and sequence loaders
        trainer = SimpleTrainer(
            model=model,
            per_timestep_train_loader=per_timestep_train,
            sequence_train_loader=sequence_train,
            per_timestep_val_loader=per_timestep_val,
            sequence_val_loader=sequence_val,
            config=config,
            loss_config=loss_config,
            experiment_name=experiment_name
        )
    else:
        # FNO/UNet: use only sequence loaders
        trainer = SimpleTrainer(
            model=model,
            per_timestep_train_loader=None,
            sequence_train_loader=seq_train,
            per_timestep_val_loader=None,
            sequence_val_loader=seq_val,
            config=config,
            loss_config=loss_config,
            experiment_name=experiment_name
        )

    console.print(f"✓ Trainer initialized on device: {trainer.device}")

    # Train model
    console.print("\n[bold cyan]═══ Starting Training ═══[/bold cyan]")
    results = trainer.train()

    # Print final results
    console.print("\n[bold cyan]═══ Final Results ═══[/bold cyan]")

    # Create results table
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Architecture", arch)
    table.add_row("Parameters", f"{num_params:,}")
    table.add_row("Total Epochs", str(config.num_epochs))
    table.add_row("Best Val Loss", f"{results['best_val_loss']:.6f}")

    # Get final epoch metrics
    final_train = results['train_history'][-1]
    final_val = results['val_history'][-1]

    table.add_row("Final Train Loss", f"{final_train['loss']:.6f}")
    table.add_row("Final Val Loss", f"{final_val['loss']:.6f}")

    # Add DeepONet-specific metrics if applicable
    if is_deeponet and 'mse_loss' in final_train:
        table.add_row("Final Train MSE Loss", f"{final_train['mse_loss']:.6f}")
        table.add_row("Final Train BSP Loss", f"{final_train['bsp_loss']:.6f}")
        if 'mse_loss' in final_val:
            table.add_row("Final Val MSE Loss", f"{final_val['mse_loss']:.6f}")
            table.add_row("Final Val BSP Loss", f"{final_val['bsp_loss']:.6f}")

    table.add_row("Checkpoint Dir", str(trainer.checkpoint_dir))

    console.print(table)

    # Return results
    return {
        'arch': arch,
        'experiment_name': experiment_name,
        'num_params': num_params,
        'config': config.to_dict(),
        'results': results,
        'checkpoint_dir': str(trainer.checkpoint_dir)
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train baseline neural operator models on CDON dataset'
    )

    # Model arguments
    parser.add_argument(
        '--arch',
        type=str,
        required=True,
        choices=['deeponet', 'fno', 'unet'],
        help='Model architecture to train'
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/cdon',
        help='Path to data directory (default: data/cdon)'
    )
    parser.add_argument(
        '--use-dummy',
        action='store_true',
        help='Use dummy data instead of real data'
    )

    # Training arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training config JSON file (optional)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training (default: cuda)'
    )
    parser.add_argument(
        '--loss-type',
        type=str,
        default='baseline',
        choices=['baseline', 'bsp', 'sa-bsp'],
        help='Loss function type (default: baseline)'
    )

    # Experiment arguments
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for this experiment (default: {arch}_baseline)'
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()

    # Override config with command-line arguments
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    # Validate config
    config.validate()

    # Train model
    try:
        results = train_baseline(
            arch=args.arch,
            data_dir=args.data_dir,
            use_dummy=args.use_dummy,
            config=config,
            loss_type=args.loss_type,
            experiment_name=args.experiment_name
        )
        return 0

    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]Error during training:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
