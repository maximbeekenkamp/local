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
from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig
from configs.loss_config import BASELINE_CONFIG, BSP_CONFIG, SA_BSP_CONFIG


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    use_dummy: bool = False
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        use_dummy: Whether to use dummy data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Determine data path
    if use_dummy:
        data_path = project_root / 'data' / 'dummy_cdon'
    else:
        data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Create normalization
    stats_path = project_root / 'configs' / 'cdon_stats.json'
    normalizer = CDONNormalization(stats_path=str(stats_path))

    # Create datasets
    train_dataset = CDONDataset(
        data_dir=str(data_path),
        split='train',
        normalize=normalizer
    )

    val_dataset = CDONDataset(
        data_dir=str(data_path),
        split='test',
        normalize=normalizer
    )

    # Create dataloaders
    # pin_memory only works with CUDA, not MPS
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


def train_baseline(
    arch: str,
    data_dir: str = 'data/cdon',
    use_dummy: bool = False,
    config: TrainingConfig = None,
    loss_type: str = 'baseline',
    experiment_name: str = None
) -> dict:
    """
    Train a baseline neural operator model.

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

    # Create dataloaders
    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_dummy=use_dummy
    )
    console.print(f"✓ Train samples: {len(train_loader.dataset)}")
    console.print(f"✓ Val samples: {len(val_loader.dataset)}")

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

    # Create trainer
    console.print("\n[bold cyan]Initializing trainer...[/bold cyan]")
    console.print(f"Loss function: [bold]{loss_type}[/bold] - {loss_config.description}")
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
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
    table.add_row("Final Train Field Error", f"{final_train['field_error']:.6f}")
    table.add_row("Final Val Loss", f"{final_val['loss']:.6f}")
    table.add_row("Final Val Field Error", f"{final_val['field_error']:.6f}")

    if 'spectrum_error' in final_val:
        table.add_row("Final Val Spectrum Error", f"{final_val['spectrum_error']:.6f}")

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
