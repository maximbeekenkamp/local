"""
Standalone script for spectral bias analysis of trained models.

Loads trained model checkpoints, generates predictions on validation data,
and creates spectral bias comparison plots.

Usage:
    # Analyze all models from checkpoint directory
    python scripts/analyze_spectral_bias.py \\
        --checkpoint-dir checkpoints/ \\
        --data-dir data/CDONData \\
        --output-dir figures/spectral_analysis

    # Analyze specific models with pattern
    python scripts/analyze_spectral_bias.py \\
        --models deeponet fno unet \\
        --checkpoint-pattern "checkpoints/{model}_real_50epochs/best_model.pt" \\
        --num-samples 100
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.models.model_factory import create_model
from src.core.visualization.spectral_analysis import (
    plot_spectral_bias_comparison,
    compute_spectral_bias_metric,
    plot_spectral_bias_metrics
)

console = Console()


def find_checkpoint(checkpoint_dir: Path, model_name: str, pattern: str = None) -> Path:
    """
    Find checkpoint file for a given model.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name of model architecture
        pattern: Optional pattern for checkpoint path

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    if pattern:
        # Use custom pattern
        checkpoint_path = Path(pattern.format(model=model_name))
    else:
        # Default pattern: checkpoints/{model}_baseline/best_model.pt
        checkpoint_path = checkpoint_dir / f"{model_name}_baseline" / "best_model.pt"

    if not checkpoint_path.exists():
        # Try alternative patterns
        alternatives = [
            checkpoint_dir / f"{model_name}_real_50epochs" / "best_model.pt",
            checkpoint_dir / f"{model_name}" / "best_model.pt",
            checkpoint_dir / f"best_model_{model_name}.pt"
        ]

        for alt_path in alternatives:
            if alt_path.exists():
                checkpoint_path = alt_path
                break

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}. "
            f"Searched: {checkpoint_path} and alternatives"
        )

    return checkpoint_path


def load_trained_model(arch: str, checkpoint_path: Path, device: str = 'cpu'):
    """
    Load trained model from checkpoint.

    Args:
        arch: Model architecture name
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Create model
    model = create_model(arch)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to eval mode
    model.eval()
    model.to(device)

    console.print(f"✓ Loaded {arch} from {checkpoint_path.name}")

    return model


def create_dataloader(data_dir: str, batch_size: int = 16, num_workers: int = 0):
    """
    Create validation dataloader.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of dataloader workers

    Returns:
        Validation DataLoader
    """
    # Create normalization
    stats_path = project_root / 'configs' / 'cdon_stats.json'
    normalizer = CDONNormalization(stats_path=str(stats_path))

    # Create validation dataset
    val_dataset = CDONDataset(
        data_dir=data_dir,
        split='test',
        normalize=normalizer
    )

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return val_loader


def generate_predictions(
    models: Dict[str, torch.nn.Module],
    dataloader: DataLoader,
    num_samples: int,
    device: str = 'cpu'
) -> tuple:
    """
    Generate predictions from all models on validation data.

    Args:
        models: Dictionary of {model_name: model}
        dataloader: Validation dataloader
        num_samples: Maximum number of samples to process
        device: Device for inference

    Returns:
        Tuple of (predictions_dict, ground_truths)
        predictions_dict: {model_name: concatenated_predictions}
        ground_truths: concatenated ground truth tensors
    """
    predictions = {name: [] for name in models.keys()}
    ground_truths = []

    samples_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating predictions..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches")
    ) as progress:

        task = progress.add_task("Predicting", total=len(dataloader))

        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                # Move to device
                batch_inputs = batch_inputs.to(device)

                # Generate predictions from each model
                for model_name, model in models.items():
                    pred = model(batch_inputs)
                    predictions[model_name].append(pred.cpu())

                # Store ground truth
                ground_truths.append(batch_targets)

                # Update progress
                progress.update(task, advance=1)

                # Check if we've processed enough samples
                samples_processed += batch_inputs.shape[0]
                if samples_processed >= num_samples:
                    break

    # Concatenate all batches
    predictions_concat = {
        name: torch.cat(preds, dim=0) for name, preds in predictions.items()
    }
    ground_truths_concat = torch.cat(ground_truths, dim=0)

    # Truncate to exact num_samples
    predictions_concat = {
        name: preds[:num_samples] for name, preds in predictions_concat.items()
    }
    ground_truths_concat = ground_truths_concat[:num_samples]

    console.print(f"✓ Generated predictions for {len(predictions_concat)} models")
    console.print(f"  Samples: {ground_truths_concat.shape[0]}")

    return predictions_concat, ground_truths_concat


def main():
    """Main entry point for spectral bias analysis."""
    parser = argparse.ArgumentParser(
        description='Spectral bias analysis of trained neural operator models'
    )

    # Model selection
    parser.add_argument(
        '--models',
        nargs='+',
        default=['deeponet', 'fno', 'unet'],
        help='List of models to analyze (default: deeponet fno unet)'
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/CDONData',
        help='Path to validation data directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of validation samples to use (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/',
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--checkpoint-pattern',
        type=str,
        default=None,
        help='Pattern for checkpoint paths (use {model} placeholder)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/spectral_analysis',
        help='Directory to save output plots and metrics'
    )

    # Analysis arguments
    parser.add_argument(
        '--n-bins',
        type=int,
        default=32,
        help='Number of frequency bins (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for inference (cpu or cuda)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Spectral Bias Analysis[/bold cyan]")
    console.print(f"Models: {', '.join(args.models)}")
    console.print(f"Data: {args.data_dir}")
    console.print(f"Samples: {args.num_samples}")
    console.print(f"Output: {output_dir}\n")

    # Load models
    console.print("[bold]Loading models...[/bold]")
    checkpoint_dir = Path(args.checkpoint_dir)
    models = {}

    for model_name in args.models:
        try:
            checkpoint_path = find_checkpoint(
                checkpoint_dir,
                model_name,
                args.checkpoint_pattern
            )
            model = load_trained_model(model_name, checkpoint_path, args.device)
            models[model_name] = model
        except FileNotFoundError as e:
            console.print(f"[yellow]Warning:[/yellow] {e}")
            console.print(f"  Skipping {model_name}")
            continue
        except Exception as e:
            console.print(f"[red]Error loading {model_name}:[/red] {e}")
            continue

    if not models:
        console.print("[red]No models loaded successfully. Exiting.[/red]")
        return 1

    # Create dataloader
    console.print(f"\n[bold]Loading validation data...[/bold]")
    try:
        val_loader = create_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=0
        )
        console.print(f"✓ Loaded {len(val_loader.dataset)} validation samples")
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        return 1

    # Generate predictions
    console.print(f"\n[bold]Generating predictions...[/bold]")
    predictions, ground_truths = generate_predictions(
        models=models,
        dataloader=val_loader,
        num_samples=args.num_samples,
        device=args.device
    )

    # Plot spectral bias comparison
    console.print(f"\n[bold]Generating spectral bias plots...[/bold]")

    # Main comparison plot
    plot_spectral_bias_comparison(
        predictions=predictions,
        ground_truth=ground_truths,
        title='Frequency Spectrum Comparison: Model Predictions vs Ground Truth',
        save_path=str(output_dir / 'spectral_bias_comparison.png'),
        n_bins=args.n_bins,
        show_uncertainty=True
    )
    console.print(f"✓ Saved spectral comparison plot")

    # Compute metrics
    console.print(f"\n[bold]Computing spectral bias metrics...[/bold]")
    metrics_dict = {}

    for model_name in models.keys():
        metrics = compute_spectral_bias_metric(
            prediction=predictions[model_name],
            ground_truth=ground_truths,
            n_bins=args.n_bins
        )
        metrics_dict[model_name] = metrics

        console.print(f"  {model_name.upper()}:")
        console.print(f"    Low freq error: {metrics['low_freq_error']:.4f}")
        console.print(f"    Mid freq error: {metrics['mid_freq_error']:.4f}")
        console.print(f"    High freq error: {metrics['high_freq_error']:.4f}")
        console.print(f"    Spectral bias ratio: {metrics['spectral_bias_ratio']:.4f}")

    # Plot metrics comparison
    plot_spectral_bias_metrics(
        metrics_dict=metrics_dict,
        save_path=str(output_dir / 'spectral_bias_metrics.png')
    )
    console.print(f"✓ Saved metrics comparison plot")

    # Save metrics to JSON
    metrics_path = output_dir / 'spectral_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    console.print(f"✓ Saved metrics to {metrics_path}")

    console.print(f"\n[bold green]Analysis complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
