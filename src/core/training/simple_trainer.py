"""
Simple trainer for neural operator models.

Provides a clean training loop with:
- CosineAnnealingLR scheduler
- Field error + spectrum error metrics
- Checkpoint management (best + latest)
- Rich progress bars for visual feedback
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional
from pathlib import Path

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.console import Console

from .optimizers.optimizer_factory import create_optimizer
from ..evaluation.metrics import (
    RelativeL2Loss,
    compute_field_error,
    compute_spectrum_error_1d
)
from ..evaluation.loss_factory import create_loss
from ..evaluation.adaptive_spectral_loss import SelfAdaptiveBSPLoss
from configs.training_config import TrainingConfig
from configs.loss_config import LossConfig, BASELINE_CONFIG


class SimpleTrainer:
    """
    Simple trainer for neural operator models.

    Handles:
    - Training loop with automatic differentiation
    - Validation with multiple metrics
    - Checkpoint saving (best + latest)
    - Learning rate scheduling
    - Progress bars for visual feedback

    Example:
        >>> config = TrainingConfig(num_epochs=100, learning_rate=1e-3)
        >>> trainer = SimpleTrainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_config: LossConfig,
        experiment_name: str = 'experiment'
    ):
        """
        Initialize trainer.

        Args:
            model: Neural operator model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            loss_config: Loss function configuration (required)
                Use BASELINE_CONFIG for RelativeL2Loss
                Use BSP_CONFIG for BSP loss
                Use SA_BSP_CONFIG for SA-BSP loss
            experiment_name: Name for this experiment (used in checkpoint paths)

        Note:
            For SA-BSP loss, this automatically creates a separate optimizer
            for adaptive weights (see self.weight_optimizer)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_name = experiment_name

        # Device setup
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # Loss function (required parameter)
        self.criterion = create_loss(loss_config)

        # Optimizer - created via factory to support adam, adamw, soap
        self.optimizer = create_optimizer(
            optimizer_type=config.optimizer_type,
            model_parameters=self.model.parameters(),
            config=config
        )

        # Weight optimizer for SA-BSP loss (if applicable)
        self.weight_optimizer = None
        self.adapt_mode = None  # Store adapt_mode for SA-PINNs gradient handling
        if self._is_sa_bsp_loss(self.criterion):
            self.weight_optimizer = Adam(
                self.criterion.spectral_loss.adaptive_weights.parameters(),
                lr=config.learning_rate
            )
            # Store adapt_mode for SA-PINNs style optimization
            self.adapt_mode = self.criterion.spectral_loss.adapt_mode

        # Scheduler setup
        self.scheduler = self._create_scheduler()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []

        # Console for rich output
        self.console = Console()

    def _is_sa_bsp_loss(self, criterion) -> bool:
        """
        Check if loss function uses SA-BSP (needs separate weight optimizer).

        Args:
            criterion: Loss function module

        Returns:
            True if criterion is CombinedLoss with SA-BSP spectral component
        """
        # Import here to avoid circular dependency
        from ..evaluation.loss_factory import CombinedLoss

        if isinstance(criterion, CombinedLoss):
            return isinstance(criterion.spectral_loss, SelfAdaptiveBSPLoss)
        return False

    def _update_adaptive_weights(self) -> None:
        """
        Update adaptive weights using SA-PINNs style optimization.

        Applies gradient negation based on adapt_mode:
        - 'per-bin': Negate ALL weight gradients (ascent on loss to emphasize hard bins)
        - 'global': Standard gradients (descent on loss for MSE/BSP balance)
        - 'hierarchical': Negate per-bin gradients (indices 1:), keep global standard (index 0)

        This implements the saddle-point optimization from SA-PINNs paper:
        - Model: min_θ L(θ, λ) (standard gradient descent)
        - Weights: max_λ L(θ, λ) (gradient ascent via negated gradients)
        """
        if self.weight_optimizer is None:
            return

        # Get adaptive weight parameters
        adaptive_params = list(self.criterion.spectral_loss.adaptive_weights.parameters())

        if self.adapt_mode == 'per-bin':
            # SA-PINNs style: NEGATE all gradients for per-bin weights
            # This performs gradient ascent on the loss, increasing weights for
            # difficult frequency bins (typically high frequencies)
            for param in adaptive_params:
                if param.grad is not None:
                    param.grad = -param.grad

        elif self.adapt_mode == 'global':
            # Standard gradients (no negation)
            # Global weight balances MSE vs BSP, should minimize total loss
            pass

        elif self.adapt_mode == 'hierarchical':
            # Mixed approach: global uses standard, per-bin uses negated
            # weights[0] = global weight (standard gradient)
            # weights[1:] = per-bin weights (negated gradients)
            for param in adaptive_params:
                if param.grad is not None:
                    # Negate all except first element (global weight)
                    param.grad[1:] = -param.grad[1:]

        # Update weights with (possibly negated) gradients
        self.weight_optimizer.step()

    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.scheduler_type == 'cosine':
            # T_max = total number of training steps
            if self.config.cosine_t_max is None:
                t_max = self.config.num_epochs * len(self.train_loader)
            else:
                t_max = self.config.cosine_t_max

            return CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=self.config.cosine_eta_min
            )

        elif self.config.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                min_lr=self.config.plateau_min_lr,
                verbose=self.config.verbose
            )

        elif self.config.scheduler_type == 'none':
            return None

        else:
            raise ValueError(
                f"Unknown scheduler type: '{self.config.scheduler_type}'"
            )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics:
            - 'loss': Average training loss
            - 'field_error': Average field error
        """
        self.model.train()

        total_loss = 0.0
        total_field_error = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            if self.weight_optimizer is not None:
                self.weight_optimizer.zero_grad()

            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping for numerical stability (prevents gradient explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update model parameters (standard gradient descent)
            self.optimizer.step()

            # Update adaptive weights (SA-PINNs style with negated gradients)
            self._update_adaptive_weights()

            # Step scheduler (for cosine annealing, step every batch)
            if self.config.scheduler_type == 'cosine':
                self.scheduler.step()

            # Compute metrics
            with torch.no_grad():
                field_error = compute_field_error(outputs, targets)

            # Accumulate
            total_loss += loss.item()
            total_field_error += field_error.item()

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_field_error = total_field_error / num_batches

        return {
            'loss': avg_loss,
            'field_error': avg_field_error
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary with validation metrics:
            - 'loss': Average validation loss
            - 'field_error': Average field error
            - 'spectrum_error': Average spectrum error (if in config)
        """
        self.model.eval()

        total_loss = 0.0
        total_field_error = 0.0
        total_spectrum_error = 0.0
        num_batches = len(self.val_loader)

        compute_spectrum = 'spectrum_error' in self.config.eval_metrics

        for inputs, targets in self.val_loader:
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Compute metrics
            field_error = compute_field_error(outputs, targets)

            if compute_spectrum:
                spectrum_error = compute_spectrum_error_1d(outputs, targets)
                total_spectrum_error += spectrum_error.item()

            # Accumulate
            total_loss += loss.item()
            total_field_error += field_error.item()

        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'field_error': total_field_error / num_batches
        }

        if compute_spectrum:
            metrics['spectrum_error'] = total_spectrum_error / num_batches

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_latest: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics dictionary
            is_best: Whether this is the best model so far
            is_latest: Whether this is the latest model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'experiment_name': self.experiment_name
        }

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            if self.config.verbose:
                self.console.print(
                    f"[bold green]✓[/bold green] Saved best model "
                    f"(val_loss: {metrics['loss']:.4f})"
                )

        # Save latest model
        if is_latest:
            latest_path = self.checkpoint_dir / 'latest_model.pt'
            torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if epoch % self.config.save_frequency == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.config.verbose:
            self.console.print(
                f"[bold green]✓[/bold green] Loaded checkpoint from epoch "
                f"{checkpoint['epoch']}"
            )

        return checkpoint['epoch']

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with training history:
            - 'train_history': List of training metrics per epoch
            - 'val_history': List of validation metrics per epoch
            - 'best_val_loss': Best validation loss achieved
            - 'final_epoch': Final epoch number
        """
        if self.config.verbose:
            self.console.print(f"\n[bold]Training {self.experiment_name}[/bold]")
            self.console.print(f"Device: {self.device}")
            self.console.print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            self.console.print(f"Training samples: {len(self.train_loader.dataset)}")
            self.console.print(f"Validation samples: {len(self.val_loader.dataset)}\n")

        # Main epoch loop with progress bar
        with Progress(
            TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.total}"),
            BarColumn(),
            TextColumn("•"),
            TextColumn("Train Loss: {task.fields[train_loss]:.4f}"),
            TextColumn("•"),
            TextColumn("Val Loss: {task.fields[val_loss]:.4f}"),
            TextColumn("•"),
            TextColumn("LR: {task.fields[lr]:.2e}"),
            TimeElapsedColumn()
        ) as progress:

            epoch_task = progress.add_task(
                "Training",
                total=self.config.num_epochs,
                epoch=0,
                train_loss=0.0,
                val_loss=0.0,
                lr=self.config.learning_rate
            )

            for epoch in range(1, self.config.num_epochs + 1):
                self.current_epoch = epoch

                # Train for one epoch
                train_metrics = self.train_epoch()

                # Validate
                if epoch % self.config.eval_frequency == 0:
                    val_metrics = self.validate()
                else:
                    val_metrics = {'loss': float('inf')}

                # Update learning rate (for plateau scheduler)
                if self.config.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['loss'])

                # Check if best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']

                # Save checkpoints
                if self.config.save_best and is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best=True)

                if self.config.save_latest:
                    self.save_checkpoint(epoch, val_metrics, is_latest=True)

                # Record history
                self.train_history.append(train_metrics)
                self.val_history.append(val_metrics)

                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                progress.update(
                    epoch_task,
                    advance=1,
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    lr=current_lr
                )

        # Print final results
        if self.config.verbose:
            self.console.print(f"\n[bold green]Training complete![/bold green]")
            self.console.print(f"Best validation loss: {self.best_val_loss:.4f}")
            self.console.print(f"Checkpoints saved to: {self.checkpoint_dir}")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch
        }
