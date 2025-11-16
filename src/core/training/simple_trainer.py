"""
Simple trainer for neural operator models.

Supports dual-batch training:
- DeepONet: Per-timestep (MSE) + Full-sequence (BSP) batches
- FNO/UNet: Full-sequence batches (both MSE and BSP computed on sequences)

Features:
- CosineAnnealingLR scheduler
- Dual loss computation (MSE + BSP)
- Checkpoint management (best + latest)
- Rich progress bars for visual feedback
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from itertools import cycle

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
    compute_field_error,
    compute_spectrum_error_1d
)
from ..evaluation.loss_factory import create_loss
from ..evaluation.adaptive_spectral_loss import SelfAdaptiveBSPLoss
from configs.training_config import TrainingConfig
from configs.loss_config import LossConfig, BASELINE_CONFIG


class SimpleTrainer:
    """
    Dual-batch trainer for neural operator models.

    Supports two training modes:
    1. **DeepONet**: Dual-batch training with per-timestep (MSE) and sequence (BSP) batches
       - Per-timestep loader: [N*T, 4000] samples with time coordinates (shuffled)
       - Sequence loader: [N, 1, 4000] full sequences (NOT shuffled for BSP consistency)
       - Alternates between loaders each iteration

    2. **FNO/UNet**: Sequence-only training with full sequences for MSE and BSP
       - Sequence loader: [N, 1, 4000] full sequences
       - Single loader training

    Handles:
    - Dual-batch training with loader cycling
    - Separate MSE (per-timestep) and BSP (sequence) loss computation
    - Validation with dual loaders
    - Checkpoint saving (best + latest)
    - Learning rate scheduling
    - Progress bars for visual feedback

    Example:
        >>> config = TrainingConfig(num_epochs=100, learning_rate=1e-3)

        >>> # Simplified API (sequence-only models like FNO/UNet)
        >>> trainer = SimpleTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     loss_config=loss_config
        ... )

        >>> # Dual-loader API (DeepONet with per-timestep + sequence batches)
        >>> trainer = SimpleTrainer(
        ...     model=model,
        ...     per_timestep_train_loader=per_ts_train,
        ...     sequence_train_loader=seq_train,
        ...     per_timestep_val_loader=per_ts_val,
        ...     sequence_val_loader=seq_val,
        ...     config=config,
        ...     loss_config=loss_config
        ... )

        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        per_timestep_train_loader: Optional[DataLoader] = None,
        sequence_train_loader: Optional[DataLoader] = None,
        per_timestep_val_loader: Optional[DataLoader] = None,
        sequence_val_loader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        loss_config: LossConfig = None,
        experiment_name: str = 'experiment',
        # Simplified API (alternative to dual-loader API)
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Initialize dual-batch trainer.

        Args:
            model: Neural operator model to train
            per_timestep_train_loader: Per-timestep training loader (DeepONet) or None (FNO/UNet)
                Format: Dict with 'input', 'target', 'time_coord', 'penalty'
            sequence_train_loader: Full-sequence training loader (all models)
                Format: Tuple (input [B,1,T], target [B,1,T])
            per_timestep_val_loader: Per-timestep validation loader or None
            sequence_val_loader: Full-sequence validation loader
            config: Training configuration
            loss_config: Loss function configuration (required)
            experiment_name: Name for this experiment (used in checkpoint paths)
            train_loader: Simplified API - training loader (used as sequence_train_loader if provided)
            val_loader: Simplified API - validation loader (used as sequence_val_loader if provided)

        Note:
            - For SA-BSP loss, creates separate optimizer for adaptive weights
            - Automatically detects model type (DeepONet vs FNO/UNet) from forward method
            - Can use either dual-loader API (per_timestep_*, sequence_*) or simplified API (train_loader, val_loader)
        """
        # Handle simplified API: if train_loader/val_loader provided, use them as sequence loaders
        if train_loader is not None and sequence_train_loader is None:
            sequence_train_loader = train_loader
        if val_loader is not None and sequence_val_loader is None:
            sequence_val_loader = val_loader

        # Validate that we have at least sequence loaders
        if sequence_train_loader is None:
            raise ValueError(
                "Must provide either 'sequence_train_loader' or 'train_loader'"
            )
        if sequence_val_loader is None:
            raise ValueError(
                "Must provide either 'sequence_val_loader' or 'val_loader'"
            )

        self.model = model
        self.per_timestep_train_loader = per_timestep_train_loader
        self.sequence_train_loader = sequence_train_loader
        self.per_timestep_val_loader = per_timestep_val_loader
        self.sequence_val_loader = sequence_val_loader
        self.config = config
        self.experiment_name = experiment_name

        # Detect model type from forward method
        self.is_deeponet = hasattr(model, 'forward_per_timestep') and hasattr(model, 'forward_sequence')
        self.is_sequence_only = not self.is_deeponet

        # Device setup
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)

        # Loss function (required parameter)
        self.criterion = create_loss(loss_config)

        # Penalty weighting (optional, from reference CausalityDeepONet)
        self.use_penalty_weighting = loss_config.loss_params.get('use_penalty', False)

        # Loss configuration for dual-batch training
        self.loss_config = loss_config

        # Extract lambda weights from loss config (for dual loss combination)
        # These are typically in loss_params: 'lambda_mse', 'lambda_bsp'
        self.lambda_mse = loss_config.loss_params.get('lambda_mse', 1.0)
        self.lambda_bsp = loss_config.loss_params.get('lambda_bsp', 1.0)

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
        else:
            self.adapt_mode = None

        # Scheduler setup - compute t_max based on actual number of steps
        # For dual-batch training with DeepONet: use per-timestep loader (has more samples)
        # For sequence-only: use sequence loader
        if self.is_deeponet and self.per_timestep_train_loader is not None:
            steps_per_epoch = len(self.per_timestep_train_loader)
        else:
            steps_per_epoch = len(self.sequence_train_loader)

        self.scheduler = self._create_scheduler(steps_per_epoch)

        # Weight scheduler for SA-BSP adaptive weights (if applicable)
        # Uses same schedule as model optimizer for synchronized learning rate decay
        self.weight_scheduler = None
        if self.weight_optimizer is not None and self.config.scheduler_type == 'cosine':
            # Calculate T_max same as model scheduler
            if self.config.cosine_t_max is None:
                t_max = self.config.num_epochs * steps_per_epoch
            else:
                t_max = self.config.cosine_t_max

            self.weight_scheduler = CosineAnnealingLR(
                self.weight_optimizer,
                T_max=t_max,
                eta_min=self.config.cosine_eta_min
            )

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

        Applies gradient negation (ascent) based on adapt_mode for competitive dynamics:
        - 'per-bin': Negate ALL gradients (32 frequency weights compete)
        - 'global': Negate ALL gradients (w_mse and w_bsp compete for MSE/BSP balance)
        - 'combined': Negate ALL gradients (w_mse, w_bsp, and per-bin weights all compete)
        - 'fft': Negate ALL gradients (32 frequency weights, optimized in spectral domain)

        This implements the saddle-point optimization from SA-PINNs paper:
        - Model: min_θ L(θ, λ) (standard gradient descent)
        - Weights: max_λ L(θ, λ) (gradient ascent via negated gradients for competitive dynamics)

        Competitive dynamics prevent weight collapse by making weights maximize the loss they weight.
        Weights naturally find equilibrium where harder-to-fit losses get higher emphasis.
        """
        if self.weight_optimizer is None:
            return

        # Get adaptive weight parameters
        adaptive_params = list(self.criterion.spectral_loss.adaptive_weights.parameters())

        if self.adapt_mode == 'per-bin':
            # SA-PINNs style: NEGATE all gradients (ascent on loss)
            # Weights for high-error frequency bins increase, creating competitive emphasis
            # on difficult-to-fit frequencies (typically high frequencies due to spectral bias)
            for param in adaptive_params:
                if param.grad is not None:
                    param.grad = -param.grad

        elif self.adapt_mode == 'global':
            # Competitive dynamics for MSE/BSP balance
            # weights[0] = w_mse (MSE weight)
            # weights[1] = w_bsp (BSP weight)
            # Both use negated gradients (ascent) to learn which is harder to satisfy
            for param in adaptive_params:
                if param.grad is not None:
                    param.grad = -param.grad

        elif self.adapt_mode == 'combined':
            # Full competitive dynamics: all weights compete
            # weights[0] = w_mse (ascent: maximize loss w.r.t. MSE weight)
            # weights[1] = w_bsp (ascent: maximize loss w.r.t. global BSP weight)
            # weights[2:] = per-bin weights (ascent: maximize loss per frequency bin)
            # Result: weights find equilibrium emphasizing hard losses
            for param in adaptive_params:
                if param.grad is not None:
                    param.grad = -param.grad

        elif self.adapt_mode == 'fft':
            # FFT mode: Spectral-domain weight optimization
            # Weights optimize using ONLY spectral loss (not combined loss)
            # 32 per-bin weights use negated gradients (ascent on spectral loss)
            # Model sees combined loss, weights see only BSP for pure spectral adaptation
            for param in adaptive_params:
                if param.grad is not None:
                    param.grad = -param.grad

        # Update weights with (possibly negated) gradients
        self.weight_optimizer.step()

        # Step weight scheduler (cosine annealing for natural weight bound)
        if self.weight_scheduler is not None:
            self.weight_scheduler.step()

    def _get_adaptive_weight_stats(self) -> Dict[str, float]:
        """
        Get adaptive weight statistics for monitoring.

        Returns:
            Dictionary with weight statistics:
            - 'weight_mean': Mean weight value
            - 'weight_std': Standard deviation of weights
            - 'weight_min': Minimum weight value
            - 'weight_max': Maximum weight value
            - For global/combined modes:
              - 'w_mse': MSE weight
              - 'w_bsp': BSP weight
        """
        if not hasattr(self.criterion, 'spectral_loss') or \
           not hasattr(self.criterion.spectral_loss, 'adaptive_weights'):
            return {}

        stats = self.criterion.spectral_loss.adaptive_weights.get_statistics()

        metrics = {
            'weight_mean': stats['mean'],
            'weight_std': stats['std'],
            'weight_min': stats['min'],
            'weight_max': stats['max']
        }

        # For global/combined modes, also track MSE/BSP weights separately
        if self.adapt_mode in ['global', 'combined']:
            weights = stats['weights']
            metrics['w_mse'] = float(weights[0])
            metrics['w_bsp'] = float(weights[1])

        return metrics

    def _create_scheduler(self, steps_per_epoch: int):
        """
        Create learning rate scheduler based on config.

        Args:
            steps_per_epoch: Number of training steps per epoch
        """
        if self.config.scheduler_type == 'cosine':
            # T_max = total number of training steps
            if self.config.cosine_t_max is None:
                t_max = self.config.num_epochs * steps_per_epoch
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
        Train for one epoch with dual-batch training.

        For DeepONet:
        - Alternates between per-timestep (MSE) and sequence (BSP) batches
        - Computes loss_mse from per-timestep predictions
        - Computes loss_bsp from full-sequence predictions
        - Combines: loss = lambda_mse * loss_mse + lambda_bsp * loss_bsp

        For FNO/UNet:
        - Uses only sequence loader
        - Computes loss from full-sequence predictions

        Returns:
            Dictionary with training metrics:
            - 'loss': Average training loss (combined MSE + BSP)
            - 'mse_loss': Average MSE loss (DeepONet only)
            - 'bsp_loss': Average BSP loss (DeepONet only)
        """
        self.model.train()

        total_loss = 0.0
        total_mse_loss = 0.0
        total_bsp_loss = 0.0
        num_batches = 0

        # For DeepONet: cycle sequence loader (fewer samples) to match per-timestep loader length
        if self.is_deeponet and self.per_timestep_train_loader is not None:
            # DeepONet with dual-batch training
            # Per-timestep loader has ~320K samples, sequence has ~100 samples
            # Cycle sequence loader to match the iteration count of per-timestep
            sequence_iter = cycle(self.sequence_train_loader)
            per_timestep_iter = iter(self.per_timestep_train_loader)

            for batch_idx, per_timestep_batch in enumerate(per_timestep_iter):
                # Get corresponding sequence batch (cycled)
                sequence_batch = next(sequence_iter)

                # ===== Per-timestep forward (MSE loss) =====
                per_ts_inputs = per_timestep_batch['input'].to(self.device)      # [B, 4000]
                per_ts_targets = per_timestep_batch['target'].to(self.device)     # [B]
                per_ts_time_coords = per_timestep_batch['time_coord'].to(self.device)  # [B]
                per_ts_penalties = per_timestep_batch['penalty'].to(self.device)  # [B]

                # Forward per-timestep
                per_ts_outputs = self.model.forward_per_timestep(per_ts_inputs, per_ts_time_coords)
                per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                # Compute MSE loss
                per_ts_loss = self.criterion(per_ts_outputs, per_ts_targets)
                if self.use_penalty_weighting:
                    mse_loss = (per_ts_loss * per_ts_penalties).mean()
                else:
                    mse_loss = per_ts_loss.mean() if per_ts_loss.ndim > 0 else per_ts_loss

                # ===== Sequence forward (BSP loss) =====
                seq_inputs = sequence_batch[0].to(self.device)    # [B, 1, 4000]
                seq_targets = sequence_batch[1].to(self.device)   # [B, 1, 4000]

                # Forward sequence
                seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]

                # Compute BSP loss
                bsp_loss = self.criterion(seq_outputs, seq_targets)
                bsp_loss = bsp_loss.mean() if bsp_loss.ndim > 0 else bsp_loss

                # ===== Combine losses =====
                combined_loss = self.lambda_mse * mse_loss + self.lambda_bsp * bsp_loss

                # ===== Backward pass =====
                self.optimizer.zero_grad()
                if self.weight_optimizer is not None:
                    self.weight_optimizer.zero_grad()

                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self._update_adaptive_weights()

                # Step scheduler (for cosine annealing, step every batch)
                if self.config.scheduler_type == 'cosine':
                    self.scheduler.step()

                # Accumulate
                total_loss += combined_loss.item()
                total_mse_loss += mse_loss.item()
                total_bsp_loss += bsp_loss.item()
                num_batches += 1

        else:
            # FNO/UNet or DeepONet sequence-only training
            for batch_idx, batch in enumerate(self.sequence_train_loader):
                seq_inputs = batch[0].to(self.device)     # [B, 1, 4000]
                seq_targets = batch[1].to(self.device)    # [B, 1, 4000]

                # Forward pass
                self.optimizer.zero_grad()
                if self.weight_optimizer is not None:
                    self.weight_optimizer.zero_grad()

                # Use appropriate forward method based on model type
                if self.is_deeponet:
                    seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]
                else:
                    seq_outputs = self.model(seq_inputs)  # [B, 1, 4000]

                # Fix: Ensure outputs and targets are 3D [B, C, T] for loss computation
                # Some models may return 4D tensors, squeeze extra dims
                while seq_outputs.dim() > 3:
                    for dim_idx in range(seq_outputs.dim()):
                        if seq_outputs.shape[dim_idx] == 1:
                            seq_outputs = seq_outputs.squeeze(dim_idx)
                            break
                while seq_targets.dim() > 3:
                    for dim_idx in range(seq_targets.dim()):
                        if seq_targets.shape[dim_idx] == 1:
                            seq_targets = seq_targets.squeeze(dim_idx)
                            break

                # Compute loss (MSE on sequences)
                loss = self.criterion(seq_outputs, seq_targets)
                final_loss = loss.mean() if loss.ndim > 0 else loss

                # Backward pass
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self._update_adaptive_weights()

                # Step scheduler
                if self.config.scheduler_type == 'cosine':
                    self.scheduler.step()

                # Accumulate
                total_loss += final_loss.item()
                num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {'loss': avg_loss}

        if self.is_deeponet:
            metrics['mse_loss'] = total_mse_loss / num_batches if num_batches > 0 else 0.0
            metrics['bsp_loss'] = total_bsp_loss / num_batches if num_batches > 0 else 0.0

        # Add SA-BSP weight monitoring
        if self.adapt_mode and self.adapt_mode != 'none':
            weight_stats = self._get_adaptive_weight_stats()
            metrics.update(weight_stats)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set with dual-batch validation.

        For DeepONet:
        - Uses both per-timestep and sequence validation loaders
        - Computes loss_mse from per-timestep predictions
        - Computes loss_bsp from full-sequence predictions
        - Combines: loss = lambda_mse * loss_mse + lambda_bsp * loss_bsp

        For FNO/UNet:
        - Uses only sequence validation loader
        - Computes loss from full-sequence predictions

        Returns:
            Dictionary with validation metrics:
            - 'loss': Average validation loss (combined MSE + BSP)
            - 'mse_loss': Average MSE loss (DeepONet only)
            - 'bsp_loss': Average BSP loss (DeepONet only)
        """
        self.model.eval()

        total_loss = 0.0
        total_mse_loss = 0.0
        total_bsp_loss = 0.0
        num_batches = 0

        if self.is_deeponet and self.per_timestep_val_loader is not None:
            # DeepONet with dual-batch validation
            # Cycle sequence loader to match per-timestep loader length
            sequence_iter = cycle(self.sequence_val_loader)
            per_timestep_iter = iter(self.per_timestep_val_loader)

            for batch_idx, per_timestep_batch in enumerate(per_timestep_iter):
                # Get corresponding sequence batch (cycled)
                sequence_batch = next(sequence_iter)

                # ===== Per-timestep validation (MSE loss) =====
                per_ts_inputs = per_timestep_batch['input'].to(self.device)      # [B, 4000]
                per_ts_targets = per_timestep_batch['target'].to(self.device)     # [B]
                per_ts_time_coords = per_timestep_batch['time_coord'].to(self.device)  # [B]

                # Forward per-timestep (no penalty weighting during validation)
                per_ts_outputs = self.model.forward_per_timestep(per_ts_inputs, per_ts_time_coords)
                per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                # Compute MSE loss
                per_ts_loss = self.criterion(per_ts_outputs, per_ts_targets)
                mse_loss = per_ts_loss.mean() if per_ts_loss.ndim > 0 else per_ts_loss

                # ===== Sequence validation (BSP loss) =====
                seq_inputs = sequence_batch[0].to(self.device)    # [B, 1, 4000]
                seq_targets = sequence_batch[1].to(self.device)   # [B, 1, 4000]

                # Forward sequence
                seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]

                # Compute BSP loss
                bsp_loss = self.criterion(seq_outputs, seq_targets)
                bsp_loss = bsp_loss.mean() if bsp_loss.ndim > 0 else bsp_loss

                # ===== Combine losses =====
                combined_loss = self.lambda_mse * mse_loss + self.lambda_bsp * bsp_loss

                # Accumulate
                total_loss += combined_loss.item()
                total_mse_loss += mse_loss.item()
                total_bsp_loss += bsp_loss.item()
                num_batches += 1

        else:
            # FNO/UNet or DeepONet with sequence-only validation
            for batch in self.sequence_val_loader:
                seq_inputs = batch[0].to(self.device)     # [B, 1, 4000]
                seq_targets = batch[1].to(self.device)    # [B, 1, 4000]

                # Forward pass (use appropriate method based on model type)
                if self.is_deeponet:
                    seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]
                else:
                    seq_outputs = self.model(seq_inputs)  # [B, 1, 4000]

                # Fix: Ensure outputs and targets are 3D [B, C, T] for loss computation
                while seq_outputs.dim() > 3:
                    for dim_idx in range(seq_outputs.dim()):
                        if seq_outputs.shape[dim_idx] == 1:
                            seq_outputs = seq_outputs.squeeze(dim_idx)
                            break
                while seq_targets.dim() > 3:
                    for dim_idx in range(seq_targets.dim()):
                        if seq_targets.shape[dim_idx] == 1:
                            seq_targets = seq_targets.squeeze(dim_idx)
                            break

                # Compute loss
                loss = self.criterion(seq_outputs, seq_targets)
                final_loss = loss.mean() if loss.ndim > 0 else loss

                # Accumulate
                total_loss += final_loss.item()
                num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {'loss': avg_loss}

        if self.is_deeponet:
            metrics['mse_loss'] = total_mse_loss / num_batches if num_batches > 0 else 0.0
            metrics['bsp_loss'] = total_bsp_loss / num_batches if num_batches > 0 else 0.0

        # Add SA-BSP weight monitoring
        if self.adapt_mode and self.adapt_mode != 'none':
            weight_stats = self._get_adaptive_weight_stats()
            metrics.update(weight_stats)

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
            'weight_optimizer_state_dict': (
                self.weight_optimizer.state_dict() if self.weight_optimizer else None
            ),
            'weight_scheduler_state_dict': (
                self.weight_scheduler.state_dict() if self.weight_scheduler else None
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

        # Restore weight optimizer state (for SA-BSP)
        if self.weight_optimizer and checkpoint.get('weight_optimizer_state_dict'):
            self.weight_optimizer.load_state_dict(checkpoint['weight_optimizer_state_dict'])

        # Restore weight scheduler state (for SA-BSP)
        if self.weight_scheduler and checkpoint.get('weight_scheduler_state_dict'):
            self.weight_scheduler.load_state_dict(checkpoint['weight_scheduler_state_dict'])

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
            self.console.print(f"Model: {'DeepONet' if self.is_deeponet else 'FNO/UNet'}")
            self.console.print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            if self.is_deeponet and self.per_timestep_train_loader is not None:
                # DeepONet with dual-batch training
                self.console.print(f"Training samples (per-timestep): {len(self.per_timestep_train_loader.dataset):,}")
                self.console.print(f"Training samples (sequences): {len(self.sequence_train_loader.dataset):,}")
                self.console.print(f"Validation samples (per-timestep): {len(self.per_timestep_val_loader.dataset):,}")
                self.console.print(f"Validation samples (sequences): {len(self.sequence_val_loader.dataset):,}")
            else:
                # FNO/UNet or DeepONet with sequence-only training
                self.console.print(f"Training samples (sequences): {len(self.sequence_train_loader.dataset):,}")
                self.console.print(f"Validation samples (sequences): {len(self.sequence_val_loader.dataset):,}")

            self.console.print(f"Loss weights: λ_mse={self.lambda_mse}, λ_bsp={self.lambda_bsp}\n")

        # Determine progress bar format based on model type
        if self.is_deeponet:
            progress_columns = [
                TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.total}"),
                BarColumn(),
                TextColumn("•"),
                TextColumn("Loss: {task.fields[train_loss]:.4f}"),
                TextColumn("[dim](MSE: {task.fields[train_mse]:.4f} BSP: {task.fields[train_bsp]:.4f})[/dim]"),
                TextColumn("•"),
                TextColumn("Val: {task.fields[val_loss]:.4f}"),
                TextColumn("•"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn()
            ]
            initial_task_fields = {
                'epoch': 0, 'train_loss': 0.0, 'train_mse': 0.0, 'train_bsp': 0.0,
                'val_loss': 0.0, 'lr': self.config.learning_rate
            }
        else:
            progress_columns = [
                TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.total}"),
                BarColumn(),
                TextColumn("•"),
                TextColumn("Train Loss: {task.fields[train_loss]:.4f}"),
                TextColumn("•"),
                TextColumn("Val Loss: {task.fields[val_loss]:.4f}"),
                TextColumn("•"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn()
            ]
            initial_task_fields = {
                'epoch': 0, 'train_loss': 0.0,
                'val_loss': 0.0, 'lr': self.config.learning_rate
            }

        # Main epoch loop with progress bar
        with Progress(*progress_columns) as progress:

            epoch_task = progress.add_task("Training", total=self.config.num_epochs, **initial_task_fields)

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
                update_fields = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'lr': current_lr
                }

                if self.is_deeponet:
                    update_fields['train_mse'] = train_metrics.get('mse_loss', 0.0)
                    update_fields['train_bsp'] = train_metrics.get('bsp_loss', 0.0)

                progress.update(epoch_task, advance=1, **update_fields)

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
