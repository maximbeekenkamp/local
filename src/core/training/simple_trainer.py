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
from torch.amp import autocast, GradScaler
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
    compute_mse,
    compute_spectrum_error_1d
)
from ..evaluation.loss_factory import create_loss, CombinedLoss
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
                Format: Dict with 'input', 'target', 'time_coord'
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
        self.criterion.to(self.device)

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

        # Mixed Precision Training (AMP) for memory reduction and stability
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.amp_device = 'cuda' if self.use_amp else 'cpu'
        if self.use_amp:
            print(f"  ✓ Automatic Mixed Precision (AMP) enabled for ~50% memory reduction")

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

    def _check_outputs_for_instability(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        epoch: int,
        batch_idx: int
    ) -> bool:
        """
        Check outputs for Inf/NaN before loss computation (early detection).

        Args:
            outputs: Model outputs to check
            inputs: Model inputs for diagnostic info
            epoch: Current epoch number
            batch_idx: Current batch index

        Returns:
            True if instability detected, False otherwise
        """
        # Check for Inf (which becomes NaN after loss computation)
        if torch.isinf(outputs).any():
            self.console.print(f"\n[bold red]❌ Inf detected in model outputs![/bold red]")
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(f"  Output range: [{outputs.min():.6e}, {outputs.max():.6e}]")
            self.console.print(f"  Input range: [{inputs.min():.6e}, {inputs.max():.6e}]")

            # Check if model parameters are corrupted
            corrupt_params = []
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    corrupt_params.append(name)

            if corrupt_params:
                self.console.print(f"  [red]Corrupted parameters:[/red] {corrupt_params[:5]}")

            self.console.print(f"\n[yellow]Likely causes:[/yellow]")
            self.console.print(f"  1. Learning rate too high (current: {self.optimizer.param_groups[0]['lr']:.2e})")
            self.console.print(f"  2. Model weights exploded from previous batches")
            self.console.print(f"  3. Activation function instability (try 'tanh' instead of 'requ')")

            return True

        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            self.console.print(f"\n[bold red]❌ NaN detected in model outputs![/bold red]")
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(f"  This usually means model parameters were already corrupted")
            return True

        return False

    def _check_model_parameters(self, epoch: int, batch_idx: int) -> bool:
        """
        Check model parameters for NaN/Inf corruption.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index

        Returns:
            True if corruption detected, False otherwise
        """
        corrupt_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                corrupt_params.append(name)

        if corrupt_params:
            self.console.print(f"\n[bold red]❌ Model parameters corrupted![/bold red]")
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(f"  Corrupted parameters: {corrupt_params[:10]}")
            self.console.print(f"\n[yellow]Recovery not possible - model weights are corrupted[/yellow]")
            self.console.print(f"  Reduce learning rate and restart training from checkpoint")
            return True

        return False

    def _check_for_nan(self, loss: torch.Tensor, loss_name: str, epoch: int, batch_idx: int) -> bool:
        """
        Check if loss contains NaN and print diagnostic information.

        Args:
            loss: Loss tensor to check
            loss_name: Name of the loss for reporting (e.g., 'mse_loss', 'bsp_loss')
            epoch: Current epoch number
            batch_idx: Current batch index

        Returns:
            True if NaN detected, False otherwise
        """
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.console.print(f"\n[bold red]❌ NaN/Inf detected in {loss_name}![/bold red]")
            self.console.print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            self.console.print(f"  Loss value: {loss.item() if loss.numel() == 1 else loss}")
            self.console.print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Check model parameters for NaN/Inf
            nan_params = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_params.append(name)

            if nan_params:
                self.console.print(f"  Parameters with NaN/Inf gradients: {nan_params[:5]}...")

            # Check gradient norms
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.console.print(f"  Total gradient norm: {total_norm:.2e}")

            self.console.print(f"\n[yellow]Diagnostic Tips:[/yellow]")
            self.console.print(f"  1. Reduce learning rate (current: {self.optimizer.param_groups[0]['lr']:.2e})")
            self.console.print(f"     Suggested: Try 1e-4 or 3e-4 for SOAP optimizer")
            self.console.print(f"  2. Gradient clipping: {'ENABLED' if self.config.max_grad_norm > 0 else 'DISABLED'} (max_norm={self.config.max_grad_norm})")
            self.console.print(f"  3. Check input data for NaN/Inf values")
            self.console.print(f"  4. Increase epsilon in loss config if using BSP")
            self.console.print(f"  5. Try Adam optimizer instead of SOAP (optimizer_type='adam' in config)")

            return True
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

        # Import loss classes for isinstance checks
        from ..evaluation.loss_factory import CombinedLoss
        from ..evaluation.binned_spectral_loss import BinnedSpectralLoss
        from ..evaluation.adaptive_spectral_loss import SelfAdaptiveBSPLoss

        total_loss = 0.0
        total_mse_loss = 0.0
        total_bsp_loss = 0.0
        num_batches = 0

        # For DeepONet: use appropriate training mode based on loss type
        # - Combined losses (MSE + BSP): Use dual-batch mode (per-timestep MSE + sequence BSP)
        # - Baseline MSE: Use per-timestep-only mode (320K samples)
        # - Sequence-based losses (BSP only): Use sequence-only mode
        use_dual_batch = (self.is_deeponet and
                         self.per_timestep_train_loader is not None and
                         isinstance(self.criterion, CombinedLoss))

        use_per_timestep_only = (self.is_deeponet and
                                self.per_timestep_train_loader is not None and
                                not isinstance(self.criterion, (CombinedLoss, BinnedSpectralLoss, SelfAdaptiveBSPLoss)))

        if use_dual_batch:
            # DeepONet with dual-batch training (combined loss only)
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

                # Forward per-timestep with AMP
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    per_ts_outputs = self.model.forward_per_timestep(per_ts_inputs, per_ts_time_coords)
                    per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                    # EARLY CHECK: Detect Inf/NaN in outputs
                    if self._check_outputs_for_instability(
                        per_ts_outputs, per_ts_inputs, self.current_epoch, batch_idx
                    ):
                        raise RuntimeError(
                            f"Model instability in per-timestep forward at epoch {self.current_epoch}, batch {batch_idx}"
                        )

                    # Compute MSE loss (no penalty weighting - removed for consistency)
                    per_ts_loss = self.criterion(per_ts_outputs, per_ts_targets)
                    mse_loss = per_ts_loss.mean() if per_ts_loss.ndim > 0 else per_ts_loss

                # Check for NaN in MSE loss
                if self._check_for_nan(mse_loss, 'mse_loss', self.current_epoch, batch_idx):
                    raise RuntimeError(f"NaN detected in MSE loss at epoch {self.current_epoch}, batch {batch_idx}")

                # ===== Sequence forward (BSP loss) =====
                # Extract inputs, targets, and sample indices
                if len(sequence_batch) == 3:  # New format with indices
                    seq_inputs, seq_targets, sample_indices = sequence_batch
                    seq_inputs = seq_inputs.to(self.device)      # [B, 1, 4000]
                    seq_targets = seq_targets.to(self.device)    # [B, 1, 4000]
                    sample_indices = sample_indices.to(self.device)  # [B]
                else:  # Fallback for old format
                    seq_inputs = sequence_batch[0].to(self.device)
                    seq_targets = sequence_batch[1].to(self.device)
                    sample_indices = None

                # Forward sequence with AMP
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]

                    # Compute BSP loss (pass sample indices for cache lookup)
                    bsp_loss = self.criterion(seq_outputs, seq_targets, sample_indices=sample_indices)
                    bsp_loss = bsp_loss.mean() if bsp_loss.ndim > 0 else bsp_loss

                # Check for NaN in BSP loss
                if self._check_for_nan(bsp_loss, 'bsp_loss', self.current_epoch, batch_idx):
                    raise RuntimeError(f"NaN detected in BSP loss at epoch {self.current_epoch}, batch {batch_idx}")

                # ===== Sequential Backward Passes (Memory Optimization) =====
                # Instead of combined_loss = λ_mse * mse_loss + λ_bsp * bsp_loss
                # We do two separate backward passes to avoid keeping both graphs in memory
                # This reduces peak memory by ~50% at the cost of 2x optimizer steps

                # First backward: MSE loss
                self.optimizer.zero_grad()
                if self.weight_optimizer is not None:
                    self.weight_optimizer.zero_grad()

                weighted_mse_loss = self.lambda_mse * mse_loss
                if self.use_amp:
                    self.scaler.scale(weighted_mse_loss).backward()
                    # Gradient clipping (unscale first for AMP)
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    weighted_mse_loss.backward()
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                # Note: Don't update adaptive weights yet - wait for BSP backward

                # Second backward: BSP loss
                self.optimizer.zero_grad()
                if self.weight_optimizer is not None:
                    self.weight_optimizer.zero_grad()

                weighted_bsp_loss = self.lambda_bsp * bsp_loss
                if self.use_amp:
                    self.scaler.scale(weighted_bsp_loss).backward()
                    # Gradient clipping (unscale first for AMP)
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    weighted_bsp_loss.backward()
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self._update_adaptive_weights()  # Update adaptive weights after BSP backward

                # Step scheduler (for cosine annealing, step every batch)
                # NOTE: Scheduler steps ONCE per iteration despite 2 optimizer steps
                # This maintains the intended learning rate schedule
                if self.config.scheduler_type == 'cosine':
                    self.scheduler.step()

                # Accumulate (compute combined loss for logging only)
                combined_loss_value = weighted_mse_loss.item() + weighted_bsp_loss.item()
                total_loss += combined_loss_value
                total_mse_loss += mse_loss.item()
                total_bsp_loss += bsp_loss.item()
                num_batches += 1

                # PERIODIC CHECK: Check for parameter corruption every 10 batches
                if batch_idx % 10 == 0 and batch_idx > 0:
                    if self._check_model_parameters(self.current_epoch, batch_idx):
                        raise RuntimeError(
                            f"Model parameters corrupted at epoch {self.current_epoch}, batch {batch_idx}"
                        )

        elif use_per_timestep_only:
            # DeepONet with baseline MSE (per-timestep only, no BSP)
            # Uses per-timestep loader with 320K samples for proper MSE training
            print(f"  Using per-timestep-only mode for DeepONet baseline MSE")

            for batch_idx, per_timestep_batch in enumerate(self.per_timestep_train_loader):
                # Extract per-timestep data
                per_ts_inputs = per_timestep_batch['input'].to(self.device)      # [B, 4000]
                per_ts_targets = per_timestep_batch['target'].to(self.device)     # [B]
                per_ts_time_coords = per_timestep_batch['time_coord'].to(self.device)  # [B]

                # Forward pass
                self.optimizer.zero_grad()

                # Forward with AMP (epsilon=1e-6 in configs prevents numerical issues)
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    per_ts_outputs = self.model.forward_per_timestep(per_ts_inputs, per_ts_time_coords)
                    per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                    # EARLY CHECK: Detect Inf/NaN in outputs BEFORE loss computation
                    if self._check_outputs_for_instability(
                        per_ts_outputs, per_ts_inputs, self.current_epoch, batch_idx
                    ):
                        raise RuntimeError(
                            f"Model instability detected at epoch {self.current_epoch}, batch {batch_idx}. "
                            f"Outputs contain Inf/NaN. See diagnostics above."
                        )

                    # Compute MSE loss
                    loss = self.criterion(per_ts_outputs, per_ts_targets)
                    final_loss = loss.mean() if loss.ndim > 0 else loss

                # Check for NaN in loss (should be caught by output check above)
                if self._check_for_nan(final_loss, 'per_timestep_mse_loss', self.current_epoch, batch_idx):
                    raise RuntimeError(f"NaN detected in per-timestep MSE loss at epoch {self.current_epoch}, batch {batch_idx}")

                # Backward pass with gradient clipping and AMP
                if self.use_amp:
                    self.scaler.scale(final_loss).backward()
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    final_loss.backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                # Step scheduler (batch-wise for cosine annealing)
                if self.config.scheduler_type == 'cosine':
                    self.scheduler.step()

                # Accumulate
                total_loss += final_loss.item()
                total_mse_loss += final_loss.item()
                total_bsp_loss += 0.0  # No BSP component
                num_batches += 1

                # PERIODIC CHECK: Check for parameter corruption every 10 batches
                if batch_idx % 10 == 0 and batch_idx > 0:
                    if self._check_model_parameters(self.current_epoch, batch_idx):
                        raise RuntimeError(
                            f"Model parameters corrupted at epoch {self.current_epoch}, batch {batch_idx}. "
                            f"Reduce learning rate and restart from checkpoint."
                        )

        else:
            # FNO/UNet sequence-only training OR DeepONet with sequence-based losses (BSP only)
            for batch_idx, batch in enumerate(self.sequence_train_loader):
                # Extract inputs, targets, and sample indices (for cache lookup)
                if len(batch) == 3:  # New format with indices
                    seq_inputs, seq_targets, sample_indices = batch
                    seq_inputs = seq_inputs.to(self.device)     # [B, 1, 4000]
                    seq_targets = seq_targets.to(self.device)    # [B, 1, 4000]
                    sample_indices = sample_indices.to(self.device)  # [B]
                else:  # Fallback for old format
                    seq_inputs = batch[0].to(self.device)
                    seq_targets = batch[1].to(self.device)
                    sample_indices = None

                # Forward pass with AMP
                self.optimizer.zero_grad()
                if self.weight_optimizer is not None:
                    self.weight_optimizer.zero_grad()

                # Use appropriate forward method based on model type
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    if self.is_deeponet:
                        seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]
                    else:
                        seq_outputs = self.model(seq_inputs)  # [B, 1, 4000]

                    # Compute loss (pass sample indices for cache lookup if supported)
                    # For baseline MSE loss, don't pass sample_indices (PyTorch MSELoss doesn't accept it)
                    if isinstance(self.criterion, (CombinedLoss, BinnedSpectralLoss, SelfAdaptiveBSPLoss)):
                        loss = self.criterion(seq_outputs, seq_targets, sample_indices=sample_indices)
                    else:
                        # Baseline MSE or other simple losses that don't accept sample_indices
                        loss = self.criterion(seq_outputs, seq_targets)
                    final_loss = loss.mean() if loss.ndim > 0 else loss

                # Check for NaN in sequence loss
                if self._check_for_nan(final_loss, 'sequence_loss', self.current_epoch, batch_idx):
                    raise RuntimeError(f"NaN detected in sequence loss at epoch {self.current_epoch}, batch {batch_idx}")

                # Extract loss components if using CombinedLoss
                if isinstance(self.criterion, CombinedLoss):
                    components = self.criterion.get_loss_components(seq_outputs, seq_targets, sample_indices=sample_indices)
                    mse_component = components['base'].mean() if components['base'].ndim > 0 else components['base']
                    bsp_component = components['spectral'].mean() if components['spectral'].ndim > 0 else components['spectral']
                else:
                    # For non-combined loss, just use the loss itself for MSE
                    mse_component = final_loss
                    bsp_component = torch.tensor(0.0)

                # Backward pass with AMP
                if self.use_amp:
                    self.scaler.scale(final_loss).backward()
                    # Gradient clipping (unscale first for AMP)
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    final_loss.backward()
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self._update_adaptive_weights()

                # Step scheduler
                if self.config.scheduler_type == 'cosine':
                    self.scheduler.step()

                # Accumulate
                total_loss += final_loss.item()
                total_mse_loss += mse_component.item()
                total_bsp_loss += bsp_component.item()
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

        # Import loss classes for isinstance checks
        from ..evaluation.loss_factory import CombinedLoss
        from ..evaluation.binned_spectral_loss import BinnedSpectralLoss
        from ..evaluation.adaptive_spectral_loss import SelfAdaptiveBSPLoss

        total_loss = 0.0
        total_mse_loss = 0.0
        total_bsp_loss = 0.0
        num_batches = 0

        # Collect predictions and targets for eval metrics
        all_predictions = []
        all_targets = []

        # Determine validation mode (same logic as training)
        use_dual_batch = (self.is_deeponet and
                         self.per_timestep_val_loader is not None and
                         isinstance(self.criterion, CombinedLoss))

        use_per_timestep_only = (self.is_deeponet and
                                self.per_timestep_val_loader is not None and
                                not isinstance(self.criterion, (CombinedLoss, BinnedSpectralLoss, SelfAdaptiveBSPLoss)))

        if use_dual_batch:
            # DeepONet with dual-batch validation (combined loss only)
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

                # Forward per-timestep with AMP
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    per_ts_outputs = self.model.forward_per_timestep(per_ts_inputs, per_ts_time_coords)
                    per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                    # Compute MSE loss
                    per_ts_loss = self.criterion(per_ts_outputs, per_ts_targets)
                    mse_loss = per_ts_loss.mean() if per_ts_loss.ndim > 0 else per_ts_loss

                # ===== Sequence validation (BSP loss) =====
                # Extract inputs, targets, and sample indices
                if len(sequence_batch) == 3:  # New format with indices
                    seq_inputs, seq_targets, sample_indices = sequence_batch
                    seq_inputs = seq_inputs.to(self.device)      # [B, 1, 4000]
                    seq_targets = seq_targets.to(self.device)    # [B, 1, 4000]
                    sample_indices = sample_indices.to(self.device)  # [B]
                else:  # Fallback for old format
                    seq_inputs = sequence_batch[0].to(self.device)
                    seq_targets = sequence_batch[1].to(self.device)
                    sample_indices = None

                # Forward sequence with AMP
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]

                    # Compute BSP loss (pass sample indices for cache lookup)
                    bsp_loss = self.criterion(seq_outputs, seq_targets, sample_indices=sample_indices)
                    bsp_loss = bsp_loss.mean() if bsp_loss.ndim > 0 else bsp_loss

                # ===== Combine losses (for logging only) =====
                combined_loss_value = self.lambda_mse * mse_loss.item() + self.lambda_bsp * bsp_loss.item()

                # Accumulate
                total_loss += combined_loss_value
                total_mse_loss += mse_loss.item()
                total_bsp_loss += bsp_loss.item()
                num_batches += 1

                # Collect for eval metrics
                all_predictions.append(seq_outputs.detach().cpu())
                all_targets.append(seq_targets.detach().cpu())

        elif use_per_timestep_only:
            # DeepONet with baseline MSE validation (per-timestep only)
            for per_timestep_batch in self.per_timestep_val_loader:
                # Extract per-timestep data
                per_ts_inputs = per_timestep_batch['input'].to(self.device)      # [B, 4000]
                per_ts_targets = per_timestep_batch['target'].to(self.device)     # [B]
                per_ts_time_coords = per_timestep_batch['time_coord'].to(self.device)  # [B]

                # Forward pass with AMP
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    per_ts_outputs = self.model.forward_per_timestep(per_ts_inputs, per_ts_time_coords)
                    per_ts_outputs = per_ts_outputs.squeeze(-1)  # [B, 1] → [B]

                    # Compute MSE loss
                    loss = self.criterion(per_ts_outputs, per_ts_targets)
                    final_loss = loss.mean() if loss.ndim > 0 else loss

                # Accumulate
                total_loss += final_loss.item()
                total_mse_loss += final_loss.item()
                total_bsp_loss += 0.0  # No BSP component
                num_batches += 1

                # Note: Can't collect full sequences for eval metrics in per-timestep mode
                # Eval metrics will be skipped for this mode

        else:
            # FNO/UNet sequence-only validation OR DeepONet with sequence-based losses
            for batch in self.sequence_val_loader:
                # Extract inputs, targets, and sample indices
                if len(batch) == 3:  # New format with indices
                    seq_inputs, seq_targets, sample_indices = batch
                    seq_inputs = seq_inputs.to(self.device)     # [B, 1, 4000]
                    seq_targets = seq_targets.to(self.device)    # [B, 1, 4000]
                    sample_indices = sample_indices.to(self.device)  # [B]
                else:  # Fallback for old format
                    seq_inputs = batch[0].to(self.device)
                    seq_targets = batch[1].to(self.device)
                    sample_indices = None

                # Forward pass with AMP (use appropriate method based on model type)
                with autocast(device_type=self.amp_device, enabled=self.use_amp):
                    if self.is_deeponet:
                        seq_outputs = self.model.forward_sequence(seq_inputs)  # [B, 1, 4000]
                    else:
                        seq_outputs = self.model(seq_inputs)  # [B, 1, 4000]

                    # Compute loss (pass sample indices for cache lookup if supported)
                    # For baseline MSE loss, don't pass sample_indices (PyTorch MSELoss doesn't accept it)
                    if isinstance(self.criterion, (CombinedLoss, BinnedSpectralLoss, SelfAdaptiveBSPLoss)):
                        loss = self.criterion(seq_outputs, seq_targets, sample_indices=sample_indices)
                    else:
                        # Baseline MSE or other simple losses that don't accept sample_indices
                        loss = self.criterion(seq_outputs, seq_targets)
                    final_loss = loss.mean() if loss.ndim > 0 else loss

                # Extract loss components if using CombinedLoss
                if isinstance(self.criterion, CombinedLoss):
                    components = self.criterion.get_loss_components(seq_outputs, seq_targets, sample_indices=sample_indices)
                    mse_component = components['base'].mean() if components['base'].ndim > 0 else components['base']
                    bsp_component = components['spectral'].mean() if components['spectral'].ndim > 0 else components['spectral']
                else:
                    # For non-combined loss, just use the loss itself for MSE
                    mse_component = final_loss
                    bsp_component = torch.tensor(0.0)

                # Accumulate
                total_loss += final_loss.item()
                total_mse_loss += mse_component.item()
                total_bsp_loss += bsp_component.item()
                num_batches += 1

                # Collect for eval metrics
                all_predictions.append(seq_outputs.detach().cpu())
                all_targets.append(seq_targets.detach().cpu())

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {'loss': avg_loss}

        if self.is_deeponet:
            metrics['mse_loss'] = total_mse_loss / num_batches if num_batches > 0 else 0.0
            metrics['bsp_loss'] = total_bsp_loss / num_batches if num_batches > 0 else 0.0

        # Compute additional evaluation metrics if requested
        if len(all_predictions) > 0 and self.config.eval_metrics:
            all_preds_tensor = torch.cat(all_predictions, dim=0)  # [N, C, T]
            all_targets_tensor = torch.cat(all_targets, dim=0)    # [N, C, T]

            if 'mse' in self.config.eval_metrics:
                mse = compute_mse(all_preds_tensor, all_targets_tensor)
                metrics['mse'] = mse

            if 'spectrum_error' in self.config.eval_metrics:
                spectrum_error = compute_spectrum_error_1d(all_preds_tensor, all_targets_tensor)
                metrics['spectrum_error'] = spectrum_error

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

        # Validate loss config matches (enforce config consistency)
        if 'loss_config' in checkpoint:
            ckpt_loss_config = checkpoint['loss_config']
            current_loss_config = {
                'loss_type': self.config.loss_config.loss_type,
                'loss_params': self.config.loss_config.loss_params
            }

            # Check loss type match
            if ckpt_loss_config['loss_type'] != current_loss_config['loss_type']:
                raise ValueError(
                    f"Loss config mismatch!\n"
                    f"  Checkpoint loss_type: {ckpt_loss_config['loss_type']}\n"
                    f"  Current loss_type: {current_loss_config['loss_type']}\n"
                    f"Cannot load checkpoint with different loss configuration.\n"
                    f"Retrain from scratch or use matching loss config."
                )

            # Check critical params for spectral losses
            if current_loss_config['loss_type'] in ['bsp', 'sa_bsp', 'combined']:
                critical_params = ['n_bins', 'use_log', 'use_minmax_norm', 'loss_type', 'adapt_mode']
                for param in critical_params:
                    ckpt_val = ckpt_loss_config.get('loss_params', {}).get(param)
                    current_val = current_loss_config['loss_params'].get(param)
                    if ckpt_val != current_val and ckpt_val is not None and current_val is not None:
                        raise ValueError(
                            f"Loss param '{param}' mismatch!\n"
                            f"  Checkpoint: {ckpt_val}\n"
                            f"  Current: {current_val}\n"
                            f"Cannot load checkpoint with different {param}.\n"
                            f"Retrain from scratch or use matching config."
                        )

        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore weight optimizer state (for SA-BSP)
        if self.weight_optimizer:
            if 'weight_optimizer_state_dict' not in checkpoint:
                raise ValueError(
                    "Checkpoint missing 'weight_optimizer_state_dict' but SA-BSP is active.\n"
                    "Cannot load checkpoint from non-SA-BSP training into SA-BSP model.\n"
                    "Retrain from scratch or use matching loss configuration."
                )
            self.weight_optimizer.load_state_dict(checkpoint['weight_optimizer_state_dict'])

        # Restore weight scheduler state (for SA-BSP)
        if self.weight_scheduler:
            if 'weight_scheduler_state_dict' not in checkpoint:
                raise ValueError(
                    "Checkpoint missing 'weight_scheduler_state_dict' but SA-BSP is active.\n"
                    "Cannot load checkpoint from incompatible training configuration.\n"
                    "Retrain from scratch or use matching loss configuration."
                )
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
