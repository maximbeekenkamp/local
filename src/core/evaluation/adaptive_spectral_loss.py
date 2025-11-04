"""
Self-Adaptive Binned Spectral Power (SA-BSP) Loss for 1D temporal neural operators.

Extends BSP loss with learnable per-bin weights inspired by SA-PINNs (Self-Adaptive
Physics-Informed Neural Networks). Weights adapt during training to balance errors
across frequency bands.

Key innovation: Weights are learned via saddle-point optimization
    min_θ max_λ: L(θ, λ) = Σ_i λ_i × error_i(θ)

Where:
- θ: Model parameters
- λ: Adaptive weights (one per frequency bin)
- error_i: Per-bin spectral error

This encourages the model to reduce errors in all frequency bands, not just
low frequencies (mitigates spectral bias more effectively than fixed BSP).

Reference:
- SA-PINNs: "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks"
- Generatively-Stabilised-NOs adaptive_spectral_loss.py (2D implementation)

Training requirements:
    1. Main optimizer for model parameters (e.g., Adam on model.parameters())
    2. Separate optimizer for adaptive weights (e.g., Adam on loss.adaptive_weights.parameters())
    3. Update both optimizers each iteration

Example:
    >>> # Create loss
    >>> loss_fn = SelfAdaptiveBSPLoss(n_bins=32, adapt_mode='per-bin')
    >>>
    >>> # Create optimizers
    >>> model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> weight_optimizer = torch.optim.Adam(loss_fn.adaptive_weights.parameters(), lr=1e-3)
    >>>
    >>> # Training loop
    >>> for batch in dataloader:
    ...     # Forward pass
    ...     pred = model(input)
    ...     loss = loss_fn(pred, target)
    ...
    ...     # Backward pass
    ...     model_optimizer.zero_grad()
    ...     weight_optimizer.zero_grad()
    ...     loss.backward()
    ...     model_optimizer.step()
    ...     weight_optimizer.step()  # Update adaptive weights
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .binned_spectral_loss import BinnedSpectralLoss


class SelfAdaptiveWeights(nn.Module):
    """
    Trainable adaptive weights for frequency bins.

    Weights are stored in log-space (log λ) to ensure positivity after
    exponentiation (λ = exp(log λ) > 0). This is the standard SA-PINNs approach.

    Attributes:
        n_components: Number of weight components (e.g., n_bins for per-bin mode)
        mode: Weight adaptation mode ('per-bin', 'global', 'both', 'none')
        log_weights: Trainable log-weights (parameters)
    """

    def __init__(
        self,
        n_components: int,
        mode: str = 'per-bin',
        init_value: float = 1.0
    ):
        """
        Initialize adaptive weights.

        Args:
            n_components: Number of weight components
                - For 'per-bin': n_components = n_bins
                - For 'global': n_components = 1
                - For 'both': n_components = n_bins + 1
                - For 'none': No parameters (fixed weights)
            mode: Weight adaptation strategy
                - 'per-bin': Independent weight per frequency bin (default)
                - 'global': Single weight for all bins
                - 'both': Global weight × per-bin weights (hierarchical)
                - 'none': Fixed unit weights (equivalent to BSP)
            init_value: Initial weight value (default: 1.0)

        Note:
            Weights are stored as log(weight) and exponentiated during forward pass.
            This ensures weights remain positive during optimization.
        """
        super().__init__()
        self.n_components = n_components
        self.mode = mode

        valid_modes = ['per-bin', 'global', 'both', 'none']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        # Initialize log-weights
        if mode == 'none':
            # No trainable parameters, use fixed weights
            self.register_buffer('log_weights', torch.zeros(n_components))
        else:
            # Trainable parameters
            init_log_value = np.log(init_value)
            self.log_weights = nn.Parameter(
                torch.full((n_components,), init_log_value, dtype=torch.float32)
            )

    def forward(self) -> torch.Tensor:
        """
        Get current adaptive weights.

        Returns:
            Weights [n_components], always positive

        Shape:
            Output: [n_components]
        """
        # Exponentiate to ensure positivity
        return torch.exp(self.log_weights)

    def get_statistics(self) -> dict:
        """
        Get weight statistics for monitoring.

        Returns:
            Dictionary with weight statistics:
            - 'mean': Mean weight
            - 'std': Standard deviation
            - 'min': Minimum weight
            - 'max': Maximum weight
            - 'weights': All weights [n_components]
        """
        with torch.no_grad():
            weights = self.forward()
            return {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'min': weights.min().item(),
                'max': weights.max().item(),
                'weights': weights.cpu().numpy()
            }


class SelfAdaptiveBSPLoss(nn.Module):
    """
    Self-Adaptive Binned Spectral Power Loss.

    Extends BSP loss with trainable per-bin weights that adapt during training
    to balance errors across frequency bands. More effective than fixed BSP at
    mitigating spectral bias.

    Attributes:
        bsp_module: Underlying BSP loss module
        adaptive_weights: Trainable weight module
        lambda_sa: Overall weight for SA-BSP loss
        adapt_mode: Weight adaptation mode
    """

    def __init__(
        self,
        n_bins: int = 32,
        lambda_sa: float = 1.0,
        adapt_mode: str = 'per-bin',
        init_weight: float = 1.0,
        epsilon: float = 1e-8,
        binning_mode: str = 'linear'
    ):
        """
        Initialize Self-Adaptive BSP Loss.

        Args:
            n_bins: Number of frequency bins (default: 32)
            lambda_sa: Overall weight for SA-BSP loss (default: 1.0)
            adapt_mode: Weight adaptation mode (default: 'per-bin')
                - 'per-bin': Independent weight per bin
                - 'global': Single weight for all bins
                - 'both': Global × per-bin (hierarchical)
                - 'none': Fixed unit weights (degenerates to BSP)
            init_weight: Initial weight value (default: 1.0)
            epsilon: Numerical stability constant (default: 1e-8)
            binning_mode: Frequency spacing ('linear' or 'log', default: 'linear')

        Note:
            When using SA-BSP in training, you MUST create a separate optimizer
            for the adaptive weights. See class docstring for example.
        """
        super().__init__()
        self.n_bins = n_bins
        self.lambda_sa = lambda_sa
        self.adapt_mode = adapt_mode
        self.epsilon = epsilon

        # Create underlying BSP module (lambda=1.0, will be weighted by adaptive weights)
        self.bsp_module = BinnedSpectralLoss(
            n_bins=n_bins,
            lambda_bsp=1.0,  # Fixed at 1.0, weighting handled by adaptive layer
            epsilon=epsilon,
            binning_mode=binning_mode
        )

        # Create adaptive weights
        if adapt_mode == 'global':
            n_components = 1
        elif adapt_mode == 'both':
            n_components = n_bins + 1  # n_bins per-bin + 1 global
        else:  # 'per-bin' or 'none'
            n_components = n_bins

        self.adaptive_weights = SelfAdaptiveWeights(
            n_components=n_components,
            mode=adapt_mode,
            init_value=init_weight
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Self-Adaptive BSP loss.

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Scalar SA-BSP loss (weighted)

        Algorithm:
            1. Compute per-bin spectral errors
            2. Get current adaptive weights
            3. Apply weights to per-bin errors
            4. Sum weighted errors
        """
        # Step 1: Compute per-bin errors
        # Shape: [n_bins]
        bin_errors = self.bsp_module.compute_bin_errors(pred, target)

        # Step 2: Get adaptive weights
        weights = self.adaptive_weights()  # [n_components]

        # Step 3: Apply adaptive weighting based on mode
        if self.adapt_mode == 'per-bin':
            # Direct per-bin weighting
            # weights: [n_bins], bin_errors: [n_bins]
            weighted_errors = weights * bin_errors

        elif self.adapt_mode == 'global':
            # Single global weight for all bins
            # weights: [1], bin_errors: [n_bins]
            weighted_errors = weights[0] * bin_errors

        elif self.adapt_mode == 'both':
            # Hierarchical: global × per-bin
            # weights: [n_bins+1] where weights[0] = global, weights[1:] = per-bin
            global_weight = weights[0]
            per_bin_weights = weights[1:]
            weighted_errors = global_weight * per_bin_weights * bin_errors

        else:  # 'none'
            # No adaptation, uniform weighting (equivalent to BSP)
            weighted_errors = bin_errors

        # Step 4: Sum weighted errors and apply overall lambda
        sa_bsp_loss = weighted_errors.mean()

        return self.lambda_sa * sa_bsp_loss

    def get_loss_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Get detailed loss breakdown for logging.

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Dictionary with:
            - 'bin_errors': Per-bin errors [n_bins]
            - 'weights': Current adaptive weights
            - 'weighted_errors': Per-bin weighted errors [n_bins]
            - 'total_loss': Total SA-BSP loss (scalar)
            - 'weight_stats': Weight statistics (mean, std, min, max)
        """
        with torch.no_grad():
            # Compute components
            bin_errors = self.bsp_module.compute_bin_errors(pred, target)
            weights = self.adaptive_weights()

            # Apply weighting
            if self.adapt_mode == 'per-bin':
                weighted_errors = weights * bin_errors
            elif self.adapt_mode == 'global':
                weighted_errors = weights[0] * bin_errors
            elif self.adapt_mode == 'both':
                global_weight = weights[0]
                per_bin_weights = weights[1:]
                weighted_errors = global_weight * per_bin_weights * bin_errors
            else:  # 'none'
                weighted_errors = bin_errors

            # Get weight statistics
            weight_stats = self.adaptive_weights.get_statistics()

        # Compute total loss (with gradients)
        total_loss = self.forward(pred, target)

        return {
            'bin_errors': bin_errors.cpu().numpy(),
            'weights': weights.cpu().numpy(),
            'weighted_errors': weighted_errors.cpu().numpy(),
            'total_loss': total_loss.item(),
            'weight_stats': weight_stats
        }

    def get_frequency_analysis(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Comprehensive frequency analysis with adaptive weights.

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Dictionary with frequency analysis including adaptive weights
        """
        # Get base frequency analysis from BSP module
        analysis = self.bsp_module.get_frequency_analysis(pred, target)

        # Add adaptive weight information
        with torch.no_grad():
            weights = self.adaptive_weights()
            weight_stats = self.adaptive_weights.get_statistics()

            analysis['adaptive_weights'] = weights.cpu().numpy()
            analysis['weight_stats'] = weight_stats
            analysis['adapt_mode'] = self.adapt_mode

        # Override loss with SA-BSP loss
        analysis['loss'] = self.forward(pred, target).item()

        return analysis


def create_optimizers_for_sa_bsp(
    model: nn.Module,
    loss_fn: SelfAdaptiveBSPLoss,
    model_lr: float = 1e-3,
    weight_lr: float = 1e-3
) -> tuple:
    """
    Helper function to create optimizers for SA-BSP training.

    Creates two optimizers:
    1. Model optimizer: For neural network parameters
    2. Weight optimizer: For adaptive weight parameters

    Args:
        model: Neural network model
        loss_fn: SA-BSP loss function
        model_lr: Learning rate for model (default: 1e-3)
        weight_lr: Learning rate for weights (default: 1e-3)

    Returns:
        Tuple of (model_optimizer, weight_optimizer)

    Example:
        >>> loss_fn = SelfAdaptiveBSPLoss(n_bins=32)
        >>> model_opt, weight_opt = create_optimizers_for_sa_bsp(model, loss_fn)
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     pred = model(input)
        ...     loss = loss_fn(pred, target)
        ...
        ...     model_opt.zero_grad()
        ...     weight_opt.zero_grad()
        ...     loss.backward()
        ...     model_opt.step()
        ...     weight_opt.step()
    """
    # Model optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)

    # Weight optimizer (only if weights are trainable)
    if loss_fn.adapt_mode != 'none':
        weight_optimizer = torch.optim.Adam(
            loss_fn.adaptive_weights.parameters(),
            lr=weight_lr
        )
    else:
        # No trainable weights, return None
        weight_optimizer = None

    return model_optimizer, weight_optimizer
