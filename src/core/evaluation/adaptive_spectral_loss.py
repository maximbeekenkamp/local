"""
Self-Adaptive Binned Spectral Power (SA-BSP) Loss for 1D temporal neural operators.

Extends BSP loss with learnable weights inspired by SA-PINNs (Self-Adaptive
Physics-Informed Neural Networks). Three variants:

1. **Per-bin** (32 weights): Emphasizes difficult frequency bins
2. **Global** (1 weight): Balances MSE vs BSP loss
3. **Hierarchical** (33 weights): Both global and per-bin adaptation

Key innovation: Uses SA-PINNs saddle-point optimization
    - Model: min_θ L(θ, λ)  (standard gradient descent)
    - Weights: max_λ L(θ, λ)  (negated gradients - ascent on loss)

For per-bin weights, this encourages the model to reduce errors in all frequency
bands by automatically increasing weights for hard-to-fit bins (typically high
frequencies), effectively mitigating spectral bias.

Implementation follows SA-PINNs:
- Raw weights (not log-space)
- Unconstrained (can grow unbounded)
- Negated gradients for per-bin weights
- Standard gradients for global weight

Reference:
- SA-PINNs: "Understanding and Mitigating Gradient Flow Pathologies in
  Physics-Informed Neural Networks" (McClenny & Braga-Neto, 2020)
- GitHub: https://github.com/levimcclenny/SA-PINNs

Training requirements:
    1. Main optimizer for model parameters
    2. Separate optimizer for adaptive weights
    3. Negate gradients for per-bin weights before optimizer step
    4. Standard gradients for global weight

Example (per-bin mode):
    >>> # Create loss
    >>> loss_fn = SelfAdaptiveSpectralLoss(n_bins=32, adapt_mode='per-bin')
    >>>
    >>> # Create optimizers
    >>> model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> weight_optimizer = torch.optim.Adam(loss_fn.adaptive_weights.parameters(), lr=1e-3)
    >>>
    >>> # Training loop
    >>> for batch in dataloader:
    ...     pred = model(input)
    ...     loss = loss_fn(pred, target)
    ...
    ...     model_optimizer.zero_grad()
    ...     weight_optimizer.zero_grad()
    ...     loss.backward()
    ...
    ...     # Standard descent for model
    ...     model_optimizer.step()
    ...
    ...     # SA-PINNs style: NEGATE gradients for per-bin weights (ascent on loss)
    ...     for param in loss_fn.adaptive_weights.parameters():
    ...         if param.grad is not None:
    ...             param.grad = -param.grad
    ...     weight_optimizer.step()
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .binned_spectral_loss import BinnedSpectralLoss


class SelfAdaptiveWeights(nn.Module):
    """
    Trainable adaptive weights for frequency bins.

    Uses RAW weights (not log-space) following SA-PINNs implementation.
    Weights are unconstrained and can grow unbounded, allowing the saddle-point
    optimization to naturally discover hard-to-fit frequency bins.

    Attributes:
        n_components: Number of weight components (e.g., n_bins for per-bin mode)
        mode: Weight adaptation mode ('per-bin', 'global', 'combined', 'none')
        weights: Trainable weights (parameters)
    """

    def __init__(
        self,
        n_components: int,
        mode: str = 'per-bin',
        init_values: torch.Tensor = None
    ):
        """
        Initialize adaptive weights.

        Args:
            n_components: Number of weight components
                - For 'per-bin': n_components = n_bins
                - For 'global': n_components = 2
                - For 'combined': n_components = n_bins + 2
                - For 'none': No parameters (fixed weights)
            mode: Weight adaptation strategy
                - 'per-bin': Independent weight per frequency bin (default)
                - 'global': Dual weights [w_mse, w_bsp] for MSE/BSP balance
                - 'combined': Global [w_mse, w_bsp] + per-bin [λ_k] weights
                - 'none': Fixed unit weights (equivalent to BSP)
            init_values: Initial weight values as tensor (default: ones if None)
                - For per-bin: [λ_k] (n_bins,) initialized from paper's λ_k
                - For global: [1.0, 1.0] (2,) for [w_mse, w_bsp]
                - For combined: [1.0, 1.0, λ_k...] (n_bins+2,)

        Note:
            Following SA-PINNs, weights are stored as RAW values (not log-space)
            and are unconstrained. This allows saddle-point optimization with
            negated gradients to naturally find optimal weights.
        """
        super().__init__()
        self.n_components = n_components
        self.mode = mode

        valid_modes = ['per-bin', 'global', 'combined', 'none']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        # Set default initialization if not provided
        if init_values is None:
            init_values = torch.ones(n_components, dtype=torch.float32)
        else:
            # Ensure correct shape
            if init_values.shape[0] != n_components:
                raise ValueError(
                    f"init_values shape {init_values.shape} doesn't match "
                    f"n_components {n_components}"
                )
            init_values = init_values.clone().float()

        # Initialize raw weights (SA-PINNs style)
        if mode == 'none':
            # No trainable parameters, use fixed weights
            self.register_buffer('weights', init_values)
        else:
            # Trainable parameters (raw weights, no log-space)
            self.weights = nn.Parameter(init_values)

    def forward(self) -> torch.Tensor:
        """
        Get current adaptive weights.

        Returns:
            Weights [n_components], unconstrained

        Shape:
            Output: [n_components]

        Note:
            No clipping or exp - returns raw weights directly.
            Following SA-PINNs, weights are unconstrained and can grow
            unbounded as training progresses.
        """
        return self.weights

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
        binning_mode: str = 'linear',
        signal_length: int = 4000,
        cache_path: str = None,
        lambda_k_mode: str = 'k_squared',
        use_log: bool = False
    ):
        """
        Initialize Self-Adaptive BSP Loss.

        Args:
            n_bins: Number of frequency bins (default: 32)
            lambda_sa: Overall weight for SA-BSP loss (default: 1.0)
            adapt_mode: Weight adaptation mode (default: 'per-bin')
                - 'per-bin': Independent weight per bin (32 weights)
                - 'global': Dual weights for MSE/BSP balance (2 weights: w_mse + w_bsp)
                - 'combined': MSE/BSP balance + per-bin (34 weights: w_mse + w_bsp + 32 per-bin)
                - 'none': Fixed unit weights (degenerates to BSP)
            init_weight: Initial weight value (default: 1.0)
            epsilon: Numerical stability constant (default: 1e-8)
            binning_mode: Frequency spacing ('linear' or 'log', default: 'linear')
            signal_length: Expected signal length in time dimension (default: 4000 for CDON)
                          Used to pre-compute static frequency bin edges for consistency
            cache_path: Optional path to precomputed spectrum cache (e.g., 'cache/true_spectrum.npz')
                       If provided, loads bin edges to ensure consistency with real data
            lambda_k_mode: Per-bin weight mode (λ_k from paper):
                - 'k_squared': λ_k = k² (paper Table 4, turbulence - default)
                - 'uniform': λ_k = 1 (paper Table 4, airfoil)
            use_log: Apply log10 to energies (log BSP variant, default: False)

        Note:
            When using SA-BSP in training, you MUST create a separate optimizer
            for the adaptive weights and use SA-PINNs style optimization (negated
            gradients for per-bin weights). See class docstring for details.
        """
        super().__init__()
        self.n_bins = n_bins
        self.lambda_sa = lambda_sa
        self.adapt_mode = adapt_mode
        self.epsilon = epsilon

        # Create underlying BSP module (μ=1.0, will be weighted by adaptive weights)
        self.bsp_module = BinnedSpectralLoss(
            n_bins=n_bins,
            mu=1.0,  # Fixed at 1.0, weighting handled by adaptive layer
            epsilon=epsilon,
            binning_mode=binning_mode,
            signal_length=signal_length,
            cache_path=cache_path,
            lambda_k_mode=lambda_k_mode,
            use_log=use_log
        )

        # Create adaptive weights with proper initialization
        # Initialize based on mode (per user's specification)
        if adapt_mode == 'global':
            # 2 weights: [w_mse=1.0, w_bsp=1.0]
            n_components = 2
            init_values = torch.tensor([1.0, 1.0], dtype=torch.float32)
        elif adapt_mode == 'combined':
            # 34 weights: [w_mse=1.0, w_bsp=1.0, 32×λ_k=k²]
            n_components = n_bins + 2
            w_global = torch.tensor([1.0, 1.0], dtype=torch.float32)
            lambda_k = self.bsp_module.bin_weights.clone()  # λ_k from BSP (k² or uniform)
            init_values = torch.cat([w_global, lambda_k])
        elif adapt_mode == 'per-bin':
            # 32 weights: λ_k initialized from BSP module (k² or uniform)
            n_components = n_bins
            init_values = self.bsp_module.bin_weights.clone()
        else:  # 'none'
            # Fixed weights (no training)
            n_components = n_bins
            init_values = torch.ones(n_bins, dtype=torch.float32)

        self.adaptive_weights = SelfAdaptiveWeights(
            n_components=n_components,
            mode=adapt_mode,
            init_values=init_values
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

        # Step 2: Get adaptive weights and ensure they're on the same device as bin_errors
        weights = self.adaptive_weights()  # [n_components]
        weights = weights.to(bin_errors.device)  # Ensure same device

        # Step 3: Apply adaptive weighting based on mode
        if self.adapt_mode == 'per-bin':
            # Direct per-bin weighting
            # weights: [n_bins], bin_errors: [n_bins]
            # Uses SA-PINNs style: emphasize high-error bins via competitive dynamics
            weighted_errors = weights * bin_errors
            sa_bsp_loss = weighted_errors.mean()

        elif self.adapt_mode == 'global':
            # Competitive dynamics for MSE/BSP balance
            # weights: [2] = [w_mse, w_bsp]
            # Returns only BSP component; CombinedLoss applies both weights
            total_bsp = bin_errors.mean()
            sa_bsp_loss = weights[1] * total_bsp
            # Note: w_mse (weights[0]) is applied in CombinedLoss for MSE term

        elif self.adapt_mode == 'combined':
            # Combined: Global MSE/BSP balance + per-bin frequency emphasis
            # weights: [n_bins+2] = [w_mse, w_bsp, w1, w2, ..., w_n_bins]
            # All weights use competitive dynamics (negated gradients)
            w_mse = weights[0]     # Not used here, applied in CombinedLoss
            w_bsp = weights[1]     # Global BSP weight
            per_bin_weights = weights[2:]  # Per-bin weights
            # Apply: w_bsp * mean(per_bin_weights * bin_errors)
            weighted_errors = per_bin_weights * bin_errors
            sa_bsp_loss = w_bsp * weighted_errors.mean()

        else:  # 'none'
            # No adaptation, uniform weighting (equivalent to BSP)
            sa_bsp_loss = bin_errors.mean()

        # Apply overall lambda scaling
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
            weights = weights.to(bin_errors.device)  # Ensure same device

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
            weights = weights.to(pred.device)  # Ensure same device
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
