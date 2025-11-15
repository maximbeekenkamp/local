"""
Loss function factory for configurable loss selection.

Supports multiple loss types for ablation studies:
- Field Error Loss (primary MSE metric - relative MSE in real space)
- Binned Spectral Power (BSP) Loss
- Self-Adaptive BSP (SA-BSP) Loss
- Combined losses with weighting

Usage:
    from configs.loss_config import LossConfig, BASELINE_CONFIG, BSP_CONFIG
    from src.core.evaluation.loss_factory import create_loss

    # Create field error loss
    loss_fn = create_loss_from_dict({
        'loss_type': 'field_error',
        'loss_params': {}
    })

    # Or use config objects
    loss_fn = create_loss(BASELINE_CONFIG)

    # Use in training
    loss = loss_fn(prediction, ground_truth)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union

from configs.loss_config import LossConfig
from .metrics import FieldErrorLoss
from .penalty_loss import PenaltyWeightedLoss, MSEWithPenalty


class CombinedLoss(nn.Module):
    """
    Combined loss: Base loss + spectral loss.

    Formula:
        L_total = L_base(pred, target) + L_spectral(pred, target)

    This is the standard approach for spectral bias mitigation:
    - Always use a base loss (MSE/FieldError) for overall accuracy
    - Add spectral loss to encourage correct frequency distribution
    - Weighting controlled by μ in spectral loss (always 1.0)

    Reference: BSP paper always uses combined loss, never spectral alone
    """

    def __init__(
        self,
        base_loss: nn.Module,
        spectral_loss: nn.Module
    ):
        """
        Initialize combined loss.

        Args:
            base_loss: Base loss module (e.g., FieldErrorLoss)
            spectral_loss: Spectral loss module (e.g., BinnedSpectralLoss with μ=1.0)
        """
        super().__init__()
        self.base_loss = base_loss
        self.spectral_loss = spectral_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Formula:
            L_total = w_mse × L_base + w_spectral × L_spectral

        Where:
            - w_mse: Weight for base loss (default: 1.0, adaptive for global/combined SA-BSP)
            - w_spectral: Weight for spectral loss (μ, default: 1.0, adaptive for global/combined SA-BSP)

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Scalar loss value

        Note:
            Even though w_mse=1.0 and w_spectral=1.0 for most modes, they are
            explicitly included in the formula for clarity and consistency.
        """
        # Import here to avoid circular dependency
        from .adaptive_spectral_loss import SelfAdaptiveBSPLoss

        loss_base = self.base_loss(pred, target)
        loss_spectral = self.spectral_loss(pred, target)

        # Default weights (static)
        w_mse = 1.0
        w_spectral = 1.0  # This is μ (mu) in BSP terminology

        # For SA-BSP global/combined modes, use adaptive weights
        if isinstance(self.spectral_loss, SelfAdaptiveBSPLoss):
            adapt_mode = self.spectral_loss.adapt_mode

            if adapt_mode in ['global', 'combined']:
                # Global/Combined: Apply adaptive weights to both MSE and BSP
                # weights[0] = w_mse (adaptive)
                # weights[1] = w_bsp (adaptive, this is w_spectral/μ)
                weights = self.spectral_loss.adaptive_weights()
                w_mse = weights[0]
                w_spectral = weights[1]

        # Combined loss formula (same structure for all modes)
        total_loss = w_mse * loss_base + w_spectral * loss_spectral
        return total_loss

    def get_loss_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual loss components for logging.

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Dictionary with 'base', 'spectral', and 'total' losses
        """
        # Import here to avoid circular dependency
        from .adaptive_spectral_loss import SelfAdaptiveBSPLoss

        loss_base = self.base_loss(pred, target)
        loss_spectral = self.spectral_loss(pred, target)

        # Default weights (static)
        w_mse = 1.0
        w_spectral = 1.0  # This is μ (mu) in BSP terminology

        # For SA-BSP global/combined modes, use adaptive weights
        if isinstance(self.spectral_loss, SelfAdaptiveBSPLoss):
            adapt_mode = self.spectral_loss.adapt_mode
            if adapt_mode in ['global', 'combined']:
                weights = self.spectral_loss.adaptive_weights()
                w_mse = weights[0]
                w_spectral = weights[1]

        # Combined loss formula (same structure as forward())
        total_loss = w_mse * loss_base + w_spectral * loss_spectral

        return {
            'base': loss_base,
            'spectral': loss_spectral,
            'total': total_loss
        }


def create_loss(config: Union[LossConfig, Dict[str, Any]]) -> nn.Module:
    """
    Factory function to create loss modules from configuration.

    Args:
        config: LossConfig object or dictionary with loss configuration

    Returns:
        Loss module (nn.Module) that implements forward(pred, target) -> loss

    Raises:
        ValueError: If loss type is unknown or parameters are invalid
        ImportError: If required loss module is not yet implemented

    Examples:
        >>> from configs.loss_config import BASELINE_CONFIG, BSP_CONFIG
        >>> loss = create_loss(BASELINE_CONFIG)
        >>> loss = create_loss({'loss_type': 'field_error', 'loss_params': {}})
    """
    # Convert dict to LossConfig if needed
    if isinstance(config, dict):
        config = LossConfig.from_dict(config)

    loss_type = config.loss_type
    params = config.loss_params

    # Extract penalty parameter (optional)
    use_penalty = params.get('use_penalty', False)
    penalty_epsilon = params.get('penalty_epsilon', 1e-8)
    penalty_per_sample = params.get('penalty_per_sample', True)

    # Create base loss
    base_loss = None

    # Case 1: Field Error Loss
    if loss_type == 'field_error':
        epsilon = params.get('epsilon', 1e-8)
        base_loss = FieldErrorLoss(epsilon=epsilon)

    # Case 2: Binned Spectral Power (BSP) Loss
    elif loss_type == 'bsp':
        try:
            from .binned_spectral_loss import BinnedSpectralLoss
        except ImportError:
            raise ImportError(
                "BinnedSpectralLoss not yet implemented. "
                "Run implementation step 2 to create binned_spectral_loss.py"
            )

        n_bins = params.get('n_bins', 32)
        mu = params.get('mu', 1.0)
        epsilon = params.get('epsilon', 1e-8)
        binning_mode = params.get('binning_mode', 'linear')
        signal_length = params.get('signal_length', 4000)
        cache_path = params.get('cache_path', None)
        lambda_k_mode = params.get('lambda_k_mode', 'k_squared')
        use_log = params.get('use_log', False)
        use_output_norm = params.get('use_output_norm', True)
        use_minmax_norm = params.get('use_minmax_norm', True)
        loss_type = params.get('loss_type', 'mspe')

        base_loss = BinnedSpectralLoss(
            n_bins=n_bins,
            mu=mu,
            epsilon=epsilon,
            binning_mode=binning_mode,
            signal_length=signal_length,
            cache_path=cache_path,
            lambda_k_mode=lambda_k_mode,
            use_log=use_log,
            use_output_norm=use_output_norm,
            use_minmax_norm=use_minmax_norm,
            loss_type=loss_type
        )

    # Case 3: Self-Adaptive BSP Loss
    elif loss_type == 'sa_bsp':
        try:
            from .adaptive_spectral_loss import SelfAdaptiveBSPLoss
        except ImportError:
            raise ImportError(
                "SelfAdaptiveBSPLoss not yet implemented. "
                "Run implementation step 4 to create adaptive_spectral_loss.py"
            )

        n_bins = params.get('n_bins', 32)
        lambda_sa = params.get('lambda_sa', 1.0)
        adapt_mode = params.get('adapt_mode', 'per-bin')
        init_weight = params.get('init_weight', 1.0)
        epsilon = params.get('epsilon', 1e-8)
        binning_mode = params.get('binning_mode', 'linear')
        signal_length = params.get('signal_length', 4000)
        cache_path = params.get('cache_path', None)
        lambda_k_mode = params.get('lambda_k_mode', 'k_squared')
        use_log = params.get('use_log', False)
        use_output_norm = params.get('use_output_norm', True)
        use_minmax_norm = params.get('use_minmax_norm', True)
        loss_type = params.get('loss_type', 'mspe')

        base_loss = SelfAdaptiveBSPLoss(
            n_bins=n_bins,
            lambda_sa=lambda_sa,
            adapt_mode=adapt_mode,
            init_weight=init_weight,
            epsilon=epsilon,
            binning_mode=binning_mode,
            signal_length=signal_length,
            cache_path=cache_path,
            lambda_k_mode=lambda_k_mode,
            use_log=use_log,
            use_output_norm=use_output_norm,
            use_minmax_norm=use_minmax_norm,
            loss_type=loss_type
        )

    # Case 4: Combined Loss (Base + Spectral)
    elif loss_type == 'combined':
        # Get base loss type (only field_error supported)
        base_loss_type = params.get('base_loss', 'field_error')
        if base_loss_type != 'field_error':
            raise ValueError(
                f"Unknown base_loss type: {base_loss_type}. "
                f"Only 'field_error' is supported."
            )
        epsilon = params.get('epsilon', 1e-8)
        base_loss = FieldErrorLoss(epsilon=epsilon)

        # Get spectral loss type
        spectral_loss_type = params.get('spectral_loss')
        if spectral_loss_type is None:
            raise ValueError("Combined loss requires 'spectral_loss' parameter")

        if spectral_loss_type == 'bsp':
            try:
                from .binned_spectral_loss import BinnedSpectralLoss
            except ImportError:
                raise ImportError(
                    "BinnedSpectralLoss not yet implemented. "
                    "Run implementation step 2 to create binned_spectral_loss.py"
                )

            n_bins = params.get('n_bins', 32)
            epsilon = params.get('epsilon', 1e-8)
            binning_mode = params.get('binning_mode', 'linear')
            signal_length = params.get('signal_length', 4000)
            cache_path = params.get('cache_path', None)
            lambda_k_mode = params.get('lambda_k_mode', 'k_squared')
            use_log = params.get('use_log', False)
            use_output_norm = params.get('use_output_norm', True)
            use_minmax_norm = params.get('use_minmax_norm', True)
            loss_type = params.get('loss_type', 'mspe')

            spectral_loss = BinnedSpectralLoss(
                n_bins=n_bins,
                mu=1.0,  # μ=1.0, weighting handled by w_spectral in CombinedLoss
                epsilon=epsilon,
                binning_mode=binning_mode,
                signal_length=signal_length,
                cache_path=cache_path,
                lambda_k_mode=lambda_k_mode,
                use_log=use_log,
                use_output_norm=use_output_norm,
                use_minmax_norm=use_minmax_norm,
                loss_type=loss_type
            )

        elif spectral_loss_type == 'sa_bsp':
            try:
                from .adaptive_spectral_loss import SelfAdaptiveBSPLoss
            except ImportError:
                raise ImportError(
                    "SelfAdaptiveBSPLoss not yet implemented. "
                    "Run implementation step 4 to create adaptive_spectral_loss.py"
                )

            n_bins = params.get('n_bins', 32)
            adapt_mode = params.get('adapt_mode', 'per-bin')
            init_weight = params.get('init_weight', 1.0)
            epsilon = params.get('epsilon', 1e-8)
            binning_mode = params.get('binning_mode', 'linear')
            signal_length = params.get('signal_length', 4000)
            cache_path = params.get('cache_path', None)
            lambda_k_mode = params.get('lambda_k_mode', 'k_squared')
            use_log = params.get('use_log', False)
            use_output_norm = params.get('use_output_norm', True)
            use_minmax_norm = params.get('use_minmax_norm', True)
            loss_type = params.get('loss_type', 'mspe')

            spectral_loss = SelfAdaptiveBSPLoss(
                n_bins=n_bins,
                lambda_sa=1.0,  # μ=1.0, weighting handled by w_spectral in CombinedLoss
                adapt_mode=adapt_mode,
                init_weight=init_weight,
                epsilon=epsilon,
                binning_mode=binning_mode,
                signal_length=signal_length,
                cache_path=cache_path,
                lambda_k_mode=lambda_k_mode,
                use_log=use_log,
                use_output_norm=use_output_norm,
                use_minmax_norm=use_minmax_norm,
                loss_type=loss_type
            )

        else:
            raise ValueError(
                f"Unknown spectral_loss type: {spectral_loss_type}. "
                f"Must be 'bsp' or 'sa_bsp'."
            )

        # Create combined loss (weighting controlled by μ=1.0 in spectral loss)
        base_loss = CombinedLoss(
            base_loss=base_loss,
            spectral_loss=spectral_loss
        )

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Must be one of: 'field_error', 'bsp', 'sa_bsp', 'combined'"
        )

    # Apply penalty weighting if requested
    if use_penalty:
        return PenaltyWeightedLoss(
            base_loss=base_loss,
            epsilon=penalty_epsilon,
            per_sample=penalty_per_sample
        )
    else:
        return base_loss


def create_loss_from_dict(config_dict: Dict[str, Any]) -> nn.Module:
    """
    Convenience function to create loss from dictionary.

    Args:
        config_dict: Dictionary with 'loss_type' and 'loss_params' keys

    Returns:
        Loss module

    Example:
        >>> loss = create_loss_from_dict({
        ...     'loss_type': 'combined',
        ...     'loss_params': {
        ...         'base_loss': 'field_error',
        ...         'spectral_loss': 'bsp',
        ...         'n_bins': 32
        ...     }
        ... })
    """
    config = LossConfig.from_dict(config_dict)
    return create_loss(config)
