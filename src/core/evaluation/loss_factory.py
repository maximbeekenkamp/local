"""
Loss function factory for configurable loss selection.

Supports multiple loss types for ablation studies:
- Relative L2 Loss (baseline)
- Binned Spectral Power (BSP) Loss
- Self-Adaptive BSP (SA-BSP) Loss
- Combined losses with weighting

Usage:
    from configs.loss_config import LossConfig, BASELINE_CONFIG, BSP_CONFIG
    from src.core.evaluation.loss_factory import create_loss

    # Create baseline loss
    loss_fn = create_loss(BASELINE_CONFIG)

    # Or directly from dict
    loss_fn = create_loss_from_dict({
        'loss_type': 'relative_l2',
        'loss_params': {}
    })

    # Use in training
    loss = loss_fn(prediction, ground_truth)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union

from configs.loss_config import LossConfig
from .metrics import RelativeL2Loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Base loss + weighted spectral loss.

    Formula:
        L_total = L_base(pred, target) + λ × L_spectral(pred, target)

    This is the standard approach for spectral bias mitigation:
    - Always use a base loss (MSE/RelativeL2) for overall accuracy
    - Add spectral loss to encourage correct frequency distribution
    - Weight λ controls the balance

    Reference: BSP paper always uses combined loss, never spectral alone
    """

    def __init__(
        self,
        base_loss: nn.Module,
        spectral_loss: nn.Module,
        lambda_spectral: float = 1.0
    ):
        """
        Initialize combined loss.

        Args:
            base_loss: Base loss module (e.g., RelativeL2Loss)
            spectral_loss: Spectral loss module (e.g., BinnedSpectralLoss)
            lambda_spectral: Weight for spectral component (default: 1.0)
        """
        super().__init__()
        self.base_loss = base_loss
        self.spectral_loss = spectral_loss
        self.lambda_spectral = lambda_spectral

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Scalar loss value

        Note:
            For SA-BSP global/combined modes with competitive dynamics,
            weights are applied directly to MSE and BSP terms rather than
            using lambda_spectral.
        """
        # Import here to avoid circular dependency
        from .adaptive_spectral_loss import SelfAdaptiveBSPLoss

        loss_base = self.base_loss(pred, target)
        loss_spectral = self.spectral_loss(pred, target)

        # Check if using SA-BSP global or combined mode (competitive dynamics for MSE/BSP)
        if isinstance(self.spectral_loss, SelfAdaptiveBSPLoss):
            adapt_mode = self.spectral_loss.adapt_mode

            if adapt_mode in ['global', 'combined']:
                # Global/Combined: Apply adaptive weights to both MSE and BSP
                # weights[0] = w_mse, weights[1] = w_bsp (for global)
                # weights[0] = w_mse, weights[1:] = w_bsp + per-bin (for combined)
                weights = self.spectral_loss.adaptive_weights()
                w_mse = weights[0]
                # Spectral loss already has w_bsp applied in its forward pass
                total_loss = w_mse * loss_base + self.lambda_spectral * loss_spectral
            else:
                # Per-bin mode: Standard formulation (MSE fixed, only BSP adaptive)
                total_loss = loss_base + self.lambda_spectral * loss_spectral
        else:
            # Standard BSP or other spectral losses
            total_loss = loss_base + self.lambda_spectral * loss_spectral

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

        # Apply same logic as forward() for global/combined modes
        if isinstance(self.spectral_loss, SelfAdaptiveBSPLoss):
            adapt_mode = self.spectral_loss.adapt_mode
            if adapt_mode in ['global', 'combined']:
                weights = self.spectral_loss.adaptive_weights()
                w_mse = weights[0]
                total_loss = w_mse * loss_base + self.lambda_spectral * loss_spectral
            else:
                total_loss = loss_base + self.lambda_spectral * loss_spectral
        else:
            total_loss = loss_base + self.lambda_spectral * loss_spectral

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
        >>> loss = create_loss({'loss_type': 'relative_l2', 'loss_params': {}})
    """
    # Convert dict to LossConfig if needed
    if isinstance(config, dict):
        config = LossConfig.from_dict(config)

    loss_type = config.loss_type
    params = config.loss_params

    # Case 1: Relative L2 Loss (baseline)
    if loss_type == 'relative_l2':
        epsilon = params.get('epsilon', 1e-8)
        return RelativeL2Loss(epsilon=epsilon)

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
        lambda_bsp = params.get('lambda_bsp', 1.0)
        epsilon = params.get('epsilon', 1e-8)
        binning_mode = params.get('binning_mode', 'linear')
        signal_length = params.get('signal_length', 4000)
        cache_path = params.get('cache_path', None)

        return BinnedSpectralLoss(
            n_bins=n_bins,
            lambda_bsp=lambda_bsp,
            epsilon=epsilon,
            binning_mode=binning_mode,
            signal_length=signal_length,
            cache_path=cache_path
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

        return SelfAdaptiveBSPLoss(
            n_bins=n_bins,
            lambda_sa=lambda_sa,
            adapt_mode=adapt_mode,
            init_weight=init_weight,
            epsilon=epsilon,
            binning_mode=binning_mode,
            signal_length=signal_length,
            cache_path=cache_path
        )

    # Case 4: Combined Loss (Base + Spectral)
    elif loss_type == 'combined':
        # Get base loss type
        base_loss_type = params.get('base_loss', 'relative_l2')
        if base_loss_type == 'relative_l2':
            epsilon = params.get('epsilon', 1e-8)
            base_loss = RelativeL2Loss(epsilon=epsilon)
        else:
            raise ValueError(
                f"Unknown base_loss type: {base_loss_type}. "
                f"Currently only 'relative_l2' is supported."
            )

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

            spectral_loss = BinnedSpectralLoss(
                n_bins=n_bins,
                lambda_bsp=1.0,  # Set to 1.0, weight applied in CombinedLoss
                epsilon=epsilon,
                binning_mode=binning_mode,
                signal_length=signal_length,
                cache_path=cache_path
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

            spectral_loss = SelfAdaptiveBSPLoss(
                n_bins=n_bins,
                lambda_sa=1.0,  # Set to 1.0, weight applied in CombinedLoss
                adapt_mode=adapt_mode,
                init_weight=init_weight,
                epsilon=epsilon,
                binning_mode=binning_mode,
                signal_length=signal_length,
                cache_path=cache_path
            )

        else:
            raise ValueError(
                f"Unknown spectral_loss type: {spectral_loss_type}. "
                f"Must be 'bsp' or 'sa_bsp'."
            )

        # Get spectral weight
        lambda_spectral = params.get('lambda_spectral', 1.0)

        # Create combined loss
        return CombinedLoss(
            base_loss=base_loss,
            spectral_loss=spectral_loss,
            lambda_spectral=lambda_spectral
        )

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Must be one of: 'relative_l2', 'bsp', 'sa_bsp', 'combined'"
        )


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
        ...         'base_loss': 'relative_l2',
        ...         'spectral_loss': 'bsp',
        ...         'lambda_spectral': 1.0,
        ...         'n_bins': 32
        ...     }
        ... })
    """
    config = LossConfig.from_dict(config_dict)
    return create_loss(config)
