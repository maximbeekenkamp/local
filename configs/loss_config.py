"""
Loss function configuration for neural operator training.

Supports multiple loss types for ablation studies:
- Relative L2 Loss (baseline)
- Binned Spectral Power (BSP) Loss
- Self-Adaptive BSP (SA-BSP) Loss
- Combined losses with weighting

Example configurations:

    # Baseline (Relative L2 only)
    loss_config = LossConfig(
        loss_type='relative_l2',
        loss_params={}
    )

    # BSP Loss (MSE + Spectral)
    loss_config = LossConfig(
        loss_type='combined',
        loss_params={
            'base_loss': 'relative_l2',
            'spectral_loss': 'bsp',
            'lambda_spectral': 1.0,
            'n_bins': 32,
            'epsilon': 1e-8
        }
    )

    # SA-BSP Loss (MSE + Adaptive Spectral)
    loss_config = LossConfig(
        loss_type='combined',
        loss_params={
            'base_loss': 'relative_l2',
            'spectral_loss': 'sa_bsp',
            'lambda_spectral': 1.0,
            'n_bins': 32,
            'adapt_mode': 'per-bin',
            'epsilon': 1e-8
        }
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LossConfig:
    """
    Configuration for loss functions in neural operator training.

    Attributes:
        loss_type: Type of loss function to use
            - 'relative_l2': Relative L2 loss (baseline)
            - 'bsp': Binned Spectral Power loss
            - 'sa_bsp': Self-Adaptive BSP loss
            - 'combined': Combination of base + spectral loss
        loss_params: Dictionary of loss-specific parameters
        description: Optional description for logging

    Loss-specific parameters:

        For 'relative_l2':
            No additional parameters needed

        For 'bsp':
            - n_bins (int): Number of frequency bins (default: 32)
            - lambda_bsp (float): Weight for spectral loss (default: 1.0)
            - epsilon (float): Numerical stability constant (default: 1e-8)
            - binning_mode (str): 'linear' or 'log' spacing (default: 'linear')

        For 'sa_bsp':
            - n_bins (int): Number of frequency bins (default: 32)
            - lambda_sa (float): Weight for adaptive spectral loss (default: 1.0)
            - adapt_mode (str): Weight adaptation mode (default: 'per-bin')
                * 'per-bin': Independent weight per bin
                * 'global': Single weight for all bins
                * 'both': Hierarchical (global × per-bin)
                * 'none': Fixed unit weights (equivalent to BSP)
            - init_weight (float): Initial weight value (default: 1.0)
            - epsilon (float): Numerical stability constant (default: 1e-8)
            - binning_mode (str): 'linear' or 'log' spacing (default: 'linear')

        For 'combined':
            - base_loss (str): Base loss type ('relative_l2')
            - spectral_loss (str): Spectral loss type ('bsp' or 'sa_bsp')
            - lambda_spectral (float): Weight for spectral component (default: 1.0)
            - n_bins (int): Number of frequency bins (default: 32)
            - epsilon (float): Numerical stability constant (default: 1e-8)
            - binning_mode (str): 'linear' or 'log' spacing (default: 'linear')
            - adapt_mode (str): For SA-BSP, weight adaptation mode (default: 'per-bin')
            - init_weight (float): For SA-BSP, initial weight value (default: 1.0)
    """
    loss_type: str
    loss_params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def __post_init__(self):
        """Validate loss configuration."""
        valid_types = ['relative_l2', 'bsp', 'sa_bsp', 'combined']
        if self.loss_type not in valid_types:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. "
                f"Must be one of {valid_types}"
            )

        # Validate combined loss has required parameters
        if self.loss_type == 'combined':
            if 'base_loss' not in self.loss_params:
                raise ValueError("Combined loss requires 'base_loss' parameter")
            if 'spectral_loss' not in self.loss_params:
                raise ValueError("Combined loss requires 'spectral_loss' parameter")

            # Validate spectral loss type
            valid_spectral = ['bsp', 'sa_bsp']
            if self.loss_params['spectral_loss'] not in valid_spectral:
                raise ValueError(
                    f"Invalid spectral_loss: {self.loss_params['spectral_loss']}. "
                    f"Must be one of {valid_spectral}"
                )

        # Validate adapt_mode for SA-BSP
        if self.loss_type in ['sa_bsp', 'combined']:
            if self.loss_type == 'sa_bsp' or \
               (self.loss_type == 'combined' and
                self.loss_params.get('spectral_loss') == 'sa_bsp'):
                adapt_mode = self.loss_params.get('adapt_mode', 'per-bin')
                valid_modes = ['per-bin', 'global', 'both', 'none']
                if adapt_mode not in valid_modes:
                    raise ValueError(
                        f"Invalid adapt_mode: {adapt_mode}. "
                        f"Must be one of {valid_modes}"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'loss_type': self.loss_type,
            'loss_params': self.loss_params,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LossConfig':
        """Create LossConfig from dictionary."""
        return cls(
            loss_type=config_dict['loss_type'],
            loss_params=config_dict.get('loss_params', {}),
            description=config_dict.get('description')
        )

    def __repr__(self) -> str:
        """Pretty string representation."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.loss_params.items())
        desc_str = f" ({self.description})" if self.description else ""
        return f"LossConfig(type={self.loss_type}, params={{{params_str}}}){desc_str}"


# Predefined configurations for common use cases
BASELINE_CONFIG = LossConfig(
    loss_type='relative_l2',
    loss_params={},
    description='Baseline: Relative L2 loss only'
)

BSP_CONFIG = LossConfig(
    loss_type='combined',
    loss_params={
        'base_loss': 'relative_l2',
        'spectral_loss': 'bsp',
        'lambda_spectral': 0.1,  # Paper's Airfoil value (Table 4, Page 26)
        'n_bins': 32,
        'epsilon': 1e-6,  # Increased from 1e-8 per paper ablation (Table 2)
        'binning_mode': 'linear'
    },
    description='MSE + Binned Spectral Power loss'
)

SA_BSP_CONFIG = LossConfig(
    loss_type='combined',
    loss_params={
        'base_loss': 'relative_l2',
        'spectral_loss': 'sa_bsp',
        'lambda_spectral': 0.1,  # Paper's Airfoil value (Table 4, Page 26)
        'n_bins': 32,
        'adapt_mode': 'per-bin',
        'init_weight': 1.0,
        'epsilon': 1e-6,  # Increased from 1e-8 per paper ablation (Table 2)
        'binning_mode': 'linear'
    },
    description='MSE + Self-Adaptive BSP loss'
)
