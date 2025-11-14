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

    # BSP Loss (MSE + Spectral, weighting controlled by μ=1.0)
    loss_config = LossConfig(
        loss_type='combined',
        loss_params={
            'base_loss': 'field_error',
            'spectral_loss': 'bsp',
            'n_bins': 32,
            'epsilon': 1e-8
        }
    )

    # SA-BSP Loss (MSE + Adaptive Spectral, weighting controlled by μ=1.0)
    loss_config = LossConfig(
        loss_type='combined',
        loss_params={
            'base_loss': 'field_error',
            'spectral_loss': 'sa_bsp',
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
            - signal_length (int): Expected signal length in time dimension (default: 4000)
            - cache_path (str): Path to precomputed spectrum cache for loading bin edges (optional)

        For 'sa_bsp':
            - n_bins (int): Number of frequency bins (default: 32)
            - lambda_sa (float): Weight for adaptive spectral loss (default: 1.0)
            - adapt_mode (str): Weight adaptation mode (default: 'per-bin')
                * 'per-bin': Independent weight per bin (32 weights)
                * 'global': Dual weights for MSE/BSP balance (2 weights: w_mse + w_bsp)
                * 'combined': Global MSE/BSP balance + per-bin (34 weights: w_mse + w_bsp + 32 per-bin)
                * 'none': Fixed unit weights (equivalent to BSP)
            - init_weight (float): Initial weight value (default: 1.0)
            - epsilon (float): Numerical stability constant (default: 1e-8)
            - binning_mode (str): 'linear' or 'log' spacing (default: 'linear')
            - signal_length (int): Expected signal length in time dimension (default: 4000)
            - cache_path (str): Path to precomputed spectrum cache for loading bin edges (optional)

        For 'combined':
            - base_loss (str): Base loss type ('relative_l2')
            - spectral_loss (str): Spectral loss type ('bsp' or 'sa_bsp')
            - lambda_spectral (float): Weight for spectral component (default: 1.0)
            - n_bins (int): Number of frequency bins (default: 32)
            - epsilon (float): Numerical stability constant (default: 1e-8)
            - binning_mode (str): 'linear' or 'log' spacing (default: 'linear')
            - signal_length (int): Expected signal length in time dimension (default: 4000)
            - cache_path (str): Path to precomputed spectrum cache for loading bin edges (optional)
            - adapt_mode (str): For SA-BSP, weight adaptation mode (default: 'per-bin')
            - init_weight (float): For SA-BSP, initial weight value (default: 1.0)

        Penalty Loss Parameters (Optional - applies to all loss types):
            - use_penalty (bool): Apply inverse-variance penalty weighting (default: False)
                                 From reference: loss *= 1 / (max(abs(target))² + ε)
                                 Emphasizes samples with larger responses
            - penalty_epsilon (float): Numerical stability for penalty (default: 1e-8)
            - penalty_per_sample (bool): Compute penalty per sample (True) or global (False)
                                        (default: True)

        Reference:
            Penalty weighting from Penwarden et al. "A metalearning approach for
            physics-informed neural networks" (2023), CausalityDeepONet implementation.
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
                valid_modes = ['per-bin', 'global', 'combined', 'fft', 'none']
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
        'base_loss': 'field_error',
        'spectral_loss': 'bsp',
        'mu': 1.0,  # μ from paper (Table 4 - turbulence cases)
        'n_bins': 32,
        'epsilon': 1e-8,  # Paper default
        'binning_mode': 'linear',
        'signal_length': 4000,  # CDON temporal resolution
        'cache_path': 'cache/true_spectrum.npz',  # Load bin edges from precomputed cache
        'lambda_k_mode': 'k_squared',  # λ_k = k² from paper Table 4
        'use_log': False,  # Standard energy (not log10)
        'use_output_norm': True,  # Per-batch output normalization: y = (y - mean) / std
        'use_minmax_norm': True,  # Per-sample min-max normalization of binned energies
        'loss_type': 'mspe'  # Mean Squared Percentage Error (matches BSP paper Algorithm 1)
    },
    description='MSE + BSP (μ=1.0, λ_k=k²) with normalization - Paper Table 4 turbulence'
)

SA_BSP_PERBIN_CONFIG = LossConfig(
    loss_type='combined',
    loss_params={
        'base_loss': 'field_error',
        'spectral_loss': 'sa_bsp',
        'n_bins': 32,
        'adapt_mode': 'per-bin',  # 32 trainable weights (one per bin)
        'init_weight': 1.0,
        'epsilon': 1e-6,  # Increased from 1e-8 per paper ablation (Table 2)
        'binning_mode': 'linear',
        'signal_length': 4000,  # CDON temporal resolution
        'cache_path': 'cache/true_spectrum.npz',  # Load bin edges from precomputed cache
        'lambda_k_mode': 'k_squared',  # λ_k = k² initialization for trainable weights
        'use_log': False,  # Standard energy (not log10)
        'use_output_norm': True,  # Per-batch output normalization
        'use_minmax_norm': True,  # Per-sample min-max normalization
        'loss_type': 'mspe'  # Mean Squared Percentage Error
    },
    description='MSE + SA-BSP (per-bin): 32 trainable λ_k weights (init: k²) with normalization - Paper Table 4'
)

SA_BSP_GLOBAL_CONFIG = LossConfig(
    loss_type='combined',
    loss_params={
        'base_loss': 'field_error',
        'spectral_loss': 'sa_bsp',
        'n_bins': 32,
        'adapt_mode': 'global',  # 2 trainable weights (w_mse + w_bsp) for MSE/BSP balance
        'init_weight': 1.0,
        'epsilon': 1e-6,  # Increased from 1e-8 per paper ablation (Table 2)
        'binning_mode': 'linear',
        'signal_length': 4000,  # CDON temporal resolution
        'cache_path': 'cache/true_spectrum.npz',  # Load bin edges from precomputed cache
        'lambda_k_mode': 'k_squared',  # λ_k = k² (static for global mode)
        'use_log': False,  # Standard energy (not log10)
        'use_output_norm': True,  # Per-batch output normalization
        'use_minmax_norm': True,  # Per-sample min-max normalization
        'loss_type': 'mspe'  # Mean Squared Percentage Error
    },
    description='MSE + SA-BSP (global): 2 trainable weights [w_mse=1.0, w_bsp=1.0] with normalization - competitive dynamics'
)

SA_BSP_COMBINED_CONFIG = LossConfig(
    loss_type='combined',
    loss_params={
        'base_loss': 'field_error',
        'spectral_loss': 'sa_bsp',
        'n_bins': 32,
        'adapt_mode': 'combined',  # 34 weights: w_mse + w_bsp + 32 per-bin
        'init_weight': 1.0,
        'epsilon': 1e-6,  # Increased from 1e-8 per paper ablation (Table 2)
        'binning_mode': 'linear',
        'signal_length': 4000,  # CDON temporal resolution
        'cache_path': 'cache/true_spectrum.npz',  # Load bin edges from precomputed cache
        'lambda_k_mode': 'k_squared',  # λ_k = k² initialization for per-bin weights
        'use_log': False,  # Standard energy (not log10)
        'use_output_norm': True,  # Per-batch output normalization
        'use_minmax_norm': True,  # Per-sample min-max normalization
        'loss_type': 'mspe'  # Mean Squared Percentage Error
    },
    description='MSE + SA-BSP (combined): 34 trainable weights [w_mse=1.0, w_bsp=1.0, 32×λ_k=k²] with normalization'
)

LOG_BSP_CONFIG = LossConfig(
    loss_type='combined',
    loss_params={
        'base_loss': 'field_error',
        'spectral_loss': 'bsp',
        'mu': 1.0,  # μ = 1.0 for log variant
        'n_bins': 32,
        'epsilon': 1e-8,
        'binning_mode': 'linear',
        'signal_length': 4000,  # CDON temporal resolution
        'cache_path': 'cache/true_spectrum.npz',
        'lambda_k_mode': 'uniform',  # λ_k = 1 for all bins (log variant)
        'use_log': True,  # Log10 transform of energies
        'use_output_norm': True,  # Per-batch output normalization: y = (y - mean) / std
        'use_minmax_norm': True,  # Per-sample min-max normalization: (E - min) / (max - min + eps)
        'loss_type': 'l2_norm'  # L2 norm loss (matches reference screenshot implementation)
    },
    description='MSE + Log-BSP: log₁₀(E) with per-sample min-max norm + L2 loss - Reference implementation'
)

# Legacy alias for backward compatibility
SA_BSP_CONFIG = SA_BSP_PERBIN_CONFIG
