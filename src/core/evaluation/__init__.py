"""Loss functions and evaluation metrics for neural operator training."""

from .loss_factory import create_loss
from .binned_spectral_loss import BinnedSpectralLoss
from .adaptive_spectral_loss import SelfAdaptiveBSPLoss

__all__ = [
    'create_loss',
    'BinnedSpectralLoss',
    'SelfAdaptiveBSPLoss',
]
