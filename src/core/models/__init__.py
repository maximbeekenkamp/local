"""
Neural operator models for CDON earthquake data.

Provides DeepONet, FNO, and UNet architectures adapted for 1D temporal
operator learning (acceleration → displacement mapping).
"""

from .deeponet_1d import DeepONet1D
from .fno_1d import FNO1D
from .unet_1d import UNet1D
from .model_factory import create_model, list_available_models, get_model_info

__all__ = [
    'DeepONet1D',
    'FNO1D',
    'UNet1D',
    'create_model',
    'list_available_models',
    'get_model_info',
]

__version__ = '0.1.0'
