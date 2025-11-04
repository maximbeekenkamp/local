"""
Fourier Neural Operator (FNO) for 1D temporal data.

Wrapper around neuralop's FNO for learning temporal operators
in the frequency domain (earthquake accelerations → structural displacements).
"""

import torch
import torch.nn as nn

try:
    from neuralop.models import FNO
    NEURALOP_AVAILABLE = True
except ImportError:
    NEURALOP_AVAILABLE = False
    FNO = None


class FNO1D(nn.Module):
    """
    Fourier Neural Operator for 1D temporal operator learning.

    Architecture:
        - Lifting layer: Project input to hidden channels
        - Spectral convolution layers: Process in Fourier domain
        - Projection layer: Project back to output channels

    Operations:
        1. FFT: Transform to frequency domain
        2. Spectral convolution: Learn low-frequency modes
        3. IFFT: Transform back to time domain
        4. Skip connection: Add to original

    Input shape: [batch, 1, 4000]
    Output shape: [batch, 1, 4000]

    Reference: Li et al. "Fourier Neural Operator for Parametric Partial
               Differential Equations" (2021)
    """

    def __init__(
        self,
        n_modes: int = 28,
        hidden_channels: int = 60,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1
    ):
        """
        Initialize FNO1D.

        Args:
            n_modes: Number of Fourier modes to keep (low-frequency, default 28)
            hidden_channels: Hidden channel dimension (default 52)
            n_layers: Number of FNO layers (default 4)
            in_channels: Number of input channels (default 1)
            out_channels: Number of output channels (default 1)

        Raises:
            ImportError: If neuralop is not installed
        """
        super().__init__()

        if not NEURALOP_AVAILABLE:
            raise ImportError(
                "neuralop is required for FNO1D. "
                "Install with: pip install -U neuraloperator"
            )

        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create FNO model from neuralop
        # For 1D, n_modes is a 1-tuple: (n_modes,)
        # The dimensionality is inferred from len(n_modes)
        self.fno = FNO(
            n_modes=(n_modes,),  # 1D: single element tuple
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            use_channel_mlp=True,  # Use channel mixing MLP
            channel_mlp_dropout=0.0,  # No dropout
            channel_mlp_expansion=0.5,  # MLP expansion factor
            non_linearity=torch.nn.functional.gelu  # Activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FNO.

        The neuralop FNO handles:
        - Real FFT (rfft) for efficiency
        - Mode truncation (keeping only n_modes low frequencies)
        - Spectral convolution with learnable weights
        - Inverse FFT (irfft) back to time domain

        Args:
            x: Input tensor of shape [batch, 1, timesteps]

        Returns:
            Output tensor of shape [batch, 1, timesteps]
        """
        return self.fno(x)

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_modes(self) -> int:
        """
        Get number of Fourier modes being used.

        Returns:
            Number of modes
        """
        return self.n_modes

    def get_hidden_channels(self) -> int:
        """
        Get hidden channel dimension.

        Returns:
            Hidden channel count
        """
        return self.hidden_channels
