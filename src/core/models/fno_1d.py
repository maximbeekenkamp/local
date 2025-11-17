"""
Fourier Neural Operator (FNO) for 1D temporal data.

Wrapper around neuralop's FNO for learning temporal operators
in the frequency domain (earthquake accelerations → structural displacements).

This model operates on full sequences and is designed to work with spectral losses (BSP, SA-BSP).
For per-timestep causal training with MSE loss, use DeepONet with prepare_causal_deeponet_data().

Reference:
- Li et al. "Fourier Neural Operator for Parametric PDEs" (2021)
- Penwarden et al. "A metalearning approach for physics-informed neural networks" (2023)
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

    Usage:
        This model operates on full sequences [batch, 1, signal_length] and is designed
        for use with spectral losses (BSP, SA-BSP) that require complete sequences
        for FFT operations.

        For per-timestep causal training with MSE loss, use DeepONet architecture
        with prepare_causal_deeponet_data() preprocessing instead.

    Input shape: [batch, 1, signal_length]
    Output shape: [batch, 1, signal_length]

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
            hidden_channels: Hidden channel dimension (default 60)
            n_layers: Number of FNO layers (default 4)
            in_channels: Number of input channels (default 1)
            out_channels: Number of output channels (default 1)

        Raises:
            ImportError: If neuralop is not installed

        Note:
            This model is designed for sequence-to-sequence prediction with spectral
            losses. It operates on full sequences without causal preprocessing.
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

        Operations:
            - Global FFT over entire sequence
            - Spectral convolution with learned weights
            - Inverse FFT back to time domain

        Args:
            x: Input tensor of shape [batch, channels, timesteps]
               Full sequences without causal preprocessing

        Returns:
            Output tensor of shape [batch, channels, timesteps]
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
