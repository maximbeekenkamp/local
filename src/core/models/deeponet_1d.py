"""
DeepONet architecture for 1D temporal operator learning.

Implements branch-trunk decomposition for learning operators mapping
input functions (earthquake accelerations) to output functions (structural displacements).
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from siren_pytorch import SirenNet
    SIREN_AVAILABLE = True
except ImportError:
    SIREN_AVAILABLE = False


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers and activation functions.

    Supports:
    - 'siren': Sinusoidal activation (default, requires siren-pytorch)
    - 'tanh': Tanh activation (stable for operator learning)
    - 'relu': ReLU activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        activation: str = 'tanh'
    ):
        """
        Initialize MLP.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function type ('tanh', 'relu', 'siren')

        Raises:
            ValueError: If activation type is not supported
            ImportError: If siren activation is requested but siren-pytorch not installed
        """
        super().__init__()

        self.activation_type = activation.lower()

        # For SIREN, use the SirenNet module from siren-pytorch
        if self.activation_type == 'siren':
            if not SIREN_AVAILABLE:
                raise ImportError(
                    "siren-pytorch is required for SIREN activation. "
                    "Install with: pip install siren-pytorch"
                )
            # SirenNet handles its own layers and activations with proper initialization
            self.network = SirenNet(
                dim_in=in_features,
                dim_hidden=hidden_layers[0] if hidden_layers else 256,
                dim_out=out_features,
                num_layers=len(hidden_layers) + 1,  # hidden layers + output layer
                final_activation=nn.Identity(),  # No activation on final layer
                w0_initial=10.0  # Reduced from 30 for stability with spectral losses
            )
        else:
            # Standard MLP with specified activation
            layers = []
            prev_size = in_features

            # Get activation module
            activation_fn = self._get_activation()

            # Hidden layers with activation
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(activation_fn())
                prev_size = hidden_size

            # Output layer (no activation)
            layers.append(nn.Linear(prev_size, out_features))

            self.network = nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        """
        Get activation module based on activation type.

        Returns:
            Activation module class

        Raises:
            ValueError: If activation type is not supported
        """
        if self.activation_type == 'tanh':
            return nn.Tanh
        elif self.activation_type == 'relu':
            return nn.ReLU
        else:
            raise ValueError(
                f"Unsupported activation: '{self.activation_type}'. "
                f"Supported: 'tanh', 'relu', 'siren'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.network(x)


class DeepONet1D(nn.Module):
    """
    DeepONet for 1D temporal operator learning.

    Architecture:
        - Branch network: Encodes input function u(t) into latent representation
        - Trunk network: Learns basis functions at query points
        - Combination: Element-wise product + sum to predict output

    Input shape: [batch, 1, timesteps]
    Output shape: [batch, 1, timesteps]

    Reference: Lu et al. "Learning nonlinear operators via DeepONet" (2021)
    """

    def __init__(
        self,
        sensor_dim: int = 4000,
        latent_dim: int = 100,
        branch_layers: List[int] = None,
        trunk_layers: List[int] = None,
        activation: str = 'siren'
    ):
        """
        Initialize DeepONet1D.

        Args:
            sensor_dim: Number of input timesteps (default 4000 for CDON)
            latent_dim: Dimension of latent space (default 100)
            branch_layers: Hidden layer sizes for branch network (default [50, 100])
            trunk_layers: Hidden layer sizes for trunk network (default [100, 100])
            activation: Activation function type ('tanh', 'relu', 'siren', default 'siren')
        """
        super().__init__()

        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim
        self.branch_layers = branch_layers or [50, 100]
        self.trunk_layers = trunk_layers or [100, 100]
        self.activation = activation

        # Branch network: processes input function
        # Input: [batch, sensor_dim] → Output: [batch, latent_dim]
        self.branch = MLP(
            in_features=sensor_dim,
            out_features=latent_dim,
            hidden_layers=self.branch_layers,
            activation=activation
        )

        # Trunk network: processes query coordinates
        # Input: [batch, timesteps, 1] → Output: [batch, timesteps, latent_dim]
        self.trunk = MLP(
            in_features=1,  # Single coordinate (time)
            out_features=latent_dim,
            hidden_layers=self.trunk_layers,
            activation=activation
        )

        # Query points: uniform grid [0, 1]
        # Generated once and reused for all forward passes
        self.register_buffer(
            'query_points',
            torch.linspace(0, 1, sensor_dim).unsqueeze(-1)  # [sensor_dim, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet.

        Args:
            x: Input tensor of shape [batch, 1, timesteps]

        Returns:
            Output tensor of shape [batch, 1, timesteps]
        """
        batch_size = x.shape[0]

        # Flatten input: [batch, 1, timesteps] → [batch, timesteps]
        x_flat = x.squeeze(1)  # [batch, sensor_dim]

        # Branch network: encode input function
        # [batch, sensor_dim] → [batch, latent_dim]
        branch_output = self.branch(x_flat)

        # Trunk network: process query points
        # Expand query points for batch: [sensor_dim, 1] → [batch, sensor_dim, 1]
        query_batch = self.query_points.unsqueeze(0).expand(batch_size, -1, -1)

        # [batch, sensor_dim, 1] → [batch, sensor_dim, latent_dim]
        trunk_output = self.trunk(query_batch)

        # Combine branch and trunk via element-wise product
        # Branch: [batch, latent_dim] → [batch, 1, latent_dim]
        # Trunk: [batch, sensor_dim, latent_dim]
        branch_expanded = branch_output.unsqueeze(1)  # [batch, 1, latent_dim]

        # Element-wise multiply and sum over latent dimension
        # [batch, 1, latent_dim] * [batch, sensor_dim, latent_dim] → [batch, sensor_dim, latent_dim]
        combined = branch_expanded * trunk_output  # Broadcasting

        # Sum over latent dimension: [batch, sensor_dim, latent_dim] → [batch, sensor_dim]
        output = combined.sum(dim=-1)

        # Add channel dimension: [batch, sensor_dim] → [batch, 1, sensor_dim]
        output = output.unsqueeze(1)

        return output

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
