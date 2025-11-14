"""
DeepONet architecture for 1D temporal operator learning.

Implements branch-trunk decomposition for learning operators mapping
input functions (earthquake accelerations) to output functions (structural displacements).

Causality is enforced through DATA PREPROCESSING (zero-padding), not architectural constraints.
This matches the reference CausalityDeepONet implementation from the CDON dataset paper.

Reference:
- Lu et al. "Learning nonlinear operators via DeepONet" (2021)
- Penwarden et al. "A metalearning approach for physics-informed neural networks (PINNs)" (2023)
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from siren_pytorch import SirenNet
    SIREN_AVAILABLE = True
except ImportError:
    SIREN_AVAILABLE = False


class ReQU(nn.Module):
    """
    ReQU activation: ReLU squared (ReLU(x)²).

    Used in reference CausalityDeepONet implementation.
    More smooth than ReLU, helps with gradient flow.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers and activation functions.

    Supports:
    - 'requ': ReQU (ReLU²) activation (reference implementation default)
    - 'siren': Sinusoidal activation (requires siren-pytorch)
    - 'tanh': Tanh activation (stable for operator learning)
    - 'relu': ReLU activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        activation: str = 'requ'
    ):
        """
        Initialize MLP.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function type ('requ', 'tanh', 'relu', 'siren')

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
        if self.activation_type == 'requ':
            return ReQU
        elif self.activation_type == 'tanh':
            return nn.Tanh
        elif self.activation_type == 'relu':
            return nn.ReLU
        else:
            raise ValueError(
                f"Unsupported activation: '{self.activation_type}'. "
                f"Supported: 'requ', 'tanh', 'relu', 'siren'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.network(x)


class DeepONet1D(nn.Module):
    """
    DeepONet for 1D temporal operator learning with per-timestep prediction.

    Implements the reference CausalityDeepONet architecture:
        - Branch network: Encodes windowed input function (4000 timesteps) → latent vector
        - Trunk network: Encodes time coordinate (scalar) → latent vector
        - Combination: Element-wise product + sum → scalar output

    Causality:
        Enforced through DATA PREPROCESSING (per-timestep windowing with zero-padding).
        At timestep t, branch receives only inputs from times [0, ..., t].

    Forward signature: forward(input, time_coord) → output
        - input: [batch, sensor_dim] - windowed input with zero-padding
        - time_coord: [batch, 1] - normalized time in [0, 1]
        - output: [batch, 1] - scalar prediction

    References:
        - Lu et al. "Learning nonlinear operators via DeepONet" (2021)
        - Penwarden et al. "A metalearning approach for physics-informed neural networks" (2023)
        - Reference implementation: Custom_dataset.py (reshapeTraining)
    """

    def __init__(
        self,
        sensor_dim: int = 4000,
        latent_dim: int = 100,
        branch_layers: List[int] = None,
        trunk_layers: List[int] = None,
        activation: str = 'requ'
    ):
        """
        Initialize DeepONet1D.

        Args:
            sensor_dim: Number of input timesteps (default 4000 for CDON)
            latent_dim: Dimension of latent space (default 100)
            branch_layers: Hidden layer sizes for branch network (default [120, 120])
                          Matches reference implementation
            trunk_layers: Hidden layer sizes for trunk network (default [120, 120])
                         Matches reference implementation
            activation: Activation function type (default 'requ' - ReLU²)
                       Options: 'requ', 'tanh', 'relu', 'siren'
                       Reference uses 'requ' (ReLU squared)
        """
        super().__init__()

        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim
        self.branch_layers = branch_layers or [120, 120]  # Reference default
        self.trunk_layers = trunk_layers or [120, 120]    # Reference default
        self.activation = activation

        # Branch network: Standard MLP processes input function
        # Input: [batch, sensor_dim] → Output: [batch, latent_dim]
        # Causality enforced through zero-padded input data
        self.branch = MLP(
            in_features=sensor_dim,
            out_features=latent_dim,
            hidden_layers=self.branch_layers,
            activation=activation
        )

        # Trunk network: processes query coordinates (time)
        # Input: [batch, 1] → Output: [batch, latent_dim]
        self.trunk = MLP(
            in_features=1,  # Single coordinate (time)
            out_features=latent_dim,
            hidden_layers=self.trunk_layers,
            activation=activation
        )

    def forward_per_timestep(self, x: torch.Tensor, time_coord: torch.Tensor) -> torch.Tensor:
        """
        Per-timestep forward pass for MSE loss (DeepONet reference mode).

        Args:
            x: Windowed input tensor of shape [batch, sensor_dim]
               Each sample is a zero-padded causal window
            time_coord: Time coordinates of shape [batch, 1] or [batch]
                       Normalized time values in [0, 1]

        Returns:
            Output tensor of shape [batch, 1] - scalar prediction per sample

        Example:
            >>> model = DeepONet1D(sensor_dim=4000, latent_dim=100)
            >>> x = torch.randn(16, 4000)  # 16 windowed samples
            >>> t = torch.rand(16, 1)      # 16 time coordinates
            >>> y = model.forward_per_timestep(x, t)  # Output: [16, 1]
        """
        # Ensure time_coord has shape [batch, 1]
        if time_coord.ndim == 1:
            time_coord = time_coord.unsqueeze(-1)  # [batch] → [batch, 1]

        # Branch network: encode windowed input function
        # [batch, sensor_dim] → [batch, latent_dim]
        branch_output = self.branch(x)

        # Trunk network: encode time coordinate
        # [batch, 1] → [batch, latent_dim]
        trunk_output = self.trunk(time_coord)

        # Combine branch and trunk via element-wise product and sum
        # [batch, latent_dim] * [batch, latent_dim] → [batch, latent_dim]
        combined = branch_output * trunk_output

        # Sum over latent dimension: [batch, latent_dim] → [batch]
        output = combined.sum(dim=-1)

        # Add output dimension: [batch] → [batch, 1]
        output = output.unsqueeze(-1)

        return output

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence forward pass for BSP loss (no causality constraint).

        Args:
            x: Full sequence input tensor of shape [batch, 1, sensor_dim]
               Raw signals without zero-padding

        Returns:
            Output tensor of shape [batch, 1, sensor_dim] - full sequence predictions

        Note:
            This method predicts ALL timesteps in one pass using all time coordinates.
            Used for BSP spectral loss computation.

        Example:
            >>> model = DeepONet1D(sensor_dim=4000, latent_dim=100)
            >>> x = torch.randn(4, 1, 4000)  # 4 full sequence samples
            >>> y = model.forward_sequence(x)  # Output: [4, 1, 4000]
        """
        batch_size = x.shape[0]

        # Flatten input: [batch, 1, sensor_dim] → [batch, sensor_dim]
        x_flat = x.squeeze(1)  # [batch, sensor_dim]

        # Branch network: encode input function
        # [batch, sensor_dim] → [batch, latent_dim]
        branch_output = self.branch(x_flat)

        # Create time grid [0, 1] for all timesteps
        time_grid = torch.linspace(0, 1, self.sensor_dim, device=x.device)  # [sensor_dim]

        # Expand for batch: [sensor_dim] → [batch, sensor_dim, 1]
        time_batch = time_grid.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)  # [batch, sensor_dim, 1]

        # Reshape for trunk: [batch, sensor_dim, 1] → [batch * sensor_dim, 1]
        time_flat = time_batch.reshape(-1, 1)

        # Trunk network: process all time coordinates
        # [batch * sensor_dim, 1] → [batch * sensor_dim, latent_dim]
        trunk_output_flat = self.trunk(time_flat)

        # Reshape back: [batch * sensor_dim, latent_dim] → [batch, sensor_dim, latent_dim]
        trunk_output = trunk_output_flat.reshape(batch_size, self.sensor_dim, self.latent_dim)

        # Expand branch output for broadcasting: [batch, latent_dim] → [batch, 1, latent_dim]
        branch_expanded = branch_output.unsqueeze(1)  # [batch, 1, latent_dim]

        # Element-wise product with broadcasting
        # [batch, 1, latent_dim] * [batch, sensor_dim, latent_dim] → [batch, sensor_dim, latent_dim]
        combined = branch_expanded * trunk_output

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
