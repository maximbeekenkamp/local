"""
DeepONet architecture for 1D temporal operator learning.

Implements branch-trunk decomposition for learning operators mapping
input functions (earthquake accelerations) to output functions (structural displacements).
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers.

    Uses Tanh activation for stability in operator learning.
    """

    def __init__(self, in_features: int, out_features: int, hidden_layers: List[int]):
        """
        Initialize MLP.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_layers: List of hidden layer sizes
        """
        super().__init__()

        layers = []
        prev_size = in_features

        # Hidden layers with Tanh activation
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, out_features))

        self.network = nn.Sequential(*layers)

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
        trunk_layers: List[int] = None
    ):
        """
        Initialize DeepONet1D.

        Args:
            sensor_dim: Number of input timesteps (default 4000 for CDON)
            latent_dim: Dimension of latent space (default 100)
            branch_layers: Hidden layer sizes for branch network (default [50, 100])
            trunk_layers: Hidden layer sizes for trunk network (default [100, 100])
        """
        super().__init__()

        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim
        self.branch_layers = branch_layers or [50, 100]
        self.trunk_layers = trunk_layers or [100, 100]

        # Branch network: processes input function
        # Input: [batch, sensor_dim] → Output: [batch, latent_dim]
        self.branch = MLP(
            in_features=sensor_dim,
            out_features=latent_dim,
            hidden_layers=self.branch_layers
        )

        # Trunk network: processes query coordinates
        # Input: [batch, timesteps, 1] → Output: [batch, timesteps, latent_dim]
        self.trunk = MLP(
            in_features=1,  # Single coordinate (time)
            out_features=latent_dim,
            hidden_layers=self.trunk_layers
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
