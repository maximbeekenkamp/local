"""
Penalty-weighted loss wrapper for inverse variance weighting.

Implements the penalty weighting approach from the CausalityDeepONet reference paper,
where loss is weighted inversely proportional to the maximum response magnitude squared.

Reference: Penwarden et al. "A metalearning approach for physics-informed neural networks" (2023)
           utils_DeepONet.py lines 50-55
"""

import torch
import torch.nn as nn
from typing import Optional


class PenaltyWeightedLoss(nn.Module):
    """
    Penalty-weighted loss wrapper.

    Wraps any loss function and applies inverse-variance weighting:
        weighted_loss = penalty * base_loss
        where penalty = 1 / (max(abs(target))**2 + epsilon)

    This emphasizes samples with larger responses, which are typically
    harder to predict accurately. From the reference implementation:

    ```python
    penalty = torch.ones((Responses.shape[0]+1,)) / torch.max(torch.abs(Responses))**2
    loss = torch.mean(penalty * torch.square(y_out - y))
    ```

    Args:
        base_loss: Base loss function (e.g., MSELoss, RelativeL2Loss, etc.)
        epsilon: Small constant for numerical stability (default 1e-8)
        per_sample: If True, compute penalty per sample in batch (default True)
                    If False, use single penalty for entire batch

    Example:
        >>> from src.core.evaluation.metrics import RelativeL2Loss
        >>> base_loss = RelativeL2Loss()
        >>> penalty_loss = PenaltyWeightedLoss(base_loss)
        >>>
        >>> pred = torch.randn(16, 1, 4000)
        >>> target = torch.randn(16, 1, 4000)
        >>> loss = penalty_loss(pred, target)  # Automatically applies penalty
    """

    def __init__(
        self,
        base_loss: nn.Module,
        epsilon: float = 1e-8,
        per_sample: bool = True
    ):
        """
        Initialize penalty-weighted loss.

        Args:
            base_loss: Base loss module to wrap
            epsilon: Numerical stability constant (default 1e-8)
            per_sample: Compute penalty per sample (True) or global (False)
        """
        super().__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.per_sample = per_sample

    def compute_penalty(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty weights from target values.

        Args:
            target: Target tensor [B, ...] where B is batch size

        Returns:
            penalty: Penalty weights [B] if per_sample=True, else scalar
        """
        if self.per_sample:
            # Compute penalty per sample in batch
            # Flatten spatial/temporal dimensions, keep batch dimension
            target_flat = target.reshape(target.shape[0], -1)  # [B, *]

            # Max absolute value per sample
            max_abs_per_sample = torch.max(torch.abs(target_flat), dim=1)[0]  # [B]

            # Penalty = 1 / max²
            penalty = 1.0 / (max_abs_per_sample ** 2 + self.epsilon)  # [B]
        else:
            # Global penalty (single value for entire batch)
            max_abs = torch.max(torch.abs(target))
            penalty = 1.0 / (max_abs ** 2 + self.epsilon)  # scalar

        return penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with penalty weighting.

        Args:
            pred: Predicted tensor [B, ...]
            target: Target tensor [B, ...]

        Returns:
            Weighted loss (scalar)
        """
        # Compute base loss (assume it reduces to scalar or per-sample)
        base_loss_value = self.base_loss(pred, target)

        # Compute penalty weights
        penalty = self.compute_penalty(target)

        # Apply penalty weighting
        if self.per_sample and base_loss_value.dim() > 0:
            # Base loss is per-sample [B]
            weighted_loss = penalty * base_loss_value
            # Average over batch
            final_loss = weighted_loss.mean()
        else:
            # Base loss is already scalar
            # Average penalty over batch if per_sample
            if self.per_sample:
                penalty = penalty.mean()
            final_loss = penalty * base_loss_value

        return final_loss

    def get_penalty_stats(self, target: torch.Tensor) -> dict:
        """
        Get statistics about penalty weights for logging/debugging.

        Args:
            target: Target tensor [B, ...]

        Returns:
            Dictionary with penalty statistics
        """
        penalty = self.compute_penalty(target)

        if self.per_sample:
            stats = {
                'penalty_mean': penalty.mean().item(),
                'penalty_std': penalty.std().item(),
                'penalty_min': penalty.min().item(),
                'penalty_max': penalty.max().item()
            }
        else:
            stats = {
                'penalty': penalty.item()
            }

        return stats


class MSEWithPenalty(nn.Module):
    """
    MSE loss with penalty weighting (exact reference implementation).

    This is the EXACT formula from the reference CausalityDeepONet paper:
        loss = mean(penalty * (pred - target)²)

    where penalty = 1 / max(abs(target))²

    This is provided as a standalone loss for exact reference matching.
    For general use with other losses, use PenaltyWeightedLoss wrapper.

    Args:
        epsilon: Numerical stability constant (default 1e-8)
        per_sample: Compute penalty per sample (True) or global (False)

    Example:
        >>> loss_fn = MSEWithPenalty()
        >>> pred = torch.randn(16, 1, 4000)
        >>> target = torch.randn(16, 1, 4000)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, epsilon: float = 1e-8, per_sample: bool = True):
        super().__init__()
        self.epsilon = epsilon
        self.per_sample = per_sample

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE with penalty weighting.

        Args:
            pred: Predicted tensor [B, ...]
            target: Target tensor [B, ...]

        Returns:
            Penalty-weighted MSE loss (scalar)
        """
        # Compute squared error
        squared_error = (pred - target) ** 2  # [B, ...]

        if self.per_sample:
            # Penalty per sample
            target_flat = target.reshape(target.shape[0], -1)
            max_abs_per_sample = torch.max(torch.abs(target_flat), dim=1)[0]
            penalty = 1.0 / (max_abs_per_sample ** 2 + self.epsilon)  # [B]

            # Reshape penalty for broadcasting
            penalty = penalty.view(-1, *([1] * (squared_error.dim() - 1)))  # [B, 1, ...]

            # Apply penalty and average
            weighted_error = penalty * squared_error
            loss = weighted_error.mean()
        else:
            # Global penalty
            max_abs = torch.max(torch.abs(target))
            penalty = 1.0 / (max_abs ** 2 + self.epsilon)

            loss = (penalty * squared_error).mean()

        return loss
