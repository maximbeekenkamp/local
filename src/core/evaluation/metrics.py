"""
Metrics and loss functions for neural operator training.

Provides:
- RelativeL2Loss: Scale-invariant loss function
- compute_field_error: Relative MSE in real space
- compute_spectrum_error_1d: Relative MSE in frequency domain (log power spectrum)

Reference:
- Field error from Generatively-Stabilised-NOs spectral_metrics.py
- Adapted spectrum error from 2D spatial to 1D temporal data
"""

import torch
import torch.nn as nn
from typing import Optional


class RelativeL2Loss(nn.Module):
    """
    Relative L2 loss function (scale-invariant).

    Formula:
        loss = mean(||pred - target||_2 / (||target||_2 + epsilon))

    Why relative L2 instead of MSE:
    - Scale-invariant: Works across different magnitudes
    - CDON has input std=0.014, output std=0.038 (2.7× difference)
    - Matches reference code philosophy for operator learning
    - Easier to compare results across different architectures

    Reference: NeuralOperator_DiffusionModel CustomLoss (train_no.py:34-42)
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize Relative L2 Loss.

        Args:
            epsilon: Small constant for numerical stability (default 1e-8)
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute relative L2 loss.

        Args:
            pred: Predicted output [batch, channels, timesteps]
            target: Ground truth [batch, channels, timesteps]

        Returns:
            Scalar loss value (mean over batch)

        Shape:
            Input: [B, C, T] where B=batch, C=channels, T=timesteps
            Output: scalar tensor
        """
        # Compute L2 norm over spatial dimensions (channels + timesteps)
        # Shape: [B, C, T] → reduce over (C, T) → [B]
        diff_norm = torch.norm(
            pred - target,
            p=2,
            dim=(-2, -1)  # Reduce over channel and time dimensions
        )

        target_norm = torch.norm(
            target,
            p=2,
            dim=(-2, -1)
        )

        # Relative error per sample
        relative_error = diff_norm / (target_norm + self.epsilon)

        # Mean over batch
        return relative_error.mean()


def compute_field_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute field error (relative MSE in real space).

    Field error measures prediction accuracy in the original (real) space,
    as opposed to frequency space. It's the primary evaluation metric.

    Formula:
        field_error = mean((pred - target)² / (target² + epsilon))

    Reference: Generatively-Stabilised-NOs spectral_metrics.py
               compute_field_error_loss()

    Args:
        pred: Predicted output [batch, channels, timesteps]
        target: Ground truth [batch, channels, timesteps]
        epsilon: Small constant for numerical stability (default 1e-8)
        reduction: 'mean' or 'none' (default 'mean')

    Returns:
        Field error (scalar if reduction='mean', [batch] if reduction='none')

    Shape:
        Input: [B, C, T]
        Output: scalar (reduction='mean') or [B] (reduction='none')
    """
    # Squared error: (pred - target)²
    squared_error = (pred - target) ** 2

    # Mean squared error over spatial dimensions (C, T)
    spatial_mse = squared_error.mean(dim=(-2, -1))  # [B]

    # Mean squared target over spatial dimensions
    spatial_mean_sq = (target ** 2).mean(dim=(-2, -1))  # [B]

    # Relative error per sample
    relative_error = spatial_mse / (spatial_mean_sq + epsilon)  # [B]

    if reduction == 'mean':
        return relative_error.mean()
    elif reduction == 'none':
        return relative_error
    else:
        raise ValueError(f"Unknown reduction: '{reduction}'. Use 'mean' or 'none'")


def compute_spectrum_error_1d(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute energy spectrum error (relative MSE in log power spectrum).

    Spectrum error measures prediction accuracy in frequency domain,
    useful for assessing how well the model captures different frequency
    components of the temporal signal.

    Process:
        1. Apply 1D FFT to temporal dimension
        2. Compute power spectrum: |FFT|²
        3. Take logarithm: log(power + epsilon)
        4. Compute relative MSE in log space

    Reference: Generatively-Stabilised-NOs spectral_metrics.py
               compute_spectrum_error_loss() (adapted from 2D to 1D)

    Args:
        pred: Predicted output [batch, channels, timesteps]
        target: Ground truth [batch, channels, timesteps]
        epsilon: Small constant for numerical stability (default 1e-8)
        reduction: 'mean' or 'none' (default 'mean')

    Returns:
        Spectrum error (scalar if reduction='mean', [batch] if reduction='none')

    Shape:
        Input: [B, C, T] where T=4000
        Output: scalar (reduction='mean') or [B] (reduction='none')

    Note:
        Uses rfft (real FFT) for efficiency since input is real-valued.
        Output frequencies: T//2 + 1 = 2001 for T=4000.
    """
    # Apply 1D real FFT along time dimension
    # Shape: [B, C, T] → [B, C, T//2 + 1] complex
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    # Compute power spectrum: |FFT|²
    # Shape: [B, C, T//2 + 1] complex → [B, C, T//2 + 1] real
    pred_power = torch.abs(pred_fft) ** 2
    target_power = torch.abs(target_fft) ** 2

    # Log power spectrum (add epsilon for stability)
    # Shape: [B, C, T//2 + 1]
    pred_log_spectrum = torch.log(pred_power + epsilon)
    target_log_spectrum = torch.log(target_power + epsilon)

    # Compute MSE in log spectrum space
    squared_error = (pred_log_spectrum - target_log_spectrum) ** 2

    # Mean over spatial dimensions (C, freq)
    spatial_mse = squared_error.mean(dim=(-2, -1))  # [B]

    # Mean squared target log spectrum
    spatial_mean_sq = (target_log_spectrum ** 2).mean(dim=(-2, -1))  # [B]

    # Relative error per sample
    relative_error = spatial_mse / (spatial_mean_sq + epsilon)  # [B]

    if reduction == 'mean':
        return relative_error.mean()
    elif reduction == 'none':
        return relative_error
    else:
        raise ValueError(f"Unknown reduction: '{reduction}'. Use 'mean' or 'none'")


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8
) -> dict:
    """
    Compute all evaluation metrics at once.

    Convenience function for validation/testing that computes:
    - Field error (real space)
    - Spectrum error (frequency space)

    Args:
        pred: Predicted output [batch, channels, timesteps]
        target: Ground truth [batch, channels, timesteps]
        epsilon: Small constant for numerical stability (default 1e-8)

    Returns:
        Dictionary with keys:
        - 'field_error': float
        - 'spectrum_error': float

    Example:
        >>> metrics = compute_all_metrics(pred, target)
        >>> print(f"Field error: {metrics['field_error']:.4f}")
        >>> print(f"Spectrum error: {metrics['spectrum_error']:.4f}")
    """
    return {
        'field_error': compute_field_error(pred, target, epsilon).item(),
        'spectrum_error': compute_spectrum_error_1d(pred, target, epsilon).item()
    }
