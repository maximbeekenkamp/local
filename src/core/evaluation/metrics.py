"""
Metrics and loss functions for neural operator training.

Provides:
- compute_mse: Standard MSE metric (matches reference CausalityDeepONet)
- compute_spectrum_error_1d: Relative MSE in frequency domain (log power spectrum)

Reference:
- CausalityDeepONet uses standard MSE (torch.nn.MSELoss)
- Spectrum error adapted from Generatively-Stabilised-NOs for 1D temporal data
"""

import torch
import torch.nn as nn
from typing import Optional


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Mean Squared Error (standard MSE).

    This is the standard MSE metric used in neural operator training,
    matching the reference CausalityDeepONet implementation.

    Formula:
        mse = mean((pred - target)²)

    Args:
        pred: Predicted output [batch, channels, timesteps] or [batch]
        target: Ground truth (same shape as pred)
        reduction: 'mean' or 'none' (default 'mean')

    Returns:
        MSE (scalar if reduction='mean', [batch] if reduction='none')

    Shape:
        Input: [B, C, T] or [B]
        Output: scalar (reduction='mean') or [B] (reduction='none')
    """
    # Squared error: (pred - target)²
    squared_error = (pred - target) ** 2

    if reduction == 'mean':
        return squared_error.mean()
    elif reduction == 'none':
        # Mean over all dimensions except batch
        if squared_error.ndim > 1:
            return squared_error.mean(dim=tuple(range(1, squared_error.ndim)))
        else:
            return squared_error
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
    - MSE (real space)
    - Spectrum error (frequency space)

    Args:
        pred: Predicted output [batch, channels, timesteps]
        target: Ground truth [batch, channels, timesteps]
        epsilon: Small constant for numerical stability (default 1e-8)

    Returns:
        Dictionary with keys:
        - 'mse': float
        - 'spectrum_error': float

    Example:
        >>> metrics = compute_all_metrics(pred, target)
        >>> print(f"MSE: {metrics['mse']:.4f}")
        >>> print(f"Spectrum error: {metrics['spectrum_error']:.4f}")
    """
    return {
        'mse': compute_mse(pred, target).item(),
        'spectrum_error': compute_spectrum_error_1d(pred, target, epsilon).item()
    }
