"""
Binned Spectral Power (BSP) Loss for 1D temporal neural operators.

Adapted from 2D spatial implementation to 1D temporal signals.
Mitigates spectral bias by comparing frequency distributions in binned space.

Key adaptations from 2D spatial to 1D temporal:
- FFT: torch.fft.fftn(dim=(-2,-1)) → torch.fft.rfft(dim=-1)
- Shape: [B, C, H, W] → [B, C, T] where T=4000
- Frequencies: 2D wavenumber grid → 1D frequency array
- Binning: Radial binning (2D) → Linear binning (1D)

Reference:
- Paper: "Mitigating Spectral Bias for Physics-Informed Neural Networks via
  Binned Spectral Power Loss"
- Original implementation: Generatively-Stabilised-NOs binned_spectral_loss.py (2D)

Formula:
    L_BSP = Σ_b w_b × mean_c [((E_pred[b,c] - E_true[b,c]) / (E_true[b,c] + ε))²]

Where:
- b: frequency bin index (1 to n_bins)
- c: channel index
- E[b,c]: Bin-averaged energy in bin b, channel c
- w_b: Optional bin weight (default: uniform)
- ε: Numerical stability constant

Usage:
    Always combine with base loss (never use BSP alone):
    L_total = L_MSE + λ × L_BSP

Example:
    >>> loss_fn = BinnedSpectralLoss(n_bins=32, lambda_bsp=1.0)
    >>> prediction = model(input)  # [16, 1, 4000]
    >>> loss = loss_fn(prediction, ground_truth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BinnedSpectralLoss(nn.Module):
    """
    Binned Spectral Power Loss for 1D temporal signals.

    Computes Mean Squared Percentage Error on bin-averaged Fourier energies.
    Encourages models to learn correct frequency distribution, mitigating
    spectral bias toward low frequencies.

    Attributes:
        n_bins: Number of frequency bins (default: 32)
        lambda_bsp: Weight for BSP loss component (default: 1.0)
        epsilon: Numerical stability constant (default: 1e-8)
        binning_mode: 'linear' or 'log' frequency spacing (default: 'linear')
    """

    def __init__(
        self,
        n_bins: int = 32,
        lambda_bsp: float = 1.0,
        epsilon: float = 1e-8,
        binning_mode: str = 'linear'
    ):
        """
        Initialize Binned Spectral Loss.

        Args:
            n_bins: Number of frequency bins (default: 32, matching paper)
            lambda_bsp: Weight for BSP loss (default: 1.0)
            epsilon: Numerical stability constant (default: 1e-8)
            binning_mode: Frequency spacing ('linear' or 'log', default: 'linear')

        Raises:
            ValueError: If binning_mode is not 'linear' or 'log'
        """
        super().__init__()
        self.n_bins = n_bins
        self.lambda_bsp = lambda_bsp
        self.epsilon = epsilon

        if binning_mode not in ['linear', 'log']:
            raise ValueError(f"binning_mode must be 'linear' or 'log', got {binning_mode}")
        self.binning_mode = binning_mode

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binned Spectral Power loss.

        Args:
            pred: Predicted output [B, C, T] where T=4000
            target: Ground truth [B, C, T]

        Returns:
            Scalar BSP loss (weighted by lambda_bsp)

        Shape:
            Input: [B, C, T]
            Output: scalar

        Algorithm:
            1. Compute 1D FFT along time dimension
            2. Compute power spectrum: |FFT|²
            3. Bin-average energies into n_bins frequency bands
            4. Compute mean squared percentage error per bin
            5. Average across bins and channels
        """
        # Step 1: Compute 1D real FFT
        # Shape: [B, C, T] → [B, C, T//2+1] complex
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        # Step 2: Compute power spectrum
        # Shape: [B, C, T//2+1] complex → [B, C, T//2+1] real
        pred_energy = torch.abs(pred_fft) ** 2
        target_energy = torch.abs(target_fft) ** 2

        # Step 3: Bin-average energies
        # Shape: [B, C, T//2+1] → [B, C, n_bins]
        T = pred.shape[-1]
        pred_binned = self._bin_energy_1d(pred_energy, T)
        target_binned = self._bin_energy_1d(target_energy, T)

        # Step 4: Mean Squared Percentage Error per bin
        # Relative error per bin: (pred - true) / (true + ε)
        relative_error = (pred_binned - target_binned) / (target_binned + self.epsilon)

        # Squared error
        squared_error = relative_error ** 2

        # Step 5: Average over bins and channels, mean over batch
        # Shape: [B, C, n_bins] → [B] → scalar
        bsp_loss = squared_error.mean(dim=(-2, -1)).mean()

        # Apply weight
        return self.lambda_bsp * bsp_loss

    def _bin_energy_1d(
        self,
        energy: torch.Tensor,
        T: int
    ) -> torch.Tensor:
        """
        Bin-average energy spectrum into frequency bands (1D).

        Uses differentiable soft binning for gradient flow.

        Args:
            energy: Power spectrum [B, C, T//2+1]
            T: Original time dimension (for frequency calculation)

        Returns:
            Binned energy [B, C, n_bins]

        Algorithm:
            1. Compute frequency values for each FFT bin
            2. Create bin edges based on binning_mode
            3. Soft-assign each frequency to bins using triangular weights
            4. Average energy within each bin (weighted mean)
        """
        B, C, n_freqs = energy.shape
        device = energy.device
        dtype = energy.dtype

        # Step 1: Get frequency values
        # torch.fft.rfftfreq returns normalized frequencies [0, 0.5]
        # Shape: [n_freqs] = [T//2+1]
        frequencies = torch.fft.rfftfreq(T, device=device, dtype=dtype)

        # Step 2: Create bin edges
        if self.binning_mode == 'linear':
            # Linear spacing from 0 to max frequency (0.5 for Nyquist)
            # Skip DC component (frequency 0) to avoid singularities
            freq_min = frequencies[1]  # First non-DC frequency
            freq_max = frequencies[-1]  # Nyquist frequency
            bin_edges = torch.linspace(
                freq_min, freq_max, self.n_bins + 1,
                device=device, dtype=dtype
            )
        else:  # 'log'
            # Logarithmic spacing (better for wide frequency ranges)
            freq_min = frequencies[1]
            freq_max = frequencies[-1]
            bin_edges = torch.logspace(
                torch.log10(freq_min), torch.log10(freq_max),
                self.n_bins + 1,
                device=device, dtype=dtype
            )

        # Step 3: Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # [n_bins]

        # Step 4: Soft binning with triangular weights
        # For each frequency, compute soft assignment to neighboring bins
        # Shape: [n_freqs, 1] vs [1, n_bins] → [n_freqs, n_bins]
        freq_expanded = frequencies[1:].unsqueeze(1)  # Skip DC, [n_freqs-1, 1]
        centers_expanded = bin_centers.unsqueeze(0)  # [1, n_bins]

        # Compute distance to bin centers
        distances = torch.abs(freq_expanded - centers_expanded)  # [n_freqs-1, n_bins]

        # Triangular weight: 1 - (distance / bin_width)
        # Clip to [0, 1] so only nearby bins contribute
        bin_width = bin_edges[1] - bin_edges[0]  # Assume uniform for simplicity
        weights = torch.clamp(1.0 - distances / bin_width, min=0.0)  # [n_freqs-1, n_bins]

        # Normalize weights so they sum to 1 per frequency
        # Shape: [n_freqs-1, n_bins]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)

        # Step 5: Weighted average energy per bin
        # energy: [B, C, n_freqs]
        # weights: [n_freqs-1, n_bins]
        # We skip DC component (index 0) in energy
        energy_no_dc = energy[:, :, 1:]  # [B, C, n_freqs-1]

        # Matrix multiply: [B, C, n_freqs-1] @ [n_freqs-1, n_bins] → [B, C, n_bins]
        # Use einsum for clarity
        # 'b' = batch, 'c' = channels, 'f' = frequencies, 'n' = n_bins
        binned_energy = torch.einsum('bcf,fn->bcn', energy_no_dc, weights)

        return binned_energy

    def compute_bin_errors(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-bin relative errors (for analysis and SA-BSP).

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Per-bin squared relative errors [n_bins]
            (averaged over batch and channels)

        This is useful for:
        - Analyzing which frequency bands have highest error
        - Self-adaptive weighting in SA-BSP loss
        """
        # Compute FFT and power spectrum
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        pred_energy = torch.abs(pred_fft) ** 2
        target_energy = torch.abs(target_fft) ** 2

        # Bin energies
        T = pred.shape[-1]
        pred_binned = self._bin_energy_1d(pred_energy, T)
        target_binned = self._bin_energy_1d(target_energy, T)

        # Relative error per bin
        relative_error = (pred_binned - target_binned) / (target_binned + self.epsilon)
        squared_error = relative_error ** 2

        # Average over batch and channels: [B, C, n_bins] → [n_bins]
        bin_errors = squared_error.mean(dim=(0, 1))

        return bin_errors

    def get_frequency_analysis(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Detailed frequency analysis for debugging and visualization.

        Args:
            pred: Predicted output [B, C, T]
            target: Ground truth [B, C, T]

        Returns:
            Dictionary with:
            - 'bin_edges': Frequency bin edges [n_bins+1]
            - 'bin_centers': Frequency bin centers [n_bins]
            - 'pred_spectrum': Binned prediction spectrum [n_bins]
            - 'target_spectrum': Binned target spectrum [n_bins]
            - 'relative_errors': Relative error per bin [n_bins]
            - 'loss': Total BSP loss (scalar)
        """
        with torch.no_grad():
            # Compute FFT
            pred_fft = torch.fft.rfft(pred, dim=-1)
            target_fft = torch.fft.rfft(target, dim=-1)
            pred_energy = torch.abs(pred_fft) ** 2
            target_energy = torch.abs(target_fft) ** 2

            # Bin energies
            T = pred.shape[-1]
            pred_binned = self._bin_energy_1d(pred_energy, T)
            target_binned = self._bin_energy_1d(target_energy, T)

            # Average over batch and channels
            pred_spectrum = pred_binned.mean(dim=(0, 1))  # [n_bins]
            target_spectrum = target_binned.mean(dim=(0, 1))  # [n_bins]

            # Relative errors
            relative_errors = (pred_spectrum - target_spectrum) / (target_spectrum + self.epsilon)

            # Bin edges and centers
            frequencies = torch.fft.rfftfreq(T, device=pred.device, dtype=pred.dtype)
            if self.binning_mode == 'linear':
                freq_min = frequencies[1]
                freq_max = frequencies[-1]
                bin_edges = torch.linspace(freq_min, freq_max, self.n_bins + 1)
            else:
                freq_min = frequencies[1]
                freq_max = frequencies[-1]
                bin_edges = torch.logspace(
                    torch.log10(freq_min), torch.log10(freq_max),
                    self.n_bins + 1
                )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute loss with gradients
        loss = self.forward(pred, target)

        return {
            'bin_edges': bin_edges.cpu().numpy(),
            'bin_centers': bin_centers.cpu().numpy(),
            'pred_spectrum': pred_spectrum.cpu().numpy(),
            'target_spectrum': target_spectrum.cpu().numpy(),
            'relative_errors': relative_errors.cpu().numpy(),
            'loss': loss.item()
        }
