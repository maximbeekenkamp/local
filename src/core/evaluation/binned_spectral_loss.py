"""
Binned Spectral Power (BSP) Loss for 1D temporal neural operators.

Adapted from 2D spatial implementation to 1D temporal signals.
Mitigates spectral bias by comparing binned spectral energies across frequency bands.

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
    >>> loss_fn = BinnedSpectralLoss(n_bins=32, mu=1.0)
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
    Uses absolute energy ratios (not normalized distributions) per paper Algorithm 1.
    Encourages models to learn correct spectral content, mitigating
    spectral bias toward low frequencies.

    Attributes:
        n_bins: Number of frequency bins (default: 32)
        mu: Weight for BSP loss component (μ from paper, default: 1.0)
        epsilon: Numerical stability constant (default: 1e-8)
        binning_mode: 'linear' or 'log' frequency spacing (default: 'linear')
    """

    def __init__(
        self,
        n_bins: int = 32,
        mu: float = 1.0,
        epsilon: float = 1e-8,
        binning_mode: str = 'linear',
        signal_length: int = 4000,
        cache_path: str = None,
        lambda_k_mode: str = 'k_squared',
        use_log: bool = False
    ):
        """
        Initialize Binned Spectral Loss.

        Args:
            n_bins: Number of frequency bins (default: 32, matching paper)
            mu: Weight for BSP loss component (μ from paper, default: 1.0)
            epsilon: Numerical stability constant (default: 1e-8)
            binning_mode: Frequency spacing ('linear' or 'log', default: 'linear')
            signal_length: Expected signal length in time dimension (default: 4000 for CDON)
                          Used to pre-compute frequency bin edges for consistency
            cache_path: Optional path to precomputed spectrum cache (e.g., 'cache/true_spectrum.npz')
                       If provided, loads bin edges and downsamples to n_bins if needed
                       This ensures all BSP instances use identical bin edges derived from real data
            lambda_k_mode: Per-bin weight mode (λ_k from paper Algorithm 1):
                - 'k_squared': λ_k = k² (paper Table 4, turbulence - default)
                - 'uniform': λ_k = 1 (paper Table 4, airfoil / log BSP)
            use_log: Apply log10 to energies before binning (log BSP variant, default: False)

        Raises:
            ValueError: If binning_mode is not 'linear' or 'log'
            ValueError: If lambda_k_mode is not 'k_squared' or 'uniform'
        """
        super().__init__()
        self.n_bins = n_bins
        self.mu = mu
        self.epsilon = epsilon
        self.signal_length = signal_length
        self.use_log = use_log

        if binning_mode not in ['linear', 'log']:
            raise ValueError(f"binning_mode must be 'linear' or 'log', got {binning_mode}")
        self.binning_mode = binning_mode

        if lambda_k_mode not in ['k_squared', 'uniform']:
            raise ValueError(f"lambda_k_mode must be 'k_squared' or 'uniform', got {lambda_k_mode}")
        self.lambda_k_mode = lambda_k_mode

        # LOAD bin edges from cache if provided, otherwise compute
        if cache_path is not None:
            bin_edges = self._load_bin_edges_from_cache(cache_path, n_bins, binning_mode)
        else:
            # Fallback: compute bin edges from signal_length
            frequencies = torch.fft.rfftfreq(signal_length, dtype=torch.float32)
            freq_min = frequencies[1]  # Skip DC (frequency = 0)
            freq_max = frequencies[-1]  # Nyquist frequency

            if binning_mode == 'linear':
                bin_edges = torch.linspace(freq_min, freq_max, n_bins + 1, dtype=torch.float32)
            else:  # 'log'
                bin_edges = torch.logspace(
                    torch.log10(freq_min), torch.log10(freq_max),
                    n_bins + 1, dtype=torch.float32
                )

        # Register as buffer so it moves with model to GPU/CPU
        self.register_buffer('bin_edges', bin_edges)

        # Bin-specific weights λ_k (BSP paper Algorithm 1, Table 4)
        # Compute based on lambda_k_mode
        if lambda_k_mode == 'k_squared':
            # λ_k = k² (paper Table 4, turbulence cases)
            # Use bin centers as frequency values
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            lambda_k = bin_centers ** 2
            # Normalize to preserve total weight: sum(λ_k) = n_bins
            lambda_k = lambda_k / lambda_k.sum() * n_bins
        elif lambda_k_mode == 'uniform':
            # λ_k = 1 (paper Table 4, airfoil case)
            lambda_k = torch.ones(n_bins, dtype=torch.float32)

        self.register_buffer('bin_weights', lambda_k)

    def _load_bin_edges_from_cache(
        self,
        cache_path: str,
        n_bins: int,
        binning_mode: str
    ) -> torch.Tensor:
        """
        Load high-resolution bin edges from cache and downsample to n_bins.

        This ensures all BSP loss instances use identical bin edges derived
        from the real data distribution, regardless of their n_bins setting.

        Args:
            cache_path: Path to precomputed spectrum cache
            n_bins: Target number of bins
            binning_mode: 'linear' or 'log' binning

        Returns:
            Bin edges [n_bins + 1] as torch.Tensor

        Algorithm:
            1. Load high-resolution bin edges from cache (e.g., 257 edges for 256 bins)
            2. If n_bins matches cached resolution, use directly
            3. If n_bins < cached resolution, downsample by selecting evenly spaced edges
            4. If n_bins > cached resolution, interpolate (rare case)
        """
        import numpy as np
        from pathlib import Path

        cache_file = Path(cache_path)
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                f"Run scripts/precompute_spectrum.py first to generate the cache."
            )

        # Load cached bin edges
        cached = np.load(cache_path)

        # Check if bin_edges are in cache (backward compatibility)
        if 'bin_edges' not in cached:
            raise ValueError(
                f"Cache file {cache_path} does not contain 'bin_edges'.\n"
                f"Re-run scripts/precompute_spectrum.py to regenerate with bin edges."
            )

        cached_bin_edges = cached['bin_edges']  # [n_bins_cached + 1]
        cached_n_bins = len(cached_bin_edges) - 1

        # Case 1: Exact match - use directly
        if n_bins == cached_n_bins:
            bin_edges = torch.from_numpy(cached_bin_edges).float()
            print(f"📂 Loaded {n_bins} bin edges from {cache_path}")
            return bin_edges

        # Case 2: Downsample - select evenly spaced edges
        elif n_bins < cached_n_bins:
            # Select n_bins+1 evenly spaced indices from cached edges
            # This preserves freq_min and freq_max exactly
            indices = np.linspace(0, cached_n_bins, n_bins + 1, dtype=int)
            downsampled_edges = cached_bin_edges[indices]
            bin_edges = torch.from_numpy(downsampled_edges).float()
            print(f"📂 Loaded and downsampled bin edges: {cached_n_bins} → {n_bins} bins from {cache_path}")
            return bin_edges

        # Case 3: Upsample (rare) - interpolate
        else:
            # Use linear interpolation to create more bins
            # Preserves freq_min and freq_max
            freq_min = cached_bin_edges[0]
            freq_max = cached_bin_edges[-1]

            if binning_mode == 'linear':
                upsampled_edges = np.linspace(freq_min, freq_max, n_bins + 1)
            else:  # 'log'
                upsampled_edges = np.logspace(
                    np.log10(freq_min), np.log10(freq_max),
                    n_bins + 1
                )

            bin_edges = torch.from_numpy(upsampled_edges).float()
            print(f"⚠️  Upsampled bin edges: {cached_n_bins} → {n_bins} bins (cache resolution lower than requested)")
            return bin_edges

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binned Spectral Power loss.

        Args:
            pred: Predicted output [B, C, T] where T=4000
            target: Ground truth [B, C, T]

        Returns:
            Scalar BSP loss (weighted by μ)

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

        # Step 2: Compute power spectrum (BSP paper Algorithm 1, line 93)
        # E = (1/2)|û|² for energy per mode
        # Shape: [B, C, T//2+1] complex → [B, C, T//2+1] real
        pred_energy = 0.5 * torch.abs(pred_fft) ** 2
        target_energy = 0.5 * torch.abs(target_fft) ** 2

        # Step 2.5: Log transform (Log BSP variant)
        if self.use_log:
            # Apply log10 to energies (helps with wide dynamic range)
            # Add epsilon before log to avoid log(0)
            pred_energy = torch.log10(pred_energy + self.epsilon)
            target_energy = torch.log10(target_energy + self.epsilon)

        # Step 3: Bin-average energies
        # Shape: [B, C, T//2+1] → [B, C, n_bins]
        T = pred.shape[-1]
        pred_binned = self._bin_energy_1d(pred_energy, T)
        target_binned = self._bin_energy_1d(target_energy, T)

        # Step 4: Mean Squared Percentage Error per bin (Paper Algorithm 1)
        # Compare absolute binned energies (NOT normalized distributions)
        # Paper formula: 1 - (E_u^bin + ε) / (E_v^bin + ε)
        relative_error = 1.0 - (pred_binned + self.epsilon) / (target_binned + self.epsilon)

        # Squared error
        squared_error = relative_error ** 2

        # Step 5: Paper formula (Algorithm 1, line 99):
        # L_spec = (1/N_k) * Σ_c Σ_i (...)²
        # = (1/N_k) mean over bins, SUM over channels, mean over batch
        # Shape: [B, C, n_bins] → [B, C] → [B] → scalar
        bsp_loss = squared_error.sum(dim=1).mean(dim=-1).mean()  # sum channels (Σ_c), mean bins (1/N_k), mean batch

        # Apply weight μ (paper's overall BSP weight)
        return self.mu * bsp_loss

    def _bin_energy_1d(
        self,
        energy: torch.Tensor,
        T: int
    ) -> torch.Tensor:
        """
        Bin-average energy spectrum using HARD binning (BSP paper Algorithm 1).

        Each frequency belongs to exactly ONE bin based on bin edges.
        Matches paper: E_u^bin(c,i) ← (1/N_i * sum_{k in bin_i} E_u(c,k))

        Args:
            energy: Power spectrum [B, C, T//2+1]
            T: Original time dimension (for frequency calculation)

        Returns:
            Binned energy [B, C, n_bins]

        Algorithm (BSP paper Algorithm 1):
            1. Compute frequency values for each FFT bin
            2. Create bin edges based on binning_mode
            3. HARD-assign each frequency to exactly ONE bin
            4. Average energy within each bin (mean over frequencies in bin)
        """
        B, C, n_freqs = energy.shape
        device = energy.device
        dtype = energy.dtype

        # Step 1: Get frequency values
        # torch.fft.rfftfreq returns normalized frequencies [0, 0.5]
        # Shape: [n_freqs] = [T//2+1]
        # Note: Explicitly move to device to ensure compatibility across PyTorch versions
        frequencies = torch.fft.rfftfreq(T, dtype=dtype).to(device)

        # Step 2: Use pre-computed bin edges (already on correct device via buffer)
        # Bin edges were computed in __init__() and registered as buffer
        bin_edges = self.bin_edges.to(device)  # Ensure bin_edges on same device

        # Step 3: HARD BINNING - assign each frequency to exactly one bin
        # frequencies[1:] to skip DC component
        freq_no_dc = frequencies[1:]  # [n_freqs-1]

        # Find which bin each frequency belongs to
        # For n_bins, we have n_bins+1 edges: [e0, e1, ..., e_n]
        # Bins are: [e0,e1), [e1,e2), ..., [e_{n-1}, e_n]
        # searchsorted(bin_edges, freq, right=False) returns i where bin_edges[i-1] <= freq < bin_edges[i]
        # So bin index = i - 1, clamped to [0, n_bins-1]
        search_idx = torch.searchsorted(bin_edges, freq_no_dc, right=False)
        bin_indices = torch.clamp(search_idx - 1, 0, self.n_bins - 1)

        # Step 4: Average energy per bin using scatter operations
        # energy_no_dc: [B, C, n_freqs-1]
        energy_no_dc = energy[:, :, 1:]  # Skip DC component

        # Initialize binned energy [B, C, n_bins]
        binned_energy = torch.zeros(B, C, self.n_bins, device=device, dtype=dtype)

        # Count frequencies per bin [n_bins]
        bin_counts = torch.zeros(self.n_bins, device=device, dtype=dtype)

        # Expand bin_indices to match energy shape [B, C, n_freqs-1]
        bin_indices_expanded = bin_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)

        # Sum energy in each bin
        # binned_energy[b, c, bin_indices[f]] += energy_no_dc[b, c, f]
        binned_energy.scatter_add_(2, bin_indices_expanded, energy_no_dc)

        # Count frequencies per bin
        bin_counts.scatter_add_(0, bin_indices, torch.ones_like(freq_no_dc))

        # Average by dividing by count: (1/N_i) * sum_{k in bin_i} E(k)
        # Add epsilon to avoid division by zero for empty bins
        bin_counts = bin_counts.unsqueeze(0).unsqueeze(0)  # [1, 1, n_bins]
        binned_energy = binned_energy / (bin_counts + 1e-10)

        # Apply bin-specific weights λ_i (BSP paper Algorithm 1, lines 96-97)
        # E^bin(c,i) ← ... · λ_i
        # Shape: [B, C, n_bins] * [1, 1, n_bins] → [B, C, n_bins]
        # Ensure bin_weights are on the same device as binned_energy
        bin_weights = self.bin_weights.to(binned_energy.device)
        binned_energy = binned_energy * bin_weights.unsqueeze(0).unsqueeze(0)

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
        # Use 0.5 factor to match forward() method (BSP paper Algorithm 1, line 93)
        pred_energy = 0.5 * torch.abs(pred_fft) ** 2
        target_energy = 0.5 * torch.abs(target_fft) ** 2

        # Bin energies
        T = pred.shape[-1]
        pred_binned = self._bin_energy_1d(pred_energy, T)
        target_binned = self._bin_energy_1d(target_energy, T)

        # Relative error per bin (Paper Algorithm 1)
        # Use absolute binned energies (same as forward() method)
        # Paper formula: 1 - (E_u^bin + ε) / (E_v^bin + ε)
        relative_error = 1.0 - (pred_binned + self.epsilon) / (target_binned + self.epsilon)
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

            # Average over batch and channels (using absolute binned energies)
            pred_spectrum = pred_binned.mean(dim=(0, 1))  # [n_bins]
            target_spectrum = target_binned.mean(dim=(0, 1))  # [n_bins]

            # Relative errors (Paper Algorithm 1 formula)
            # Formula: 1 - (E_u^bin + ε) / (E_v^bin + ε)
            relative_errors = 1.0 - (pred_spectrum + self.epsilon) / (target_spectrum + self.epsilon)

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
