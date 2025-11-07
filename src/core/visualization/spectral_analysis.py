"""
Spectral bias analysis for 1D temporal neural operator predictions.

Adapted from 2D spatial energy spectrum analysis to 1D temporal frequency analysis.
Provides tools to visualize and quantify spectral bias in model predictions.

Reference: Binned Spectral Power Loss paper - spectral bias mitigation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from configs.visualization_config import N_BINS_VISUALIZATION, SPECTRUM_CACHE_FILENAME, CACHE_DIR


def compute_cached_true_spectrum(
    data: torch.Tensor,
    cache_path: str,
    n_bins: int = 64,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute true spectrum from ALL data and cache to disk.

    Uses all available samples to compute a smooth, accurate reference spectrum.
    Results are cached to avoid recomputation on subsequent runs.

    Args:
        data: All data samples [N, C, T] where N is full dataset size
        cache_path: Path to cache file (e.g., 'cache/true_spectrum_64bins.npz')
        n_bins: Number of frequency bins (default 64 for visualization)
        force_recompute: If True, recompute even if cache exists

    Returns:
        frequencies: Bin centers [n_bins] (normalized frequency, 0 to 0.5)
        energy: Power spectrum E(f) [n_bins] averaged over all samples

    Example:
        >>> # First call: computes and caches
        >>> freq, energy = compute_cached_true_spectrum(all_data, 'cache/spectrum.npz')
        ⚙️  Computing true spectrum from 1000 samples with 64 bins...
        💾 Saved true spectrum to cache/spectrum.npz

        >>> # Subsequent calls: loads from cache
        >>> freq, energy = compute_cached_true_spectrum(all_data, 'cache/spectrum.npz')
        📂 Loading cached true spectrum from cache/spectrum.npz
    """
    cache_file = Path(cache_path)

    # Check if cache exists
    if cache_file.exists() and not force_recompute:
        print(f"📂 Loading cached true spectrum from {cache_path}")
        cached = np.load(cache_path)
        return cached['frequencies'], cached['energy']

    # Compute from ALL data
    print(f"⚙️  Computing true spectrum from {data.shape[0]} samples with {n_bins} bins...")
    freq, energy = compute_frequency_spectrum_1d(data, n_bins=n_bins)

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, frequencies=freq, energy=energy, n_bins=n_bins)
    print(f"💾 Saved true spectrum to {cache_path}")

    return freq, energy


def compute_frequency_spectrum_1d(
    signal: torch.Tensor,
    n_bins: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D frequency spectrum for temporal data.

    Adapted from 2D spatial energy spectrum to 1D temporal case.
    Uses 1D FFT to analyze frequency content of temporal signals.

    Args:
        signal: [B, C, T] tensor (e.g., [16, 1, 4000])
                B=batch, C=channels, T=timesteps
        n_bins: Number of frequency bins (default 32, matching BSP paper)

    Returns:
        frequencies: Bin centers (normalized frequency, 0 to 0.5)
        energy: Power spectrum E(f) averaged over batch and channels

    Algorithm:
        1. Average over batch and channel dimensions
        2. Apply 1D real FFT
        3. Compute power spectrum: |FFT|²
        4. Bin frequencies with linear spacing
        5. Average energy within each bin

    Example:
        >>> signal = torch.randn(16, 1, 4000)
        >>> freqs, energy = compute_frequency_spectrum_1d(signal)
        >>> plt.loglog(freqs, energy)
    """
    # Convert to numpy and average over batch and channel
    if isinstance(signal, torch.Tensor):
        signal_np = signal.cpu().numpy()
    else:
        signal_np = signal

    # Average over batch (axis 0) and channel (axis 1)
    # Shape: [B, C, T] → [T]
    signal_avg = signal_np.mean(axis=(0, 1))

    # Compute 1D FFT (real signal → complex FFT)
    fft_result = np.fft.rfft(signal_avg)  # Shape: [T//2 + 1]

    # Compute power spectrum
    power_spectrum = np.abs(fft_result) ** 2

    # Create frequency bins
    n_freqs = len(power_spectrum)
    freq_indices = np.arange(n_freqs)

    # Linear binning (like BSP paper)
    bin_edges = np.linspace(0, n_freqs, n_bins + 1)

    # Compute bin-averaged energy
    energy_binned = np.zeros(n_bins)
    freq_binned = np.zeros(n_bins)

    for i in range(n_bins):
        # Find indices in this bin
        bin_mask = (freq_indices >= bin_edges[i]) & (freq_indices < bin_edges[i + 1])

        if bin_mask.any():
            # Average energy in this bin
            energy_binned[i] = power_spectrum[bin_mask].mean()
            # Bin center frequency
            freq_binned[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        else:
            energy_binned[i] = 0.0
            freq_binned[i] = bin_edges[i]

    # Normalize frequencies to [0, 0.5] (Nyquist range)
    timesteps = signal_np.shape[-1]
    freq_binned_normalized = freq_binned / timesteps

    return freq_binned_normalized, energy_binned


def compute_frequency_spectrum_batch(
    signal: torch.Tensor,
    n_bins: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frequency spectrum for each sample in batch (for uncertainty).

    Args:
        signal: [B, C, T] tensor
        n_bins: Number of frequency bins

    Returns:
        frequencies: Bin centers [n_bins]
        mean_energy: Mean power spectrum [n_bins]
        std_energy: Std dev of power spectrum [n_bins]
    """
    batch_size = signal.shape[0]

    # Compute spectrum for each sample
    spectra = []
    for b in range(batch_size):
        freq, energy = compute_frequency_spectrum_1d(signal[b:b+1], n_bins=n_bins)
        spectra.append(energy)

    # Stack and compute statistics
    spectra = np.array(spectra)  # [B, n_bins]
    mean_energy = spectra.mean(axis=0)
    std_energy = spectra.std(axis=0)

    return freq, mean_energy, std_energy


def plot_spectral_bias_comparison(
    predictions: Dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
    title: str = 'Frequency Spectrum Comparison (1D Temporal)',
    save_path: Optional[str] = None,
    n_bins: int = 32,
    show_uncertainty: bool = True,
    figsize: Tuple[int, int] = (10, 7)
):
    """
    Recreate the spectral bias plot from reference PDF.

    Creates log-log plot comparing model predictions to ground truth
    in frequency domain. Shows which models capture high-frequency
    content better (less spectral bias).

    Args:
        predictions: {'deeponet': pred_tensor, 'fno': pred_tensor, ...}
                     Each tensor is [B, C, T]
        ground_truth: [B, C, T] target tensor
        title: Plot title
        save_path: Optional path to save figure
        n_bins: Number of frequency bins
        show_uncertainty: If True, plot shaded ±1 std bands
        figsize: Figure size (width, height)

    Visualization style matches reference:
        - Log-log axes
        - Shaded uncertainty bands (±1 std across batch)
        - Color scheme: DeepONet=blue, FNO=orange, UNet=green
        - Ground truth = black solid line with label "True"

    Example:
        >>> predictions = {
        ...     'deeponet': deeponet_pred,  # [16, 1, 4000]
        ...     'fno': fno_pred,
        ...     'unet': unet_pred
        ... }
        >>> plot_spectral_bias_comparison(predictions, ground_truth)
    """
    # Color scheme matching reference plot
    colors = {
        'deeponet': '#1f77b4',  # Blue
        'fno': '#ff7f0e',       # Orange
        'unet': '#2ca02c',      # Green
        'unet_fno': '#d62728',  # Red (if present)
    }

    # Model display names
    display_names = {
        'deeponet': 'DeepONet',
        'fno': 'FNO',
        'unet': 'UNet',
        'unet_fno': 'UNet+FNO'
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ground truth using cached spectrum from ALL data
    # Always use N_BINS_VISUALIZATION for smooth, high-resolution visualization (independent of n_bins parameter)
    cache_dir = Path(save_path).parent / CACHE_DIR if save_path else Path(CACHE_DIR)
    cache_path = cache_dir / SPECTRUM_CACHE_FILENAME
    freq_gt, mean_gt = compute_cached_true_spectrum(
        ground_truth,
        cache_path=str(cache_path),
        n_bins=N_BINS_VISUALIZATION  # High resolution for smooth visualization
    )

    ax.plot(freq_gt, mean_gt, 'k-', linewidth=2, label='True', zorder=10)

    # Plot each model's predictions
    for model_name, pred in predictions.items():
        color = colors.get(model_name.lower(), 'gray')
        display_name = display_names.get(model_name.lower(), model_name.upper())

        if show_uncertainty:
            freq, mean_energy, std_energy = compute_frequency_spectrum_batch(
                pred, n_bins=n_bins
            )
            # Shaded uncertainty band
            ax.fill_between(freq, mean_energy - std_energy, mean_energy + std_energy,
                           alpha=0.3, color=color)
        else:
            freq, mean_energy = compute_frequency_spectrum_1d(pred, n_bins=n_bins)

        # Model prediction line
        ax.plot(freq, mean_energy, color=color, linewidth=2,
               label=display_name, alpha=0.9)

    # Log-log axes (like reference plot)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and formatting
    ax.set_xlabel('Normalized Frequency', fontsize=12)
    ax.set_ylabel('Power Spectrum E(f)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Set reasonable axis limits
    ax.set_xlim(freq_gt[1], freq_gt[-1])  # Skip DC component

    plt.tight_layout()

    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectral bias plot to: {save_path}")

    plt.show()

    return fig, ax


def compute_spectral_bias_metric(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    low_freq_cutoff: float = 0.3,
    high_freq_cutoff: float = 0.7,
    n_bins: int = 32
) -> Dict[str, float]:
    """
    Quantify spectral bias as error ratio across frequency ranges.

    Spectral bias: Models tend to learn low frequencies better than
    high frequencies. This metric quantifies that effect.

    Args:
        prediction: [B, C, T] predicted tensor
        ground_truth: [B, C, T] target tensor
        low_freq_cutoff: Normalized frequency threshold for "low" (default 0.3)
        high_freq_cutoff: Normalized frequency threshold for "high" (default 0.7)
        n_bins: Number of frequency bins

    Returns:
        {
            'low_freq_error': Relative error in low frequencies
            'mid_freq_error': Relative error in mid frequencies
            'high_freq_error': Relative error in high frequencies
            'spectral_bias_ratio': high_error / low_error (>1 = biased)
            'total_error': Overall spectrum error
        }

    Interpretation:
        - spectral_bias_ratio ≈ 1.0: No bias (equal errors)
        - spectral_bias_ratio > 2.0: Significant bias to low frequencies
        - spectral_bias_ratio < 0.5: Model better at high frequencies (rare)
    """
    # Compute spectra
    freq_pred, energy_pred = compute_frequency_spectrum_1d(prediction, n_bins=n_bins)
    freq_gt, energy_gt = compute_frequency_spectrum_1d(ground_truth, n_bins=n_bins)

    # Compute adaptive cutoffs based on actual Nyquist frequency
    # rfft gives frequencies [0, 0.5] for real signals, not [0, 1.0]
    max_freq = freq_pred[-1]  # Nyquist frequency (~0.5 for rfft)
    low_cutoff = 0.3 * max_freq      # 30% of Nyquist
    high_cutoff = 0.7 * max_freq     # 70% of Nyquist

    # Compute relative error per bin
    # Avoid division by zero
    epsilon = 1e-10
    relative_error = np.abs(energy_pred - energy_gt) / (energy_gt + epsilon)

    # Divide into frequency ranges (using adaptive cutoffs)
    low_mask = freq_pred < low_cutoff
    mid_mask = (freq_pred >= low_cutoff) & (freq_pred < high_cutoff)
    high_mask = freq_pred >= high_cutoff

    # Average error in each range
    low_freq_error = relative_error[low_mask].mean() if low_mask.any() else 0.0
    mid_freq_error = relative_error[mid_mask].mean() if mid_mask.any() else 0.0
    high_freq_error = relative_error[high_mask].mean() if high_mask.any() else 0.0
    total_error = relative_error.mean()

    # Spectral bias ratio
    if low_freq_error > epsilon:
        spectral_bias_ratio = high_freq_error / low_freq_error
    else:
        spectral_bias_ratio = float('inf')  # All low-freq correct, high-freq wrong

    return {
        'low_freq_error': float(low_freq_error),
        'mid_freq_error': float(mid_freq_error),
        'high_freq_error': float(high_freq_error),
        'spectral_bias_ratio': float(spectral_bias_ratio),
        'total_error': float(total_error)
    }


def plot_spectral_bias_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot bar chart of spectral bias metrics for multiple models.

    Args:
        metrics_dict: {'deeponet': metrics, 'fno': metrics, ...}
                      Each metrics is output of compute_spectral_bias_metric
        save_path: Optional path to save figure

    Example:
        >>> metrics = {}
        >>> for model in ['deeponet', 'fno', 'unet']:
        ...     metrics[model] = compute_spectral_bias_metric(preds[model], gt)
        >>> plot_spectral_bias_metrics(metrics)
    """
    models = list(metrics_dict.keys())
    n_models = len(models)

    # Extract metrics
    low_errors = [metrics_dict[m]['low_freq_error'] for m in models]
    mid_errors = [metrics_dict[m]['mid_freq_error'] for m in models]
    high_errors = [metrics_dict[m]['high_freq_error'] for m in models]
    bias_ratios = [metrics_dict[m]['spectral_bias_ratio'] for m in models]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Errors by frequency range
    x = np.arange(n_models)
    width = 0.25

    axes[0].bar(x - width, low_errors, width, label='Low Freq', color='#2ca02c', alpha=0.8)
    axes[0].bar(x, mid_errors, width, label='Mid Freq', color='#ff7f0e', alpha=0.8)
    axes[0].bar(x + width, high_errors, width, label='High Freq', color='#d62728', alpha=0.8)

    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Relative Error', fontsize=12)
    axes[0].set_title('Spectral Error by Frequency Range', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in models])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Spectral bias ratio
    colors_ratio = ['#1f77b4', '#ff7f0e', '#2ca02c'][:n_models]
    axes[1].bar(x, bias_ratios, color=colors_ratio, alpha=0.8)
    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                    label='No Bias (ratio=1)')

    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Spectral Bias Ratio\n(High Freq Error / Low Freq Error)', fontsize=12)
    axes[1].set_title('Spectral Bias Ratio', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in models])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectral bias metrics plot to: {save_path}")

    plt.show()

    return fig, axes
