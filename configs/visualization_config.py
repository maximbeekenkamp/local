"""
Visualization and analysis configuration for spectral methods.

Centralizes all visualization-related constants to prevent configuration drift
across different modules and scripts.
"""

# Cache filename for precomputed true spectrum
# Contains both:
# - Unbinned spectrum (full FFT resolution ~2000 frequencies) for visualization
# - Binned spectrum (BSP n_bins, e.g. 32) for training consistency
SPECTRUM_CACHE_FILENAME = 'true_spectrum.npz'

# Cache directory relative to project root
CACHE_DIR = 'cache'

# CDON dataset temporal parameters (from dataset paper)
# 4000 timesteps at 50 Hz sampling rate
# Nyquist frequency: 25 Hz (half of sampling rate)
SAMPLING_RATE_HZ = 50.0  # Hz
NYQUIST_FREQ_HZ = 25.0   # Hz (= SAMPLING_RATE_HZ / 2)
SIGNAL_LENGTH = 4000     # Number of timesteps

# Visualization parameters
N_BINS_VISUALIZATION = 200  # High-resolution binning for smooth plots
FREQ_RANGE_HZ = (0, 25.0)   # Frequency range for plots (0-25Hz per dataset paper)
