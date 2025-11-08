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
