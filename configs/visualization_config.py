"""
Visualization and analysis configuration for spectral methods.

Centralizes all visualization-related constants to prevent configuration drift
across different modules and scripts.
"""

# High-resolution frequency binning for visualization and analysis
# Used for plotting energy spectra and analyzing spectral bias
# Higher resolution (256 bins) provides smooth, publication-quality plots
N_BINS_VISUALIZATION = 256

# Cache filename for precomputed true spectrum
# Updated to match N_BINS_VISUALIZATION for consistency
SPECTRUM_CACHE_FILENAME = 'true_spectrum_256bins.npz'

# Cache directory relative to project root
CACHE_DIR = 'cache'
