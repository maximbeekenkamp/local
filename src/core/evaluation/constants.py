"""
Constants for evaluation and loss functions.

Centralizes magic numbers to improve maintainability and documentation.
Values are based on BSP paper defaults, SA-PINNs implementation, and empirical tuning.
"""

# ============================================================================
# Numerical Stability
# ============================================================================

# Standard epsilon for division stability
EPSILON_DEFAULT = 1e-8

# Increased epsilon for SA-BSP (from BSP paper Table 2 ablation)
EPSILON_SA_BSP = 1e-6

# Minimum epsilon for numerical stability in log operations
EPSILON_LOG = 1e-10

# Mean-one normalization stability constant
EPSILON_MEAN_NORM = 1e-8


# ============================================================================
# SA-PINNs / Self-Adaptive Weights
# ============================================================================

# EMA smoothing factor for adaptive weights
# From: "Inspired by techniques in pruning literature" (needs proper citation)
# β=0.999 means: new_ema = 0.999 * old_ema + 0.001 * current_weight
EMA_BETA = 0.999

# Initial weight value for adaptive weights
# All weights initialized to 1.0 for fair starting point
INIT_WEIGHT_DEFAULT = 1.0


# ============================================================================
# Spectral Analysis
# ============================================================================

# Default number of frequency bins for BSP loss
# From BSP paper: 32 bins provides good frequency resolution
N_BINS_DEFAULT = 32

# CDON dataset signal length
# All temporal signals are 4000 timesteps
SIGNAL_LENGTH_CDON = 4000

# Percentile bounds for uncertainty visualization
# 16th and 84th percentiles approximate ±1σ for Gaussian distributions
PERCENTILE_LOWER = 16  # ~-1σ
PERCENTILE_MEDIAN = 50  # Median
PERCENTILE_UPPER = 84  # ~+1σ


# ============================================================================
# Training Configuration Defaults
# ============================================================================

# Default learning rate for SOAP optimizer
# Lowered from 3e-3 to 1e-3 for stability with large models (567K params)
LEARNING_RATE_DEFAULT = 1e-3

# Minimum learning rate for cosine annealing
LEARNING_RATE_MIN = 1e-6

# Default weight decay for regularization
WEIGHT_DECAY_DEFAULT = 1e-4

# Gradient clipping norm (prevents NaN with large models)
GRAD_CLIP_NORM = 1.0


# ============================================================================
# Batch Sizes
# ============================================================================

# Per-timestep batch size (for MSE loss on causal data)
# DeepONet generates ~320K samples, can use larger batches
BATCH_SIZE_PER_TIMESTEP = 32

# Sequence batch size (for BSP loss on full trajectories)
# FFT on predictions is memory-intensive, keep small
BATCH_SIZE_SEQUENCE = 4

# Ultra-conservative batch size for very large models in Colab
BATCH_SIZE_SEQUENCE_COLAB = 2


# ============================================================================
# Cache and File Paths
# ============================================================================

# Default cache directory
CACHE_DIR_DEFAULT = 'cache'

# Spectrum cache filename
SPECTRUM_CACHE_FILENAME = 'true_spectrum.npz'

# Target spectra cache filenames
TARGET_CACHE_LINEAR = 'target_spectra_linear.npz'
TARGET_CACHE_LOG = 'target_spectra_log.npz'


# ============================================================================
# Loss Function Defaults
# ============================================================================

# Default mu (μ) weight for BSP loss component
# μ=1.0 means MSE and BSP weighted equally
MU_DEFAULT = 1.0

# Default binning mode for frequency bins
BINNING_MODE_DEFAULT = 'linear'  # Options: 'linear' or 'log'

# Default lambda_k mode for per-bin weighting
LAMBDA_K_MODE_DEFAULT = 'k_squared'  # Options: 'k_squared' or 'uniform'

# Default loss type for BSP
LOSS_TYPE_BSP_DEFAULT = 'mspe'  # Mean Squared Percentage Error

# Default loss type for Log-BSP
LOSS_TYPE_LOG_BSP_DEFAULT = 'l2_norm'  # L2 norm of log differences


# ============================================================================
# Normalization Flags
# ============================================================================

# Default: apply per-batch output normalization before FFT
USE_OUTPUT_NORM_DEFAULT = True

# Default: apply per-sample min-max normalization to binned energies
USE_MINMAX_NORM_DEFAULT = True

# Default: use log transform for energies (Log-BSP)
USE_LOG_DEFAULT = False
