#!/usr/bin/env python3
"""
Generate demo_training.ipynb from Python code.

This script creates the entire notebook programmatically.
To regenerate the notebook, simply run: python scripts/generate_demo_notebook.py
"""

import json
from pathlib import Path


def markdown_cell(source):
    """Create a markdown cell."""
    lines = source.split("\n")
    # Add newline character to all lines except the last one
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    }


def code_cell(source):
    """Create a code cell."""
    lines = source.split("\n")
    # Add newline character to all lines except the last one
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    }


# Cell definitions extracted from notebook
CELLS = [
    # Cell 0: Markdown
    markdown_cell("""# Neural Operator Training Demo: CDON Dataset

This notebook demonstrates end-to-end training of neural operator models (DeepONet, FNO, UNet) on the CDON dataset.

**Features:**
- Trains on **real CDON data**
- **Sequential training with all 6 loss functions**:
  - **BASELINE**: Relative L2 loss only (baseline MSE)
  - **BSP**: MSE + fixed BSP loss with k² weighting
  - **Log-BSP**: MSE + BSP with log₁₀ spectral energies (uniform λ_k weighting)
  - **SA-BSP (Per-bin)**: MSE + 32 adaptive per-bin weights (negated gradients for frequency emphasis)
  - **SA-BSP (Global)**: MSE + 2 adaptive weights (w_mse + w_bsp) for MSE/BSP balance
  - **SA-BSP (Combined)**: MSE + 34 weights (w_mse + w_bsp + 32 per-bin) with full competitive dynamics
- **Multi-loss comparison plots** showing training metrics
- **Energy spectrum visualization** (E(k) vs wavenumber) to identify spectral bias
- **Spectral bias quantification** with metrics and comparison plots
- Compatible with Google Colab

**Models available:**
- `deeponet`: Branch-trunk architecture with SIREN activation (~235K params)
- `fno`: Fourier Neural Operator (~261K params)
- `unet`: Encoder-decoder with skip connections (~249K params)

**SA-PINNs Implementation:**
Uses saddle-point optimization with negated gradients (gradient ascent on loss) to enable competitive dynamics. This automatically emphasizes difficult frequency bins and finds optimal loss balance through min-max optimization."""),

    # Cell 1: Markdown
    markdown_cell("""## Cell 0: Force Reload Modules

Run this cell to reload all project modules after code changes."""),

    # Cell 2: Code
    code_cell("""# Force reload of all modules
import sys
import importlib

# Get list of all loaded modules from the project
modules_to_reload = []
for module_name in list(sys.modules.keys()):
    if any(x in module_name for x in ['src.', 'configs.']):
        modules_to_reload.append(module_name)

# Remove modules from sys.modules to force reload
for module_name in modules_to_reload:
    if module_name in sys.modules:
        del sys.modules[module_name]

print(f"✓ Cleared {len(modules_to_reload)} cached modules")
print("  Run Cell 1 to reimport all modules with latest code")"""),

    # Cell 3: Markdown
    markdown_cell("""## Cell 1: Setup & Imports (Colab-Ready)"""),

    # Cell 4: Code
    code_cell("""# Google Colab setup
import sys
import os
from pathlib import Path

# Ensure we're in /content
try:
    os.chdir('/content')
except:
    pass

# Clone repository if running in Colab
repo_path = Path('/content/local')
if not repo_path.exists():
    print("📥 Cloning repository...")
    !git clone https://github.com/maximbeekenkamp/local.git
    print("✅ Repository cloned")
else:
    print("📥 Updating repository...")
    !git -C /content/local pull
    print("✅ Repository updated")

# Change to repo directory
try:
    os.chdir('/content/local')
    print(f"✅ Changed to: {os.getcwd()}")
except:
    pass

# Install dependencies - IMPORTANT: Upgrade numpy FIRST to avoid binary incompatibility
print("\\n📦 Installing dependencies...")
print("🔧 Upgrading numpy to 2.x (fixes binary compatibility)...")
!pip install "numpy>=2.0.0" --upgrade -q
print("✅ NumPy upgraded")

print("📦 Installing other dependencies...")
!pip install -r requirements.txt -q
print("✅ Dependencies installed")

# Standard imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Project imports
from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.models.model_factory import create_model
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig

print("\\n✓ Imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")"""),

    # Cell 5: Markdown
    markdown_cell("""## Cell 2: Configuration

Configure optional features before loading data:
- **Causal padding**: Zero-padding preprocessing (Reference CausalityDeepONet)
- **DeepONet activation**: Choose activation function (REQU, TANH, RELU, SIREN)
- **Penalty loss**: Optional inverse-variance weighting"""),

    # Cell 6: Code
    code_cell("""# ============================================================================
# CONFIGURATION
# ============================================================================

# 1. CAUSALITY: Zero-padding preprocessing (Reference CausalityDeepONet)
USE_CAUSAL_PADDING = True  # ENABLED BY DEFAULT (matches reference)
# Set to False to disable causal padding (standard preprocessing)

# 2. DEEPONET ACTIVATION: Choose activation function
DEEPONET_ACTIVATION = 'requ'  # Options: 'requ' (default), 'tanh', 'relu', 'siren'
# 'requ' = ReLU² (reference default, smooth gradients)
# 'tanh' = Stable for operator learning
# 'relu' = Standard ReLU
# 'siren' = Sinusoidal activation (requires siren-pytorch)

# 3. PENALTY LOSS: Optional inverse-variance weighting
USE_PENALTY_LOSS = False  # Set to True to enable penalty weighting
PENALTY_EPSILON = 1e-8     # Numerical stability for penalty
PENALTY_PER_SAMPLE = True  # Per-sample (True) or global (False) penalty

print("✓ Configuration loaded:")
print(f"  Causal padding:     {'ENABLED' if USE_CAUSAL_PADDING else 'DISABLED'}")
print(f"  DeepONet activation: {DEEPONET_ACTIVATION.upper()}")
print(f"  Penalty loss:       {'ENABLED' if USE_PENALTY_LOSS else 'DISABLED'}")
print()

# ============================================================================
# LOAD REAL CDON DATA
# ============================================================================

# Get project root
project_root = Path.cwd()
print(f"Project root: {project_root}")

# Data directory
DATA_DIR = project_root / 'CDONData'
print(f"Data directory: {DATA_DIR}")

# Create normalization object (required by CDONDataset)
stats_path = project_root / 'configs' / 'cdon_stats.json'
print(f"Loading stats from: {stats_path}")
normalizer = CDONNormalization(stats_path=str(stats_path))

# Create datasets with optional causal padding
train_dataset = CDONDataset(
    data_dir=str(DATA_DIR),
    split='train',
    normalize=normalizer,
    mode='sequence',  # Use sequence mode for BSP loss training
    use_causal_sequence=USE_CAUSAL_PADDING,  # Apply causal padding if enabled
    signal_length=4000
)

val_dataset = CDONDataset(
    data_dir=str(DATA_DIR),
    split='test',
    normalize=normalizer,
    mode='sequence',  # Use sequence mode for BSP loss training
    use_causal_sequence=USE_CAUSAL_PADDING,  # Apply causal padding if enabled
    signal_length=4000
)

# Create dataloaders
BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"\\n✓ Data loaded successfully")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  Batch size: {BATCH_SIZE}")

# Inspect a sample
sample_input, sample_target = train_dataset[0]
print(f"\\nSample shapes:")
print(f"  Input: {sample_input.shape}")
print(f"  Target: {sample_target.shape}")

if USE_CAUSAL_PADDING:
    expected_input_len = 4000 + (4000 - 1)  # signal_length + padding
    if sample_input.shape[-1] == expected_input_len:
        print(f"  ✓ Causal padding applied correctly (input length: {expected_input_len})")
    else:
        print(f"  ⚠ Warning: Expected input length {expected_input_len}, got {sample_input.shape[-1]}")"""),

    # Cell 7: Markdown
    markdown_cell("""## Cell 3.5: Loss Variant Configuration (Quick Switch)

**Easily switch between loss variants by changing `LOSS_VARIANT`:**

Available variants:
- `'bsp'`: Fixed BSP with k² weighting (static spectral loss)
- `'log_bsp'`: Log-domain BSP with uniform weighting (wide dynamic range)
- `'sa_bsp'`: Self-Adaptive BSP with trainable per-bin weights (emphasis on difficult frequencies)
- `'sa_log_bsp'`: Self-Adaptive Log-BSP (combines adaptive weights with log-domain energies)

Each variant can use different adaptation modes (per-bin, global, combined) via `SA_ADAPT_MODE`."""),

    # Cell 8: Code
    code_cell("""# ============================================================================
# LOSS VARIANT CONFIGURATION - Change this to switch between loss types
# ============================================================================

# Choose loss variant
LOSS_VARIANT = 'bsp'  # Options: 'bsp', 'log_bsp', 'sa_bsp', 'sa_log_bsp'

# For SA-BSP variants, choose adaptation mode
SA_ADAPT_MODE = 'per-bin'  # Options: 'per-bin', 'global', 'combined'
# 'per-bin': 32 trainable weights (one per frequency bin)
# 'global': 2 trainable weights (w_mse + w_bsp for MSE/BSP balance)
# 'combined': 34 trainable weights (w_mse + w_bsp + 32 per-bin)

# Common parameters for all loss variants
N_BINS = 32
SIGNAL_LENGTH = 4000
CACHE_PATH = 'cache/true_spectrum.npz'
EPSILON = 1e-8

# Loss variant configurations
LOSS_CONFIGS = {
    # Fixed BSP with k² weighting
    'bsp': {
        'loss_type': 'combined',
        'loss_params': {
            'base_loss': 'field_error',
            'spectral_loss': 'bsp',
            'mu': 1.0,
            'n_bins': N_BINS,
            'epsilon': EPSILON,
            'binning_mode': 'linear',
            'signal_length': SIGNAL_LENGTH,
            'cache_path': CACHE_PATH,
            'lambda_k_mode': 'k_squared',  # Static k² weighting for turbulence
            'use_log': False,              # Standard energy (not log)
            'use_output_norm': True,       # Per-batch output normalization
            'use_minmax_norm': True,       # Per-sample min-max normalization
            'loss_type': 'mspe'            # Mean Squared Percentage Error
        },
        'description': 'MSE + BSP (μ=1.0, λ_k=k²) with normalization - Fixed spectral loss'
    },
    
    # Log-domain BSP with uniform weighting
    'log_bsp': {
        'loss_type': 'combined',
        'loss_params': {
            'base_loss': 'field_error',
            'spectral_loss': 'bsp',
            'mu': 1.0,
            'n_bins': N_BINS,
            'epsilon': EPSILON,
            'binning_mode': 'linear',
            'signal_length': SIGNAL_LENGTH,
            'cache_path': CACHE_PATH,
            'lambda_k_mode': 'uniform',    # Uniform λ_k = 1 for all bins
            'use_log': True,               # Log₁₀ transform of energies
            'use_output_norm': True,       # Per-batch output normalization
            'use_minmax_norm': True,       # Per-sample min-max normalization
            'loss_type': 'l2_norm'         # L2 norm loss
        },
        'description': 'MSE + Log-BSP: log₁₀(E) with uniform weighting - Wide dynamic range'
    },
    
    # Self-Adaptive BSP (trainable weights)
    'sa_bsp': {
        'loss_type': 'combined',
        'loss_params': {
            'base_loss': 'field_error',
            'spectral_loss': 'sa_bsp',
            'n_bins': N_BINS,
            'adapt_mode': SA_ADAPT_MODE,   # Controlled by SA_ADAPT_MODE variable above
            'init_weight': 1.0,
            'epsilon': 1e-6,               # Increased from 1e-8 per paper ablation
            'binning_mode': 'linear',
            'signal_length': SIGNAL_LENGTH,
            'cache_path': CACHE_PATH,
            'lambda_k_mode': 'k_squared',  # k² initialization for trainable weights
            'use_log': False,              # Standard energy (not log)
            'use_output_norm': True,       # Per-batch output normalization
            'use_minmax_norm': True,       # Per-sample min-max normalization
            'loss_type': 'mspe'            # Mean Squared Percentage Error
        },
        'description': f'MSE + SA-BSP ({SA_ADAPT_MODE}): Trainable λ_k weights (init: k²) - Adaptive emphasis'
    },
    
    # Self-Adaptive Log-BSP (trainable weights + log-domain)
    'sa_log_bsp': {
        'loss_type': 'combined',
        'loss_params': {
            'base_loss': 'field_error',
            'spectral_loss': 'sa_bsp',
            'n_bins': N_BINS,
            'adapt_mode': SA_ADAPT_MODE,   # Controlled by SA_ADAPT_MODE variable above
            'init_weight': 1.0,
            'epsilon': 1e-6,               # Increased from 1e-8 per paper ablation
            'binning_mode': 'linear',
            'signal_length': SIGNAL_LENGTH,
            'cache_path': CACHE_PATH,
            'lambda_k_mode': 'uniform',    # Uniform initialization for log-domain
            'use_log': True,               # Log₁₀ transform of energies
            'use_output_norm': True,       # Per-batch output normalization
            'use_minmax_norm': True,       # Per-sample min-max normalization
            'loss_type': 'l2_norm'         # L2 norm loss
        },
        'description': f'MSE + SA-Log-BSP ({SA_ADAPT_MODE}): Trainable weights + log₁₀(E) - Adaptive + wide range'
    }
}

# Get selected loss configuration
from configs.loss_config import LossConfig
selected_loss_config = LossConfig.from_dict(LOSS_CONFIGS[LOSS_VARIANT])

print("✓ Loss variant configuration loaded:")
print(f"  Variant: {LOSS_VARIANT.upper()}")
if 'sa_' in LOSS_VARIANT:
    print(f"  SA Mode: {SA_ADAPT_MODE.upper()}")
print(f"  Description: {selected_loss_config.description}")
print(f"\\n  Parameters:")
for key, value in selected_loss_config.loss_params.items():
    print(f"    {key}: {value}")"""),

    # Cell 9: Markdown
    markdown_cell("""## Cell 3.7: Spectrum Comparison Utilities

Helper functions for comparing energy spectra across different loss variants."""),

    # Cell 10: Code
    code_cell("""# ============================================================================
# SPECTRUM COMPARISON UTILITIES
# ============================================================================

def compare_loss_variants(
    trained_models_dict, 
    val_loader, 
    loss_variants=['bsp', 'log_bsp', 'sa_bsp', 'sa_log_bsp'],
    model_arch='deeponet',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    \"\"\"
    Compare energy spectra across different loss variants.
    
    Args:
        trained_models_dict: Dictionary of trained models {variant: model}
        val_loader: Validation data loader
        loss_variants: List of loss variant names to compare
        model_arch: Model architecture name
        device: Device for inference
        
    Returns:
        Dictionary of spectra {variant: {frequencies, energy_median, energy_p16, energy_p84}}
    \"\"\"
    from src.core.visualization.spectral_analysis import compute_unbinned_spectrum, compute_cached_true_spectrum
    from configs.visualization_config import SPECTRUM_CACHE_FILENAME, CACHE_DIR
    
    print("Computing energy spectra for loss variant comparison...")
    
    # Load true spectrum from cache
    cache_path = f'{CACHE_DIR}/{SPECTRUM_CACHE_FILENAME}'
    print(f"Loading true spectrum from cache: {cache_path}")
    cached = np.load(cache_path)
    k_true = cached['unbinned_frequencies']
    E_true_median = cached['unbinned_energy_median']
    E_true_p16 = cached['unbinned_energy_p16']
    E_true_p84 = cached['unbinned_energy_p84']
    print(f"✓ True spectrum loaded ({len(k_true)} frequencies)")
    
    spectra = {}
    
    # Store true spectrum
    spectra['True'] = {
        'frequencies': k_true,
        'energy_median': E_true_median,
        'energy_p16': E_true_p16,
        'energy_p84': E_true_p84
    }
    
    # Compute spectra for each loss variant
    for variant in loss_variants:
        key = f"{model_arch}_{variant}"
        
        if key not in trained_models_dict:
            print(f"  ⚠️  Skipping {variant}: model key '{key}' not found")
            continue
        
        model = trained_models_dict[key]
        model.eval()
        model.to(device)
        
        try:
            # Collect predictions from all validation batches
            all_preds = []
            print(f"  Processing {variant.upper()}...", end='')
            
            with torch.no_grad():
                for val_input, _ in val_loader:
                    val_input = val_input.to(device)
                    pred = model(val_input)
                    all_preds.append(pred.cpu())
            
            # Stack predictions
            all_preds_tensor = torch.cat(all_preds, dim=0)
            
            # Compute unbinned spectrum with percentile-based uncertainty
            k_pred, E_pred_median, E_pred_p16, E_pred_p84 = compute_unbinned_spectrum(all_preds_tensor)
            
            spectra[variant] = {
                'frequencies': k_pred,
                'energy_median': E_pred_median,
                'energy_p16': E_pred_p16,
                'energy_p84': E_pred_p84
            }
            
            print(f" ✓ ({all_preds_tensor.shape[0]} samples)")
        except Exception as e:
            print(f" ❌ Error: {e}")
            continue
    
    print(f"\\n✓ Spectra computed for {len(spectra)} entries\\n")
    return spectra


def plot_spectrum_comparison(
    spectra,
    variants_to_plot=['bsp', 'log_bsp', 'sa_bsp', 'sa_log_bsp'],
    title_suffix="",
    figsize=(14, 9)
):
    \"\"\"
    Plot energy spectrum comparison for selected loss variants.
    
    Args:
        spectra: Dictionary of spectra from compare_loss_variants()
        variants_to_plot: List of variant names to include in plot
        title_suffix: Additional text for plot title
        figsize: Figure size tuple
    \"\"\"
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme
    colors = {
        'True': '#000000',
        'bsp': '#ff7f0e',
        'log_bsp': '#2ca02c',
        'sa_bsp': '#d62728',
        'sa_log_bsp': '#9467bd'
    }
    
    # Plot ground truth
    if 'True' in spectra:
        data = spectra['True']
        k = data['frequencies']
        E_median = data['energy_median']
        E_p16 = data['energy_p16']
        E_p84 = data['energy_p84']
        
        ax.loglog(k, E_median, color=colors['True'], linewidth=3, 
                 label='True (Real Data)', zorder=10, alpha=0.9)
        ax.fill_between(k, E_p16, E_p84,
                        color=colors['True'], alpha=0.15, zorder=9,
                        label='True (16th-84th percentile)')
    
    # Plot model predictions
    label_map = {
        'bsp': 'BSP (k² weighting)',
        'log_bsp': 'Log-BSP (uniform λ)',
        'sa_bsp': 'SA-BSP (adaptive λ)',
        'sa_log_bsp': 'SA-Log-BSP (adaptive + log)'
    }
    
    for variant in variants_to_plot:
        if variant not in spectra:
            print(f"  ⚠️  Skipping {variant}: not in spectra dictionary")
            continue
        
        data = spectra[variant]
        k = data['frequencies']
        E_median = data['energy_median']
        E_p16 = data['energy_p16']
        E_p84 = data['energy_p84']
        
        color = colors.get(variant, '#888888')
        label = label_map.get(variant, variant.upper())
        
        # Plot median line
        ax.loglog(k, E_median, color=color, linewidth=2.5, 
                 alpha=0.85, label=label, zorder=5)
        
        # Plot uncertainty band
        ax.fill_between(k, E_p16, E_p84,
                        color=color, alpha=0.12, zorder=4)
    
    # Configure plot
    ax.set_xlabel('Frequency (normalized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('E(k) - Spectral Power', fontsize=14, fontweight='bold')
    ax.set_title(f'Energy Spectrum Comparison: Loss Variants{title_suffix}', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Spectrum comparison plot complete")


def plot_side_by_side_spectra(
    spectra_dict_1,
    spectra_dict_2,
    label_1="Model 1",
    label_2="Model 2",
    variants=['bsp', 'log_bsp', 'sa_bsp', 'sa_log_bsp'],
    figsize=(20, 9)
):
    \"\"\"
    Plot two spectrum comparisons side-by-side.
    
    Args:
        spectra_dict_1: First spectra dictionary
        spectra_dict_2: Second spectra dictionary
        label_1: Label for first model
        label_2: Label for second model
        variants: List of variants to plot
        figsize: Figure size tuple
    \"\"\"
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = {
        'True': '#000000',
        'bsp': '#ff7f0e',
        'log_bsp': '#2ca02c',
        'sa_bsp': '#d62728',
        'sa_log_bsp': '#9467bd'
    }
    
    label_map = {
        'bsp': 'BSP (k²)',
        'log_bsp': 'Log-BSP',
        'sa_bsp': 'SA-BSP',
        'sa_log_bsp': 'SA-Log-BSP'
    }
    
    for ax, spectra, title in [(ax1, spectra_dict_1, label_1), (ax2, spectra_dict_2, label_2)]:
        # Plot ground truth
        if 'True' in spectra:
            data = spectra['True']
            ax.loglog(data['frequencies'], data['energy_median'], 
                     color=colors['True'], linewidth=3, label='True', zorder=10)
            ax.fill_between(data['frequencies'], data['energy_p16'], data['energy_p84'],
                           color=colors['True'], alpha=0.15, zorder=9)
        
        # Plot variants
        for variant in variants:
            if variant in spectra:
                data = spectra[variant]
                color = colors.get(variant, '#888888')
                label = label_map.get(variant, variant)
                
                ax.loglog(data['frequencies'], data['energy_median'],
                         color=color, linewidth=2.5, alpha=0.85, label=label, zorder=5)
                ax.fill_between(data['frequencies'], data['energy_p16'], data['energy_p84'],
                               color=color, alpha=0.12, zorder=4)
        
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('E(k)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Side-by-side spectrum comparison complete")


print("✓ Spectrum comparison utilities loaded")
print("\\nAvailable functions:")
print("  • compare_loss_variants(trained_models_dict, val_loader, loss_variants, ...)")
print("  • plot_spectrum_comparison(spectra, variants_to_plot, ...)")
print("  • plot_side_by_side_spectra(spectra_1, spectra_2, label_1, label_2, ...)")"""),

    # Cell 11: Markdown
    markdown_cell("""## Cell 3.9: Usage Example - Quick Loss Variant Comparison

**Example workflow for comparing loss variants:**

1. **Train with different variants** by changing `LOSS_VARIANT` in Cell 3.5:
   ```python
   # Train BSP
   LOSS_VARIANT = 'bsp'
   # Run training cells...
   
   # Train Log-BSP
   LOSS_VARIANT = 'log_bsp'
   # Run training cells...
   
   # Train SA-BSP
   LOSS_VARIANT = 'sa_bsp'
   SA_ADAPT_MODE = 'per-bin'
   # Run training cells...
   ```

2. **Compare spectra** using the utility functions:
   ```python
   # After training multiple variants, compare their spectra
   spectra = compare_loss_variants(
       trained_models, 
       val_loader, 
       loss_variants=['bsp', 'log_bsp', 'sa_bsp'],
       model_arch='deeponet'
   )
   
   # Plot comparison
   plot_spectrum_comparison(
       spectra, 
       variants_to_plot=['bsp', 'log_bsp', 'sa_bsp'],
       title_suffix=' - DeepONet Model'
   )
   ```

3. **Side-by-side comparison** for different model architectures:
   ```python
   # Compare FNO vs DeepONet with same loss variant
   plot_side_by_side_spectra(
       spectra_fno, 
       spectra_deeponet,
       label_1="FNO + BSP",
       label_2="DeepONet + BSP",
       variants=['bsp', 'log_bsp']
   )
   ```

**Quick switches:**
- Change `LOSS_VARIANT` to switch between 'bsp', 'log_bsp', 'sa_bsp', 'sa_log_bsp'
- Change `SA_ADAPT_MODE` for SA-BSP variants: 'per-bin', 'global', 'combined'
- Change `MODEL_ARCH` to try different architectures: 'deeponet', 'fno', 'unet'"""),

    # Cell 12: Code
    code_cell("""# ============================================================================
# NOTE: Configuration has been moved to Cell 2 (above)
# ============================================================================
# This cell is kept for backward compatibility but is no longer needed.
# All configuration (USE_CAUSAL_PADDING, DEEPONET_ACTIVATION, etc.) 
# is now defined in Cell 2 before data loading.

print("⚠️  This cell is deprecated - configuration is now in Cell 2")"""),

    # Cell 13: Code
    code_cell("""# Choose model architecture
MODEL_ARCH = 'deeponet'  # Options: 'deeponet', 'fno', 'unet'

# Create model with optional DeepONet activation
if MODEL_ARCH == 'deeponet':
    model = create_model(MODEL_ARCH, config={'activation': DEEPONET_ACTIVATION})
    print(f"✓ Created {MODEL_ARCH.upper()} model with {DEEPONET_ACTIVATION.upper()} activation")
else:
    model = create_model(MODEL_ARCH)
    print(f"✓ Created {MODEL_ARCH.upper()} model")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {num_params:,}")"""),

    # Cell 14: Markdown
    markdown_cell("""## Cell 3: Choose Model Architecture

**Change `MODEL_ARCH` to try different models:**
- `'deeponet'`: Branch-trunk architecture with SIREN activation
- `'fno'`: Fourier Neural Operator
- `'unet'`: U-Net encoder-decoder"""),

    # Cell 15: Code
    code_cell("""# Choose model architecture
MODEL_ARCH = 'deeponet'  # Options: 'deeponet', 'fno', 'unet'

# Create model with optional DeepONet activation
if MODEL_ARCH == 'deeponet':
    model = create_model(MODEL_ARCH, config={'activation': DEEPONET_ACTIVATION})
    print(f"✓ Created {MODEL_ARCH.upper()} model with {DEEPONET_ACTIVATION.upper()} activation")
else:
    model = create_model(MODEL_ARCH)
    print(f"✓ Created {MODEL_ARCH.upper()} model")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {num_params:,}")"""),

    # Cell 16: Markdown
    markdown_cell("""## Cell 4: Initialize Results Storage

We'll train with all 6 loss types sequentially and store results for comparison:
- **BASELINE**: Relative L2 loss only (MSE baseline)
- **BSP**: MSE + fixed BSP loss with k² weighting
- **Log-BSP**: MSE + BSP with log₁₀ spectral energies (uniform weighting)
- **SA-BSP-PERBIN**: MSE + 32 adaptive per-bin weights (negated gradients for frequency emphasis)
- **SA-BSP-GLOBAL**: 2 adaptive weights (w_mse + w_bsp) with negated gradients for MSE/BSP balance
- **SA-BSP-COMBINED**: 34 weights (w_mse + w_bsp + 32 per-bin) with full competitive dynamics"""),

    # Cell 17: Code
    code_cell("""# Import loss configurations
from configs.loss_config import (
    BASELINE_CONFIG, 
    BSP_CONFIG,
    LOG_BSP_CONFIG,
    SA_BSP_PERBIN_CONFIG,
    SA_BSP_GLOBAL_CONFIG,
    SA_BSP_COMBINED_CONFIG
)
from src.core.evaluation.loss_factory import create_loss

# Loss configuration map
loss_config_map = {
    'baseline': BASELINE_CONFIG,
    'bsp': BSP_CONFIG,
    'log-bsp': LOG_BSP_CONFIG,
    'sa-bsp-perbin': SA_BSP_PERBIN_CONFIG,
    'sa-bsp-global': SA_BSP_GLOBAL_CONFIG,
    'sa-bsp-combined': SA_BSP_COMBINED_CONFIG
}

# Storage dictionaries for results from all loss types
all_training_results = {}  # Key: f"{MODEL_ARCH}_{loss_type}"
all_trainers = {}
trained_models = {}

print("✓ Storage initialized for multi-loss training")
print("\\nWill train with 6 loss types:")
print("  1. BASELINE:", BASELINE_CONFIG.description)
print("  2. BSP:", BSP_CONFIG.description)
print("  3. LOG-BSP:", LOG_BSP_CONFIG.description)
print("  4. SA-BSP-PERBIN:", SA_BSP_PERBIN_CONFIG.description)
print("  5. SA-BSP-GLOBAL:", SA_BSP_GLOBAL_CONFIG.description)
print("  6. SA-BSP-COMBINED:", SA_BSP_COMBINED_CONFIG.description)"""),

    # Cell 18: Markdown
    markdown_cell("""## Cell 5: Sequential Training with All Loss Types

Train the same model architecture with all 6 loss functions sequentially:
1. **BASELINE** - Pure MSE baseline
2. **BSP** - Fixed spectral loss with k² weighting
3. **Log-BSP** - Spectral loss with log₁₀ energies and uniform weighting
4. **SA-BSP-PERBIN** - 32 adaptive weights (emphasize hard frequency bins)
5. **SA-BSP-GLOBAL** - 2 adaptive weights (learn MSE/BSP balance)
6. **SA-BSP-COMBINED** - 34 adaptive weights (full competitive dynamics)"""),

    # Cell 19: Code
    code_cell("""# Train with all 6 loss types sequentially
loss_types_to_train = ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']

for LOSS_TYPE in loss_types_to_train:
    print(f"\\n{'='*70}")
    print(f"Training {MODEL_ARCH.upper()} with {LOSS_TYPE.upper()} Loss")
    print(f"{'='*70}\\n")

    # Select loss configuration
    selected_loss_config = loss_config_map[LOSS_TYPE]
    print(f"Loss config: {selected_loss_config.description}")

    # Create loss function
    criterion = create_loss(selected_loss_config)
    print(f"✓ Loss function created: {type(criterion).__name__}")

    # Create FRESH model for this loss type (important!)
    model_for_loss = create_model(MODEL_ARCH)
    num_params = sum(p.numel() for p in model_for_loss.parameters() if p.requires_grad)
    print(f"✓ Fresh model created ({num_params:,} parameters)")

    # ========================================================================
    # AUTOMATIC LOADER SELECTION: DeepONet uses dual-batch for BSP losses
    # ========================================================================

    # Determine which loaders are needed based on model and loss
    use_dual_batch = (MODEL_ARCH == 'deeponet' and LOSS_TYPE != 'baseline')

    if use_dual_batch:
        # DeepONet + BSP/SA-BSP: DUAL-BATCH training
        # MSE uses per-timestep (320K samples), BSP uses sequences (80 samples)
        print(f"\\n📊 Creating dual-batch loaders (per-timestep + sequence)...")

        # Create per-timestep datasets for MSE component
        per_ts_train_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='train',
            normalize=normalizer,
            mode='per_timestep',
            use_causal_sequence=USE_CAUSAL_PADDING,
            signal_length=4000
        )
        per_ts_val_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='test',
            normalize=normalizer,
            mode='per_timestep',
            use_causal_sequence=USE_CAUSAL_PADDING,
            signal_length=4000
        )

        # Create sequence datasets for BSP component
        seq_train_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='train',
            normalize=normalizer,
            mode='sequence',
            use_causal_sequence=False,  # Sequences never use causal padding
            signal_length=4000
        )
        seq_val_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='test',
            normalize=normalizer,
            mode='sequence',
            use_causal_sequence=False,
            signal_length=4000
        )

        # Create per-timestep loaders
        per_ts_train_loader = DataLoader(
            per_ts_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        per_ts_val_loader = DataLoader(
            per_ts_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Create sequence loaders
        seq_train_loader = DataLoader(
            seq_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        seq_val_loader = DataLoader(
            seq_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"  ✓ Per-timestep train: {len(per_ts_train_dataset):,} samples")
        print(f"  ✓ Per-timestep val:   {len(per_ts_val_dataset):,} samples")
        print(f"  ✓ Sequence train:     {len(seq_train_dataset)} samples")
        print(f"  ✓ Sequence val:       {len(seq_val_dataset)} samples")

    elif MODEL_ARCH == 'deeponet' and LOSS_TYPE == 'baseline':
        # DeepONet + baseline: PER-TIMESTEP only (MSE only, no BSP)
        print(f"\\n📊 Creating per-timestep loaders (MSE-only baseline)...")

        per_ts_train_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='train',
            normalize=normalizer,
            mode='per_timestep',
            use_causal_sequence=USE_CAUSAL_PADDING,
            signal_length=4000
        )
        per_ts_val_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='test',
            normalize=normalizer,
            mode='per_timestep',
            use_causal_sequence=USE_CAUSAL_PADDING,
            signal_length=4000
        )

        per_ts_train_loader = DataLoader(
            per_ts_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        per_ts_val_loader = DataLoader(
            per_ts_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"  ✓ Per-timestep train: {len(per_ts_train_dataset):,} samples")
        print(f"  ✓ Per-timestep val:   {len(per_ts_val_dataset):,} samples")

    else:
        # FNO/UNet: SEQUENCE only (all losses)
        print(f"\\n📊 Creating sequence loaders ({MODEL_ARCH.upper()} architecture)...")

        seq_train_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='train',
            normalize=normalizer,
            mode='sequence',
            use_causal_sequence=False,
            signal_length=4000
        )
        seq_val_dataset = CDONDataset(
            data_dir=str(DATA_DIR),
            split='test',
            normalize=normalizer,
            mode='sequence',
            use_causal_sequence=False,
            signal_length=4000
        )

        seq_train_loader = DataLoader(
            seq_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        seq_val_loader = DataLoader(
            seq_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"  ✓ Sequence train: {len(seq_train_dataset)} samples")
        print(f"  ✓ Sequence val:   {len(seq_val_dataset)} samples")

    # ========================================================================
    # CREATE TRAINER with appropriate loader API
    # ========================================================================

    # Create training config
    optimizer_type = 'adam' if MODEL_ARCH == 'fno' else 'soap'

    config = TrainingConfig(
        num_epochs=50,
        learning_rate=1e-3,
        optimizer_type=optimizer_type,
        batch_size=BATCH_SIZE,
        weight_decay=1e-4,
        scheduler_type='cosine',
        cosine_eta_min=1e-6,
        eval_metrics=['field_error', 'spectrum_error'],
        eval_frequency=1,
        checkpoint_dir=f'checkpoints/{MODEL_ARCH}_{LOSS_TYPE}',
        save_best=False,
        save_latest=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_workers=2,
        verbose=True
    )

    # Create trainer with appropriate loader API
    if use_dual_batch:
        # Dual-batch API: pass both per-timestep and sequence loaders
        trainer = SimpleTrainer(
            model=model_for_loss,
            per_timestep_train_loader=per_ts_train_loader,
            sequence_train_loader=seq_train_loader,
            per_timestep_val_loader=per_ts_val_loader,
            sequence_val_loader=seq_val_loader,
            config=config,
            loss_config=selected_loss_config,
            experiment_name=f'{MODEL_ARCH}_{LOSS_TYPE}'
        )
        print(f"\\n✓ Trainer initialized with DUAL-BATCH mode")

    elif MODEL_ARCH == 'deeponet' and LOSS_TYPE == 'baseline':
        # Baseline: Use simplified API (maps train_loader to sequence_train_loader internally)
        trainer = SimpleTrainer(
            model=model_for_loss,
            train_loader=per_ts_train_loader,    # Use simplified API
            val_loader=per_ts_val_loader,        # Maps to sequence loaders internally
            config=config,
            loss_config=selected_loss_config,
            experiment_name=f'{MODEL_ARCH}_{LOSS_TYPE}'
        )
        print(f"\\n✓ Trainer initialized with PER-TIMESTEP mode (via simplified API)")

    else:
        # Sequence-only API
        trainer = SimpleTrainer(
            model=model_for_loss,
            train_loader=seq_train_loader,
            val_loader=seq_val_loader,
            config=config,
            loss_config=selected_loss_config,
            experiment_name=f'{MODEL_ARCH}_{LOSS_TYPE}'
        )
        print(f"\\n✓ Trainer initialized with SEQUENCE mode")

    print(f"  Device: {trainer.device}")
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")

    # Check for weight optimizer (SA-BSP variants only)
    if 'sa-bsp' in LOSS_TYPE:
        if trainer.weight_optimizer is not None:
            adapt_mode = trainer.adapt_mode
            print(f"  Weight optimizer: ✓ Created for SA-BSP ({adapt_mode} mode)")
        else:
            print(f"  ⚠ WARNING: SA-BSP but no weight_optimizer!")

    print(f"\\n🚀 Starting training...\\n")

    # Train
    results = trainer.train()

    # Store results
    key = f"{MODEL_ARCH}_{LOSS_TYPE}"
    all_training_results[key] = results
    all_trainers[key] = trainer
    trained_models[key] = model_for_loss

    print(f"\\n✅ {LOSS_TYPE.upper()} training complete!")
    print(f"   Best val loss: {results['best_val_loss']:.6f}")
    print(f"   Final val loss: {results['val_history'][-1]['loss']:.6f}")

print(f"\\n{'='*70}")
print(f"ALL TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"Trained {len(all_training_results)} models with different loss functions")"""),

    # Cell 20: Markdown
    markdown_cell("""## Cell 6: Multi-Loss Training Comparison

Compare training metrics across all 6 loss functions."""),

    # Cell 21: Code
    code_cell("""# Create multi-loss comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Color scheme for loss types
colors = {
    'baseline': '#1f77b4',       # Blue
    'bsp': '#ff7f0e',             # Orange
    'log-bsp': '#2ca02c',         # Green
    'sa-bsp-perbin': '#d62728',   # Red
    'sa-bsp-global': '#9467bd',   # Purple
    'sa-bsp-combined': '#17becf'  # Cyan
}
linestyles = {
    'baseline': '-', 
    'bsp': '--', 
    'log-bsp': '-.', 
    'sa-bsp-perbin': ':', 
    'sa-bsp-global': '-',
    'sa-bsp-combined': '--'
}
markers = {
    'baseline': 'o', 
    'bsp': 's', 
    'log-bsp': '^', 
    'sa-bsp-perbin': 'D', 
    'sa-bsp-global': 'v',
    'sa-bsp-combined': 'p'
}

for loss_type in ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']:
    key = f"{MODEL_ARCH}_{loss_type}"
    results = all_training_results[key]
    
    # Extract metrics
    val_losses = [h['loss'] for h in results['val_history']]
    val_field_errors = [h['field_error'] for h in results['val_history']]
    val_spectrum_errors = [h['spectrum_error'] for h in results['val_history']]
    epochs = range(1, len(val_losses) + 1)
    
    # Create label with short name
    label_map = {
        'baseline': 'BASELINE',
        'bsp': 'BSP',
        'log-bsp': 'Log-BSP',
        'sa-bsp-perbin': 'SA-BSP (Per-bin)',
        'sa-bsp-global': 'SA-BSP (Global)',
        'sa-bsp-combined': 'SA-BSP (Combined)'
    }
    label = label_map[loss_type]
    
    # Plot on all 3 axes
    axes[0].plot(epochs, val_losses, label=label, 
                color=colors[loss_type], linestyle=linestyles[loss_type],
                linewidth=2, alpha=0.9, marker=markers[loss_type], markersize=4, markevery=5)
    
    axes[1].plot(epochs, val_field_errors, label=label,
                color=colors[loss_type], linestyle=linestyles[loss_type],
                linewidth=2, alpha=0.9, marker=markers[loss_type], markersize=4, markevery=5)
    
    axes[2].plot(epochs, val_spectrum_errors, label=label,
                color=colors[loss_type], linestyle=linestyles[loss_type],
                linewidth=2, alpha=0.9, marker=markers[loss_type], markersize=4, markevery=5)

# Configure axes with LOG SCALE on y-axis
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Validation Loss', fontsize=12)
axes[0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')  # LOG SCALE
axes[0].set_ylim(bottom=1e-5, top=1.0)  # Clip for readability
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.3, which='both')

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Field Error', fontsize=12)
axes[1].set_title('Field Error (Real Space)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')  # LOG SCALE
axes[1].set_ylim(bottom=1e-5, top=1.0)  # Clip for readability
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.3, which='both')

axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Spectrum Error', fontsize=12)
axes[2].set_title('Spectrum Error (Frequency Space)', fontsize=14, fontweight='bold')
axes[2].set_yscale('log')  # LOG SCALE
axes[2].set_ylim(bottom=1e-5, top=1.0)  # Clip for readability
axes[2].legend(fontsize=9, loc='best')
axes[2].grid(True, alpha=0.3, which='both')

plt.suptitle(f'{MODEL_ARCH.upper()}: Loss Function Comparison (6 Variants)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print final metrics table
print(f"\\n{'='*70}")
print("Final Metrics Summary")
print(f"{'='*70}")
print(f"{'Loss Type':<25} {'Val Loss':<12} {'Field Error':<15} {'Spectrum Error':<15}")
print("-"*70)

for loss_type in ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']:
    key = f"{MODEL_ARCH}_{loss_type}"
    results = all_training_results[key]
    final_val = results['val_history'][-1]
    
    label = loss_type.upper()
    print(f"{label:<25} {final_val['loss']:<12.6f} "
          f"{final_val['field_error']:<15.6f} {final_val['spectrum_error']:<15.6f}")"""),

    # Cell 22: Markdown
    markdown_cell("""## Cell 7: Spectral Bias Visualization (Energy Spectrum)

Visualize E(k) vs wavenumber to identify spectral bias in trained models."""),

    # Cell 23: Code
    code_cell("""import torch.fft as fft
from src.core.visualization.spectral_analysis import compute_unbinned_spectrum, compute_cached_true_spectrum
from configs.visualization_config import SPECTRUM_CACHE_FILENAME, CACHE_DIR

# Get validation batch for energy spectrum analysis
print("Computing energy spectra for all trained models...")

# Check if models have been trained
if 'trained_models' not in globals() or len(trained_models) == 0:
    print("\\n⚠️  WARNING: No trained models found!")
    print("   Please run Cell 12 (training) first before running this cell.")
    print("   This cell requires the 'trained_models' dictionary to be populated.\\n")
else:
    print(f"✓ Found {len(trained_models)} trained models")
    print(f"  Keys: {list(trained_models.keys())}\\n")

val_batch_input, val_batch_target = next(iter(val_loader))

# Move to device for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_batch_input = val_batch_input.to(device)
val_batch_target = val_batch_target.to(device)

# Compute ground truth spectrum with percentile-based uncertainty bands from cache
cache_path = f'{CACHE_DIR}/{SPECTRUM_CACHE_FILENAME}'
print(f"Loading true spectrum from cache: {cache_path}")
cached = np.load(cache_path)
k_true = cached['unbinned_frequencies']  # Full FFT resolution (~2000 frequencies)
E_true_median = cached['unbinned_energy_median']  # Median (50th percentile)
E_true_p16 = cached['unbinned_energy_p16']        # Lower bound (16th percentile ≈ -1σ)
E_true_p84 = cached['unbinned_energy_p84']        # Upper bound (84th percentile ≈ +1σ)
print(f"✓ True spectrum loaded ({len(k_true)} frequencies, unbinned)")
print(f"  Using percentile-based uncertainty bands (16th-84th ≈ ±1σ)")

# Collect ALL validation predictions for uncertainty bands
print("\\nComputing unbinned spectra with percentile-based uncertainty bands for all models...")
spectra = {}

# Store true spectrum with percentile uncertainty bounds
spectra['True'] = {
    'frequencies': k_true,
    'energy_median': E_true_median,
    'energy_p16': E_true_p16,
    'energy_p84': E_true_p84
}

for loss_type in ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']:
    key = f"{MODEL_ARCH}_{loss_type}"
    
    # Check if model exists
    if key not in trained_models:
        print(f"  ⚠️  Skipping {loss_type.upper()}: model key '{key}' not found in trained_models")
        continue
    
    model_trained = trained_models[key]
    model_trained.eval()
    model_trained.to(device)
    
    try:
        # Collect predictions from ALL validation batches for uncertainty bands
        all_preds = []
        print(f"  Processing {loss_type.upper()}...", end='')
        
        with torch.no_grad():
            for val_input, _ in val_loader:
                val_input = val_input.to(device)
                pred = model_trained(val_input)
                all_preds.append(pred.cpu())
        
        # Stack all predictions: [total_val_samples, C, T]
        all_preds_tensor = torch.cat(all_preds, dim=0)
        
        # Compute unbinned spectrum with percentile-based uncertainty bands
        k_pred, E_pred_median, E_pred_p16, E_pred_p84 = compute_unbinned_spectrum(all_preds_tensor)
        
        # Create display label
        label_map = {
            'baseline': 'BASELINE',
            'bsp': 'BSP',
            'log-bsp': 'Log-BSP',
            'sa-bsp-perbin': 'SA-BSP (Per-bin)',
            'sa-bsp-global': 'SA-BSP (Global)',
            'sa-bsp-combined': 'SA-BSP (Combined)'
        }
        spec_key = f"{MODEL_ARCH.upper()} + {label_map[loss_type]}"
        
        spectra[spec_key] = {
            'frequencies': k_pred,
            'energy_median': E_pred_median,
            'energy_p16': E_pred_p16,
            'energy_p84': E_pred_p84
        }
        
        print(f" ✓ ({all_preds_tensor.shape[0]} samples)")
    except Exception as e:
        print(f" ❌ Error: {e}")
        continue

print(f"\\n✓ Spectra computed for {len(spectra)} entries with percentile-based uncertainty bands\\n")

# Plot energy spectrum with percentile-based uncertainty bands (safe for log scale!)
fig, ax = plt.subplots(figsize=(14, 9))

# Color scheme for loss types
colors_plot = {
    'True': '#000000',  # Black for ground truth
    'baseline': '#1f77b4',
    'bsp': '#ff7f0e',
    'log-bsp': '#2ca02c',
    'sa-bsp-perbin': '#d62728',
    'sa-bsp-global': '#9467bd',
    'sa-bsp-combined': '#17becf'
}

# Plot ground truth with uncertainty band (black)
if 'True' in spectra:
    data = spectra['True']
    k = data['frequencies']
    E_median = data['energy_median']
    E_p16 = data['energy_p16']
    E_p84 = data['energy_p84']
    
    # Plot median line
    ax.loglog(k, E_median, color=colors_plot['True'], linewidth=3, 
             label='True (Real Data)', zorder=10, alpha=0.9)
    
    # Plot percentile-based uncertainty band (16th-84th percentiles ≈ ±1σ)
    # These are GUARANTEED to be positive → safe for log scale!
    ax.fill_between(k, E_p16, E_p84,
                     color=colors_plot['True'], alpha=0.15, zorder=9,
                     label='True (16th-84th percentile)')

# Plot model predictions with percentile-based uncertainty bands
for loss_type in ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']:
    label_map = {
        'baseline': 'BASELINE',
        'bsp': 'BSP',
        'log-bsp': 'Log-BSP',
        'sa-bsp-perbin': 'SA-BSP (Per-bin)',
        'sa-bsp-global': 'SA-BSP (Global)',
        'sa-bsp-combined': 'SA-BSP (Combined)'
    }
    label_key = f"{MODEL_ARCH.upper()} + {label_map[loss_type]}"
    
    # Check if spectrum exists before plotting
    if label_key not in spectra:
        print(f"  ⚠️  Skipping plot for {loss_type.upper()}: '{label_key}' not in spectra dictionary")
        continue
    
    data = spectra[label_key]
    k = data['frequencies']
    E_median = data['energy_median']
    E_p16 = data['energy_p16']
    E_p84 = data['energy_p84']
    
    color = colors_plot[loss_type]
    
    # Plot median line
    ax.loglog(k, E_median, color=color, linewidth=2.5, 
             alpha=0.85, label=label_key, zorder=5)
    
    # Plot percentile-based uncertainty band (guaranteed positive for log scale)
    ax.fill_between(k, E_p16, E_p84,
                     color=color, alpha=0.12, zorder=4)

# Configure plot
ax.set_xlabel('Frequency (normalized)', fontsize=14, fontweight='bold')
ax.set_ylabel('E(k) - Spectral Power', fontsize=14, fontweight='bold')
ax.set_title(f'Energy Spectrum Comparison with Percentile Uncertainty Bands\\n{MODEL_ARCH.upper()} Model (6 Loss Variants)', 
            fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='best', framealpha=0.95, ncol=1)
ax.grid(True, alpha=0.3, which='both', linestyle='--')

# Set frequency range to 0-25Hz (full dataset range)
ax.set_xlim(0, 25.0)

plt.tight_layout()
plt.show()

print(f"\\n✓ Energy spectrum plot complete")
print(f"  • Unbinned spectrum: Full FFT resolution (~{len(k_true)} frequencies)")
print(f"  • Uncertainty bands: 16th-84th percentiles (≈ ±1σ) across all validation samples")
print(f"  • Percentiles are ALWAYS positive → safe for log-scale display!")
print(f"  • This visualization shows spectral bias: deviation from ground truth at high frequencies")
print(f"  • Log-BSP and SA-BSP variants should show better high-frequency matching than baseline")"""),

    # Cell 24: Markdown
    markdown_cell("""## Cell 8: Spectral Bias Quantification

Compute spectral bias metrics to quantify how well each model captures high-frequency content."""),

    # Cell 25: Code
    code_cell("""from src.core.visualization.spectral_analysis import compute_spectral_bias_metric

print("="*70)
print("SPECTRAL BIAS METRICS")
print("="*70)
print("\\nQuantifies how well each model captures different frequency ranges.")
print("Spectral Bias Ratio = High Freq Error / Low Freq Error")
print("  - Ratio > 2.0: Significant spectral bias (struggles with high frequencies)")
print("  - Ratio > 1.5: Moderate spectral bias")
print("  - Ratio ≤ 1.5: Low spectral bias (captures frequencies well)")
print("="*70)

# Compute metrics for each trained model
spectral_metrics = {}

for loss_type in ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']:
    key = f"{MODEL_ARCH}_{loss_type}"
    model_trained = trained_models[key]
    model_trained.eval()
    model_trained.to(device)
    
    with torch.no_grad():
        pred = model_trained(val_input)
    
    metrics = compute_spectral_bias_metric(pred.cpu(), val_target.cpu(), n_bins=32)
    spectral_metrics[loss_type] = metrics
    
    label_map = {
        'baseline': 'BASELINE',
        'bsp': 'BSP',
        'log-bsp': 'Log-BSP',
        'sa-bsp-perbin': 'SA-BSP (Per-bin)',
        'sa-bsp-global': 'SA-BSP (Global)',
        'sa-bsp-combined': 'SA-BSP (Combined)'
    }
    
    print(f"\\n{MODEL_ARCH.upper()} + {label_map[loss_type]}:")
    print(f"  Low frequency error:   {metrics['low_freq_error']:.6f}")
    print(f"  Mid frequency error:   {metrics['mid_freq_error']:.6f}")
    print(f"  High frequency error:  {metrics['high_freq_error']:.6f}")
    print(f"  Spectral bias ratio:   {metrics['spectral_bias_ratio']:.4f}")
    
    # Interpretation
    if metrics['spectral_bias_ratio'] > 2.0:
        print(f"  → ⚠️  SIGNIFICANT spectral bias detected!")
        print(f"     Model struggles with high-frequency content")
    elif metrics['spectral_bias_ratio'] > 1.5:
        print(f"  → ⚡ MODERATE spectral bias")
        print(f"     Some difficulty with high frequencies")
    else:
        print(f"  → ✅ LOW spectral bias")
        print(f"     Model captures frequency content well")

# Create comparison visualization
print(f"\\n{'='*70}")
print("Spectral Bias Comparison")
print(f"{'='*70}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Bar plot 1: Frequency errors
loss_types = ['baseline', 'bsp', 'log-bsp', 'sa-bsp-perbin', 'sa-bsp-global', 'sa-bsp-combined']
x = np.arange(len(loss_types))
width = 0.2

low_errors = [spectral_metrics[lt]['low_freq_error'] for lt in loss_types]
mid_errors = [spectral_metrics[lt]['mid_freq_error'] for lt in loss_types]
high_errors = [spectral_metrics[lt]['high_freq_error'] for lt in loss_types]

ax1.bar(x - width, low_errors, width, label='Low Freq', color='#2ca02c', alpha=0.8)
ax1.bar(x, mid_errors, width, label='Mid Freq', color='#ff7f0e', alpha=0.8)
ax1.bar(x + width, high_errors, width, label='High Freq', color='#d62728', alpha=0.8)

ax1.set_xlabel('Loss Type', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency Error', fontsize=12, fontweight='bold')
ax1.set_title('Frequency Range Errors', fontsize=14, fontweight='bold')
ax1.set_yscale('log')  # LOG SCALE
ax1.set_xticks(x)
ax1.set_xticklabels(['BASE', 'BSP', 'Log-BSP', 'SA-Per', 'SA-Glob', 'SA-Comb'], rotation=15, ha='right')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y', which='both')

# Bar plot 2: Spectral bias ratio
bias_ratios = [spectral_metrics[lt]['spectral_bias_ratio'] for lt in loss_types]
colors_bars = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

bars = ax2.bar(x, bias_ratios, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add threshold lines
ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Significant bias threshold')
ax2.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate bias threshold')

ax2.set_xlabel('Loss Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('Spectral Bias Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Spectral Bias Ratio (High/Low)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['BASE', 'BSP', 'Log-BSP', 'SA-Per', 'SA-Glob', 'SA-Comb'], rotation=15, ha='right')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, ratio) in enumerate(zip(bars, bias_ratios)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{ratio:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle(f'{MODEL_ARCH.upper()}: Spectral Bias Analysis (6 Loss Variants)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\\n{'='*70}")
print("✅ Spectral bias analysis complete!")
print(f"{'='*70}")
print("\\nKey Findings:")
print("  • Baseline: Pure MSE - typically shows significant spectral bias")
print("  • BSP: Fixed spectral loss with k² weighting - moderate improvement")
print("  • Log-BSP: Log-domain spectral loss - addresses wide dynamic range")
print("  • SA-BSP (Per-bin): Adaptive per-bin weights - emphasize hard frequencies")
print("  • SA-BSP (Global): Adaptive MSE/BSP balance - optimize overall trade-off")
print("  • SA-BSP (Combined): Full competitive dynamics - most expressive approach")"""),

    # Cell 26: Markdown
    markdown_cell("""## Summary

This notebook demonstrated:
1. ✓ Loading real CDON data with proper normalization
2. ✓ Creating neural operator models (DeepONet, FNO, UNet)
3. ✓ **Sequential training with all 6 loss functions**:
   - **BASELINE**: Relative L2 loss only (MSE baseline)
   - **BSP**: MSE + fixed BSP loss with k² weighting
   - **Log-BSP**: MSE + BSP with log₁₀ spectral energies (uniform weighting)
   - **SA-BSP (Per-bin)**: MSE + 32 adaptive per-bin weights (negated gradients for frequency emphasis)
   - **SA-BSP (Global)**: MSE + 2 adaptive weights (w_mse + w_bsp, negated gradients for MSE/BSP balance)
   - **SA-BSP (Combined)**: MSE + 34 weights (w_mse + w_bsp + 32 per-bin, all negated gradients for full competitive dynamics)
4. ✓ **Multi-loss comparison plots** showing training metrics
5. ✓ **Energy spectrum visualization** (E(k) vs wavenumber) to identify spectral bias
6. ✓ **Spectral bias quantification** with metrics and comparison plots

**Key Results:**
- All 6 loss types trained on the same model architecture
- Direct comparison shows which loss function best mitigates spectral bias
- Energy spectrum plot reveals how well each model captures high-frequency content
- Quantitative metrics identify spectral bias ratio for each approach

**SA-PINNs Implementation:**
- **Per-bin mode**: Uses negated gradients (ascent) to emphasize difficult frequency bins
- **Global mode**: Uses negated gradients (ascent) to learn optimal MSE/BSP balance via competitive dynamics
- **Combined mode**: Full competitive dynamics with all weights (w_mse, w_bsp, and 32 per-bin) using negated gradients

**Experiment with different configurations:**
- **Cell 0**: Run to force reload modules after code changes
- **Cell 3**: Change `MODEL_ARCH` to try different models ('deeponet', 'fno', 'unet')
- **Cell 5**: Adjust hyperparameters (epochs, learning rate, etc.) in TrainingConfig
- Run all cells sequentially to train and compare all 6 loss types automatically!"""),

]


def main():
    """Generate the notebook."""
    # Notebook structure
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Write notebook
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "notebooks" / "demo_training.ipynb"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"✅ Generated notebook: {output_path}")
    print(f"   Total cells: {len(CELLS)}")
    markdown_count = sum(1 for c in CELLS if c.get('cell_type') == 'markdown')
    code_count = sum(1 for c in CELLS if c.get('cell_type') == 'code')
    print(f"   Markdown: {markdown_count}")
    print(f"   Code: {code_count}")


if __name__ == "__main__":
    main()