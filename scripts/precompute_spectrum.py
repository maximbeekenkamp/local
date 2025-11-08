#!/usr/bin/env python3
"""
Precompute and cache the true frequency spectrum for the full dataset.

This script:
1. Loads the CDON dataset (train + val)
2. Computes the true frequency spectrum across all samples
3. Caches it for use in analysis

The cached spectrum can then be committed to git to avoid recomputation.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from src.core.visualization.spectral_analysis import compute_cached_true_spectrum
from configs.visualization_config import SPECTRUM_CACHE_FILENAME, CACHE_DIR
from configs.loss_config import BSP_CONFIG


def main():
    print("="*70)
    print("Precomputing True Frequency Spectrum")
    print("="*70)

    # Paths
    DATA_DIR = project_root / 'CDONData'
    STATS_PATH = project_root / 'configs' / 'cdon_stats.json'
    CACHE_PATH = project_root / CACHE_DIR / SPECTRUM_CACHE_FILENAME

    print(f"\nData directory: {DATA_DIR}")
    print(f"Stats file: {STATS_PATH}")
    print(f"Cache output: {CACHE_PATH}")

    # Load normalization
    print(f"\n📥 Loading normalization stats...")
    normalizer = CDONNormalization(stats_path=str(STATS_PATH))

    # Load full dataset (train + val)
    print(f"📥 Loading CDON dataset...")
    train_dataset = CDONDataset(
        data_dir=str(DATA_DIR),
        split='train',
        normalize=normalizer
    )

    val_dataset = CDONDataset(
        data_dir=str(DATA_DIR),
        split='test',
        normalize=normalizer
    )

    # Combine datasets for full spectrum
    full_dataset = ConcatDataset([train_dataset, val_dataset])

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Total samples: {len(full_dataset)}")

    # Create dataloader
    loader = DataLoader(
        full_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Load all data into memory
    print(f"\n💾 Loading all data into memory...")
    all_targets = []

    for inputs, targets in loader:
        all_targets.append(targets)

    all_targets = torch.cat(all_targets, dim=0)
    print(f"   Loaded shape: {all_targets.shape}")

    # Get BSP n_bins for training
    bsp_n_bins = BSP_CONFIG.loss_params['n_bins']

    # Compute and cache spectrum (unbinned for viz + binned for BSP)
    print(f"\n⚙️  Computing spectra...")
    print(f"   Unbinned: Full FFT resolution (~2000 frequencies) for visualization")
    print(f"   Binned: {bsp_n_bins} bins for BSP loss training")
    freq, energy = compute_cached_true_spectrum(
        data=all_targets,
        cache_path=str(CACHE_PATH),
        n_bins=bsp_n_bins,
        force_recompute=True
    )

    print(f"\n✅ Success!")
    print(f"   Unbinned spectrum shape: {energy.shape} (visualization)")
    print(f"   Frequency range: [{freq[0]:.6f}, {freq[-1]:.6f}]")
    print(f"   Energy range: [{energy.min():.6e}, {energy.max():.6e}]")
    print(f"\n💾 Cached to: {CACHE_PATH}")
    print(f"\nYou can now git add cache/{SPECTRUM_CACHE_FILENAME}")


if __name__ == '__main__':
    main()
