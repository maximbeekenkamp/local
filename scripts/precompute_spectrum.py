#!/usr/bin/env python3
"""
Precompute and cache the true frequency spectrum for the full dataset.

This script:
1. Loads the CDON dataset (train + val + test - ALL available data)
2. Computes the true frequency spectrum across all samples
3. Caches it for use in analysis

The cached spectrum can then be committed to git to avoid recomputation.

Note: Uses ALL data for smoothest "ground truth" visualization.
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
import numpy as np


def precompute_target_spectra(
    data_dir: Path,
    stats_path: Path,
    cache_dir: Path,
    n_bins: int = 32,
    signal_length: int = 4000,
    use_log: bool = False
):
    """
    Precompute binned target spectra for all dataset samples.

    Creates cache for BSP loss to avoid recomputing targets every forward pass.

    Args:
        data_dir: Path to CDON data
        stats_path: Path to normalization stats
        cache_dir: Where to save cache
        n_bins: Number of frequency bins
        signal_length: Signal length
        use_log: If True, compute log₁₀(energy) for Log-BSP
    """
    from src.core.evaluation.binned_spectral_loss import BinnedSpectralLoss

    print(f"\n{'='*70}")
    print(f"Precomputing Per-Sample Target Spectra (use_log={use_log})")
    print(f"{'='*70}")

    # Load dataset (sequence mode for full sequences)
    normalizer = CDONNormalization(stats_path=str(stats_path))

    splits = ['train', 'test']  # val is same as test for CDON
    all_targets = []
    all_split_sizes = []

    for split in splits:
        print(f"\n📥 Processing {split} split...")
        dataset = CDONDataset(
            data_dir=str(data_dir),
            split=split,
            normalize=normalizer,
            mode='sequence',  # Full sequences
            use_causal_sequence=False,  # BSP uses non-causal sequences
            signal_length=signal_length
        )

        print(f"   Samples: {len(dataset)}")

        # Create temporary BSP loss for preprocessing
        temp_loss = BinnedSpectralLoss(
            n_bins=n_bins,
            signal_length=signal_length,
            cache_path=str(cache_dir / 'true_spectrum.npz'),  # For bin edges
            use_log=use_log,
            use_output_norm=True,
            use_minmax_norm=True,
            lambda_k_mode='uniform',  # Doesn't affect preprocessing
            loss_type='l2_norm',
            epsilon=1e-8
        )

        # Process all samples
        split_targets = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if len(sample) == 3:
                _, target, _ = sample  # (input, target, sample_idx)
            else:
                _, target = sample  # Fallback for old format
            target = target.unsqueeze(0)  # [1, 1, 4000] for batch processing

            with torch.no_grad():
                # Use preprocessing method from BinnedSpectralLoss
                target_binned = temp_loss._preprocess_target_only(target)

            split_targets.append(target_binned.cpu().numpy())

        all_targets.extend(split_targets)
        all_split_sizes.append(len(split_targets))
        print(f"   ✓ Processed {len(split_targets)} samples")

    # Stack and save
    all_targets_array = np.concatenate(all_targets, axis=0)  # [N, 1, n_bins]

    cache_name = 'log' if use_log else 'linear'
    cache_path = cache_dir / f'target_spectra_{cache_name}.npz'

    np.savez(
        cache_path,
        target_binned=all_targets_array,
        n_bins=n_bins,
        use_log=use_log,
        signal_length=signal_length,
        split_sizes=all_split_sizes,  # [train_size, test_size]
    )

    print(f"\n✅ Saved to {cache_path}")
    print(f"   Shape: {all_targets_array.shape}")
    print(f"   Size: {cache_path.stat().st_size / 1024:.1f} KB")
    print(f"   Train samples: {all_split_sizes[0]}")
    print(f"   Test samples: {all_split_sizes[1]}")

    return cache_path


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

    # Load full dataset (train + val + test - ALL available data)
    print(f"📥 Loading CDON dataset (all splits)...")
    train_dataset = CDONDataset(
        data_dir=str(DATA_DIR),
        split='train',
        normalize=normalizer,
        mode='sequence',  # Use full sequences for spectrum computation
        use_causal_sequence=False,
        signal_length=4000
    )

    val_dataset = CDONDataset(
        data_dir=str(DATA_DIR),
        split='val',
        normalize=normalizer,
        mode='sequence',
        use_causal_sequence=False,
        signal_length=4000
    )

    test_dataset = CDONDataset(
        data_dir=str(DATA_DIR),
        split='test',
        normalize=normalizer,
        mode='sequence',
        use_causal_sequence=False,
        signal_length=4000
    )

    # Combine ALL datasets for smoothest ground truth spectrum
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Total samples: {len(full_dataset)} (using ALL data for ground truth)")

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

    for batch in loader:
        if len(batch) == 3:
            inputs, targets, indices = batch
        else:
            inputs, targets = batch  # Fallback for old format
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

    # Precompute per-sample target spectra for BSP loss caching
    print(f"\n{'='*70}")
    print("Precomputing Per-Sample Target Spectra for BSP Loss")
    print(f"{'='*70}")
    print("This creates caches to avoid recomputing target spectra during training.")
    print("Two caches are created:")
    print("  1. Linear energy (for BSP, SA-BSP variants)")
    print("  2. Log energy (for Log-BSP variants)")

    cache_dir_path = project_root / CACHE_DIR

    # Linear energy (for BSP, SA-BSP variants)
    precompute_target_spectra(
        data_dir=DATA_DIR,
        stats_path=STATS_PATH,
        cache_dir=cache_dir_path,
        n_bins=bsp_n_bins,
        signal_length=4000,
        use_log=False
    )

    # Log energy (for Log-BSP variants)
    precompute_target_spectra(
        data_dir=DATA_DIR,
        stats_path=STATS_PATH,
        cache_dir=cache_dir_path,
        n_bins=bsp_n_bins,
        signal_length=4000,
        use_log=True
    )

    print(f"\n{'='*70}")
    print("✅ ALL CACHES GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  1. {CACHE_PATH} (true spectrum for visualization)")
    print(f"  2. {cache_dir_path / 'target_spectra_linear.npz'} (for BSP, SA-BSP)")
    print(f"  3. {cache_dir_path / 'target_spectra_log.npz'} (for Log-BSP)")
    print(f"\nYou can now:")
    print(f"  - git add cache/*.npz")
    print(f"  - Run training with cached target spectra")


if __name__ == '__main__':
    main()
