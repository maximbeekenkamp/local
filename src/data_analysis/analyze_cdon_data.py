"""
CDON Dataset Analysis and Normalization

This module provides functions to analyze the CDON earthquake dataset,
compute normalization statistics, and generate analysis reports.
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Any


def load_cdon_data(data_dir: str) -> Dict[str, np.ndarray]:
    """
    Load CDON dataset from directory.

    Args:
        data_dir: Path to directory containing .npy files

    Returns:
        Dictionary with keys: 'train_loads', 'train_responses', 'test_loads', 'test_responses'

    Raises:
        FileNotFoundError: If any required file is missing
        ValueError: If shapes are inconsistent
    """
    required_files = [
        'train_Loads.npy',
        'train_Responses.npy',
        'test_Loads.npy',
        'test_Responses.npy'
    ]

    # Check all files exist
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")

    # Load all files
    train_loads = np.load(os.path.join(data_dir, 'train_Loads.npy'))
    train_responses = np.load(os.path.join(data_dir, 'train_Responses.npy'))
    test_loads = np.load(os.path.join(data_dir, 'test_Loads.npy'))
    test_responses = np.load(os.path.join(data_dir, 'test_Responses.npy'))

    # Validate shapes
    if train_loads.shape != train_responses.shape:
        raise ValueError(
            f"Train shape mismatch: loads {train_loads.shape} vs responses {train_responses.shape}"
        )

    if test_loads.shape != test_responses.shape:
        raise ValueError(
            f"Test shape mismatch: loads {test_loads.shape} vs responses {test_responses.shape}"
        )

    if train_loads.shape[1] != test_loads.shape[1]:
        raise ValueError(
            f"Time dimension mismatch: train {train_loads.shape[1]} vs test {test_loads.shape[1]}"
        )

    return {
        'train_loads': train_loads,
        'train_responses': train_responses,
        'test_loads': test_loads,
        'test_responses': test_responses
    }


def compute_per_sample_statistics(data: np.ndarray, name: str) -> Dict[str, Any]:
    """
    Compute statistics for each sample in dataset.

    For each sample, computes statistics across the time dimension.

    Args:
        data: Array of shape [n_samples, n_timesteps]
        name: Identifier string (e.g., 'train_loads')

    Returns:
        Dictionary containing:
            - 'name': str
            - 'n_samples': int
            - 'n_timesteps': int
            - 'means': array of per-sample means
            - 'stds': array of per-sample stds
            - 'mins': array of per-sample mins
            - 'maxs': array of per-sample maxs
            - 'energies': array of per-sample L2 norms
            - 'ptps': array of per-sample peak-to-peak
            - 'aggregate_mean': mean of all samples
            - 'aggregate_std': std of all samples
            - 'aggregate_min': min across all samples
            - 'aggregate_max': max across all samples
    """
    n_samples, n_timesteps = data.shape

    # Compute per-sample statistics (across time dimension, axis=1)
    per_sample_means = np.mean(data, axis=1)
    per_sample_stds = np.std(data, axis=1)
    per_sample_mins = np.min(data, axis=1)
    per_sample_maxs = np.max(data, axis=1)
    per_sample_energies = np.linalg.norm(data, axis=1)
    per_sample_ptps = np.ptp(data, axis=1)

    # Aggregate statistics
    aggregate_mean = np.mean(data)
    aggregate_std = np.std(data)
    aggregate_min = np.min(data)
    aggregate_max = np.max(data)

    return {
        'name': name,
        'n_samples': n_samples,
        'n_timesteps': n_timesteps,
        'means': per_sample_means,
        'stds': per_sample_stds,
        'mins': per_sample_mins,
        'maxs': per_sample_maxs,
        'energies': per_sample_energies,
        'ptps': per_sample_ptps,
        'aggregate_mean': aggregate_mean,
        'aggregate_std': aggregate_std,
        'aggregate_min': aggregate_min,
        'aggregate_max': aggregate_max
    }


def compute_global_statistics(loads: np.ndarray, responses: np.ndarray) -> Dict[str, float]:
    """
    Compute global normalization statistics for entire dataset.

    Args:
        loads: Training loads array [n_samples, n_timesteps]
        responses: Training responses array [n_samples, n_timesteps]

    Returns:
        Dictionary containing:
            - 'load_mean': float
            - 'load_std': float
            - 'load_min': float
            - 'load_max': float
            - 'response_mean': float
            - 'response_std': float
            - 'response_min': float
            - 'response_max': float
            - 'scale_ratio': response_std / load_std
    """
    # Flatten arrays for global statistics
    loads_flat = loads.flatten()
    responses_flat = responses.flatten()

    # Compute statistics
    load_mean = np.mean(loads_flat)
    load_std = np.std(loads_flat)
    load_min = np.min(loads_flat)
    load_max = np.max(loads_flat)

    response_mean = np.mean(responses_flat)
    response_std = np.std(responses_flat)
    response_min = np.min(responses_flat)
    response_max = np.max(responses_flat)

    # Compute scale ratio
    scale_ratio = response_std / load_std if load_std != 0 else 0.0

    return {
        'load_mean': float(load_mean),
        'load_std': float(load_std),
        'load_min': float(load_min),
        'load_max': float(load_max),
        'response_mean': float(response_mean),
        'response_std': float(response_std),
        'response_min': float(response_min),
        'response_max': float(response_max),
        'scale_ratio': float(scale_ratio)
    }


def analyze_scale_consistency(loads: np.ndarray, responses: np.ndarray) -> Dict[str, Any]:
    """
    Analyze scale consistency between input/output pairs.

    Args:
        loads: Loads array [n_samples, n_timesteps]
        responses: Responses array [n_samples, n_timesteps]

    Returns:
        Dictionary containing:
            - 'per_sample_ratios': array of std(response) / std(load) for each sample
            - 'mean_ratio': mean of ratios
            - 'std_ratio': std of ratios
            - 'min_ratio': minimum ratio
            - 'max_ratio': maximum ratio
            - 'is_consistent': bool (True if std_ratio < 0.5 * mean_ratio)
    """
    n_samples = loads.shape[0]
    per_sample_ratios = np.zeros(n_samples)

    for i in range(n_samples):
        load_std = np.std(loads[i])
        response_std = np.std(responses[i])

        # Compute ratio (handle division by zero)
        if load_std != 0:
            per_sample_ratios[i] = response_std / load_std
        else:
            per_sample_ratios[i] = 0.0

    # Aggregate statistics
    mean_ratio = np.mean(per_sample_ratios)
    std_ratio = np.std(per_sample_ratios)
    min_ratio = np.min(per_sample_ratios)
    max_ratio = np.max(per_sample_ratios)

    # Check consistency (low variance in ratios indicates consistent scaling)
    is_consistent = std_ratio < 0.5 * mean_ratio

    return {
        'per_sample_ratios': per_sample_ratios,
        'mean_ratio': float(mean_ratio),
        'std_ratio': float(std_ratio),
        'min_ratio': float(min_ratio),
        'max_ratio': float(max_ratio),
        'is_consistent': bool(is_consistent)
    }


def generate_normalization_report(stats: Dict[str, Any], output_path: str) -> None:
    """
    Generate markdown report with data analysis findings.

    Args:
        stats: Dictionary containing all computed statistics from analysis
        output_path: Path where markdown report should be saved

    Side Effects:
        Creates markdown file at output_path with formatted report
    """
    global_stats = stats['global']
    scale_stats = stats['scale']
    train_loads = stats['train_loads']
    train_responses = stats['train_responses']
    test_loads = stats['test_loads']
    test_responses = stats['test_responses']

    # Build report content
    lines = []
    lines.append("# CDON Dataset Analysis Report\n")
    lines.append("## Dataset Overview\n")
    lines.append(f"- Train samples: {train_loads['n_samples']}, shape: [{train_loads['n_samples']}, {train_loads['n_timesteps']}]\n")
    lines.append(f"- Test samples: {test_loads['n_samples']}, shape: [{test_loads['n_samples']}, {test_loads['n_timesteps']}]\n")
    lines.append(f"- Timesteps: {train_loads['n_timesteps']}\n")
    lines.append("\n")

    # Global statistics table
    lines.append("## Global Statistics\n\n")
    lines.append("| Field | Mean | Std | Min | Max |\n")
    lines.append("|-------|------|-----|-----|-----|\n")
    lines.append(f"| Loads (Acceleration) | {global_stats['load_mean']:.6e} | {global_stats['load_std']:.6f} | {global_stats['load_min']:.6f} | {global_stats['load_max']:.6f} |\n")
    lines.append(f"| Responses (Displacement) | {global_stats['response_mean']:.6e} | {global_stats['response_std']:.6f} | {global_stats['response_min']:.6f} | {global_stats['response_max']:.6f} |\n")
    lines.append("\n")
    lines.append(f"**Scale Ratio (response_std / load_std):** {global_stats['scale_ratio']:.4f}\n\n")

    # Scale consistency analysis
    lines.append("## Scale Consistency Analysis\n\n")
    lines.append("| Metric | Value |\n")
    lines.append("|--------|-------|\n")
    lines.append(f"| Mean per-sample scale ratio | {scale_stats['mean_ratio']:.4f} |\n")
    lines.append(f"| Std of scale ratios | {scale_stats['std_ratio']:.4f} |\n")
    lines.append(f"| Min scale ratio | {scale_stats['min_ratio']:.4f} |\n")
    lines.append(f"| Max scale ratio | {scale_stats['max_ratio']:.4f} |\n")
    consistency_str = "Consistent" if scale_stats['is_consistent'] else "Variable"
    lines.append(f"| Consistency | {consistency_str} |\n")
    lines.append("\n")

    # Per-sample statistics summary (statistics of statistics)
    lines.append("## Per-Sample Statistics Summary\n\n")
    lines.append("Summary of per-sample variability (not individual rows):\n\n")
    lines.append("| Statistic | Load Mean | Load Std | Load Min | Load Max | Response Mean | Response Std | Response Min | Response Max |\n")
    lines.append("|-----------|-----------|----------|----------|----------|---------------|--------------|--------------|--------------|\n")

    # Mean of means, stds, mins, maxs
    lines.append(f"| Mean | {np.mean(train_loads['means']):.6e} | {np.mean(train_loads['stds']):.6f} | {np.mean(train_loads['mins']):.6f} | {np.mean(train_loads['maxs']):.6f} | {np.mean(train_responses['means']):.6e} | {np.mean(train_responses['stds']):.6f} | {np.mean(train_responses['mins']):.6f} | {np.mean(train_responses['maxs']):.6f} |\n")
    lines.append(f"| Std | {np.std(train_loads['means']):.6e} | {np.std(train_loads['stds']):.6f} | {np.std(train_loads['mins']):.6f} | {np.std(train_loads['maxs']):.6f} | {np.std(train_responses['means']):.6e} | {np.std(train_responses['stds']):.6f} | {np.std(train_responses['mins']):.6f} | {np.std(train_responses['maxs']):.6f} |\n")
    lines.append(f"| Min | {np.min(train_loads['means']):.6e} | {np.min(train_loads['stds']):.6f} | {np.min(train_loads['mins']):.6f} | {np.min(train_loads['maxs']):.6f} | {np.min(train_responses['means']):.6e} | {np.min(train_responses['stds']):.6f} | {np.min(train_responses['mins']):.6f} | {np.min(train_responses['maxs']):.6f} |\n")
    lines.append(f"| Max | {np.max(train_loads['means']):.6e} | {np.max(train_loads['stds']):.6f} | {np.max(train_loads['maxs']):.6f} | {np.max(train_loads['maxs']):.6f} | {np.max(train_responses['means']):.6e} | {np.max(train_responses['stds']):.6f} | {np.max(train_responses['mins']):.6f} | {np.max(train_responses['maxs']):.6f} |\n")
    lines.append("\n")

    # Interpretation
    mean_std_load = np.mean(train_loads['stds'])
    std_std_load = np.std(train_loads['stds'])
    mean_std_response = np.mean(train_responses['stds'])
    std_std_response = np.std(train_responses['stds'])

    variability_load = (std_std_load / mean_std_load) if mean_std_load != 0 else 0
    variability_response = (std_std_response / mean_std_response) if mean_std_response != 0 else 0

    if variability_load < 0.2 and variability_response < 0.2:
        interp = "Samples have very similar statistics across the dataset (low variability)."
    elif variability_load < 0.5 and variability_response < 0.5:
        interp = "Samples have moderately similar statistics (moderate variability)."
    else:
        interp = "Samples show high variability in statistics across the dataset."

    lines.append(f"**Interpretation:** {interp}\n\n")

    # Normalization recommendation
    lines.append("## Normalization Recommendation\n\n")
    lines.append("**Strategy:** Global z-score normalization\n")
    lines.append("- Load normalization: (x - load_mean) / load_std\n")
    lines.append("- Response normalization: (y - response_mean) / response_std\n\n")
    lines.append("**Rationale:** Global normalization ensures consistent scaling across all samples for neural operator training. Min/Max values provided for reference only.\n")

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(lines)


def main():
    """
    Run complete CDON data analysis workflow.

    Steps:
        1. Load data from CDONData/
        2. Compute per-sample statistics for all 4 files
        3. Compute global statistics for normalization
        4. Analyze scale consistency
        5. Generate report
        6. Save normalization params to JSON
    """
    # Define paths
    data_dir = "CDONData"
    output_dir = "src/data_analysis"

    print("Loading CDON data...")
    data = load_cdon_data(data_dir)

    print("Computing per-sample statistics...")
    train_load_stats = compute_per_sample_statistics(data['train_loads'], 'train_loads')
    train_response_stats = compute_per_sample_statistics(data['train_responses'], 'train_responses')
    test_load_stats = compute_per_sample_statistics(data['test_loads'], 'test_loads')
    test_response_stats = compute_per_sample_statistics(data['test_responses'], 'test_responses')

    print("Computing global statistics...")
    global_stats = compute_global_statistics(data['train_loads'], data['train_responses'])

    print("Analyzing scale consistency...")
    scale_stats = analyze_scale_consistency(data['train_loads'], data['train_responses'])

    # Combine all statistics
    all_stats = {
        'global': global_stats,
        'scale': scale_stats,
        'train_loads': train_load_stats,
        'train_responses': train_response_stats,
        'test_loads': test_load_stats,
        'test_responses': test_response_stats
    }

    print("Generating report...")
    report_path = os.path.join(output_dir, 'scale_analysis_report.md')
    generate_normalization_report(all_stats, report_path)

    print("Saving normalization parameters...")
    norm_params = {
        'load_mean': float(global_stats['load_mean']),
        'load_std': float(global_stats['load_std']),
        'load_min': float(global_stats['load_min']),
        'load_max': float(global_stats['load_max']),
        'response_mean': float(global_stats['response_mean']),
        'response_std': float(global_stats['response_std']),
        'response_min': float(global_stats['response_min']),
        'response_max': float(global_stats['response_max']),
        'scale_ratio': float(global_stats['scale_ratio'])
    }

    os.makedirs('configs', exist_ok=True)
    with open('configs/cdon_stats.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    print(f"\nAnalysis complete!")
    print(f"Report saved to {report_path}")
    print(f"Normalization parameters saved to configs/cdon_stats.json")
    print(f"\nKey findings:")
    print(f"  Load std: {global_stats['load_std']:.6f}")
    print(f"  Response std: {global_stats['response_std']:.6f}")
    print(f"  Scale ratio: {global_stats['scale_ratio']:.4f}")
    print(f"  Scale consistency: {'Consistent' if scale_stats['is_consistent'] else 'Variable'}")


if __name__ == "__main__":
    main()
