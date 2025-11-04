"""
Unit tests for CDON data analysis module.

Tests all functions in analyze_cdon_data.py with both synthetic and real data.
"""

import numpy as np
import os
import json
import tempfile
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_analysis.analyze_cdon_data import (
    load_cdon_data,
    compute_per_sample_statistics,
    compute_global_statistics,
    analyze_scale_consistency,
    generate_normalization_report
)


class TestDataLoading:
    """Tests for load_cdon_data function"""

    def test_data_loading_all_files(self):
        """Verify all 4 files load correctly with expected shapes"""
        # Use real CDON data
        data_dir = "CDONData"

        if not os.path.exists(data_dir):
            pytest.skip("CDON data not available")

        data = load_cdon_data(data_dir)

        # Assert returned dict has 4 keys
        assert len(data) == 4
        assert 'train_loads' in data
        assert 'train_responses' in data
        assert 'test_loads' in data
        'test_responses' in data

        # Assert shapes are correct
        assert data['train_loads'].shape == (100, 4000)
        assert data['train_responses'].shape == (100, 4000)
        assert data['test_loads'].shape == (44, 4000)
        assert data['test_responses'].shape == (44, 4000)

    def test_data_loading_missing_file(self):
        """Verify error handling when file is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only 3 files
            np.save(os.path.join(tmpdir, 'train_Loads.npy'), np.zeros((100, 4000)))
            np.save(os.path.join(tmpdir, 'train_Responses.npy'), np.zeros((100, 4000)))
            np.save(os.path.join(tmpdir, 'test_Loads.npy'), np.zeros((44, 4000)))
            # Missing test_Responses.npy

            with pytest.raises(FileNotFoundError) as excinfo:
                load_cdon_data(tmpdir)

            assert 'test_Responses.npy' in str(excinfo.value)

    def test_shape_consistency_train_test(self):
        """Ensure shape validation works correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with inconsistent time dimensions
            np.save(os.path.join(tmpdir, 'train_Loads.npy'), np.zeros((100, 4000)))
            np.save(os.path.join(tmpdir, 'train_Responses.npy'), np.zeros((100, 4000)))
            np.save(os.path.join(tmpdir, 'test_Loads.npy'), np.zeros((44, 3000)))  # Wrong size
            np.save(os.path.join(tmpdir, 'test_Responses.npy'), np.zeros((44, 3000)))

            with pytest.raises(ValueError) as excinfo:
                load_cdon_data(tmpdir)

            assert 'Time dimension mismatch' in str(excinfo.value)


class TestPerSampleStatistics:
    """Tests for compute_per_sample_statistics function"""

    def test_per_sample_statistics_computation(self):
        """Validate per-sample statistics are computed correctly"""
        # Create simple synthetic array
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        result = compute_per_sample_statistics(data, 'test_data')

        # Check basic metadata
        assert result['name'] == 'test_data'
        assert result['n_samples'] == 2
        assert result['n_timesteps'] == 3

        # Check per-sample means
        np.testing.assert_array_almost_equal(result['means'], [2.0, 5.0])

        # Check per-sample stds
        expected_stds = [np.std([1, 2, 3]), np.std([4, 5, 6])]
        np.testing.assert_array_almost_equal(result['stds'], expected_stds)

        # Check per-sample mins and maxs
        np.testing.assert_array_equal(result['mins'], [1, 4])
        np.testing.assert_array_equal(result['maxs'], [3, 6])

        # Check per-sample energies
        expected_energies = [np.linalg.norm([1, 2, 3]), np.linalg.norm([4, 5, 6])]
        np.testing.assert_array_almost_equal(result['energies'], expected_energies)

        # Check aggregate statistics
        assert result['aggregate_mean'] == np.mean(data)
        assert result['aggregate_std'] == np.std(data)
        assert result['aggregate_min'] == 1
        assert result['aggregate_max'] == 6


class TestGlobalStatistics:
    """Tests for compute_global_statistics function"""

    def test_global_statistics_correctness(self):
        """Verify global statistics match numpy functions"""
        # Create synthetic data
        loads = np.random.randn(10, 100)
        responses = np.random.randn(10, 100) * 2.5  # Different scale

        result = compute_global_statistics(loads, responses)

        # Check loads statistics
        assert result['load_mean'] == pytest.approx(np.mean(loads))
        assert result['load_std'] == pytest.approx(np.std(loads))
        assert result['load_min'] == pytest.approx(np.min(loads))
        assert result['load_max'] == pytest.approx(np.max(loads))

        # Check responses statistics
        assert result['response_mean'] == pytest.approx(np.mean(responses))
        assert result['response_std'] == pytest.approx(np.std(responses))
        assert result['response_min'] == pytest.approx(np.min(responses))
        assert result['response_max'] == pytest.approx(np.max(responses))

        # Check scale ratio
        expected_ratio = np.std(responses) / np.std(loads)
        assert result['scale_ratio'] == pytest.approx(expected_ratio)


class TestScaleConsistency:
    """Tests for analyze_scale_consistency function"""

    def test_scale_ratio_computation(self):
        """Verify scale ratio analysis is correct"""
        # Create loads with known std (all samples std=1.0)
        np.random.seed(42)
        loads = np.random.randn(10, 100)
        # Normalize each sample to have std=1.0
        for i in range(loads.shape[0]):
            loads[i] = (loads[i] - np.mean(loads[i])) / np.std(loads[i])

        # Create responses with known std (all samples std=2.5)
        responses = np.random.randn(10, 100)
        for i in range(responses.shape[0]):
            responses[i] = (responses[i] - np.mean(responses[i])) / np.std(responses[i]) * 2.5

        result = analyze_scale_consistency(loads, responses)

        # All ratios should be approximately 2.5
        assert result['mean_ratio'] == pytest.approx(2.5, abs=0.1)

        # Check consistency flag (should be True since all samples have same ratio)
        assert result['is_consistent'] == True

        # Check that we have the right number of ratios
        assert len(result['per_sample_ratios']) == 10

    def test_scale_ratio_in_reasonable_range(self):
        """Check scale ratios are within expected bounds using real data"""
        data_dir = "CDONData"

        if not os.path.exists(data_dir):
            pytest.skip("CDON data not available")

        data = load_cdon_data(data_dir)
        result = analyze_scale_consistency(data['train_loads'], data['train_responses'])

        # All ratios should be between 0.1 and 10.0 (sanity check)
        assert np.all(result['per_sample_ratios'] > 0.1)
        assert np.all(result['per_sample_ratios'] < 10.0)

        # Mean ratio should be approximately 2.7 (from preliminary analysis)
        assert result['mean_ratio'] == pytest.approx(2.7, abs=0.5)


class TestReportGeneration:
    """Tests for generate_normalization_report function"""

    def test_report_generation_creates_file(self):
        """Ensure report file is created with expected content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock stats dictionary
            stats = {
                'global': {
                    'load_mean': -1e-6,
                    'load_std': 0.014,
                    'load_min': -0.31,
                    'load_max': 0.30,
                    'response_mean': -2e-5,
                    'response_std': 0.038,
                    'response_min': -0.51,
                    'response_max': 0.59,
                    'scale_ratio': 2.71
                },
                'scale': {
                    'mean_ratio': 2.71,
                    'std_ratio': 0.15,
                    'min_ratio': 2.3,
                    'max_ratio': 3.1,
                    'is_consistent': True,
                    'per_sample_ratios': np.ones(100) * 2.71
                },
                'train_loads': {
                    'n_samples': 100,
                    'n_timesteps': 4000,
                    'means': np.zeros(100),
                    'stds': np.ones(100) * 0.014,
                    'mins': np.ones(100) * -0.3,
                    'maxs': np.ones(100) * 0.3,
                },
                'train_responses': {
                    'n_samples': 100,
                    'n_timesteps': 4000,
                    'means': np.zeros(100),
                    'stds': np.ones(100) * 0.038,
                    'mins': np.ones(100) * -0.5,
                    'maxs': np.ones(100) * 0.6,
                },
                'test_loads': {
                    'n_samples': 44,
                    'n_timesteps': 4000,
                    'means': np.zeros(44),
                    'stds': np.ones(44) * 0.014,
                    'mins': np.ones(44) * -0.3,
                    'maxs': np.ones(44) * 0.3,
                },
                'test_responses': {
                    'n_samples': 44,
                    'n_timesteps': 4000,
                    'means': np.zeros(44),
                    'stds': np.ones(44) * 0.038,
                    'mins': np.ones(44) * -0.5,
                    'maxs': np.ones(44) * 0.6,
                }
            }

            output_path = os.path.join(tmpdir, 'test_report.md')
            generate_normalization_report(stats, output_path)

            # Assert file exists
            assert os.path.exists(output_path)

            # Read file and check contents
            with open(output_path, 'r') as f:
                content = f.read()

            # Check for expected sections
            assert '## Dataset Overview' in content
            assert '## Global Statistics' in content
            assert '## Scale Consistency Analysis' in content
            assert '## Per-Sample Statistics Summary' in content
            assert '## Normalization Recommendation' in content

            # Verify summary statistics table is present (not full 144-row table)
            assert 'Summary of per-sample variability' in content
            assert '| Mean |' in content
            assert '| Std |' in content

    def test_report_contains_all_statistics(self):
        """Verify report includes all computed statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create stats dict with known values
            stats = {
                'global': {
                    'load_mean': 1.23e-5,
                    'load_std': 0.0142,
                    'load_min': -0.31,
                    'load_max': 0.30,
                    'response_mean': 4.56e-5,
                    'response_std': 0.0379,
                    'response_min': -0.51,
                    'response_max': 0.59,
                    'scale_ratio': 2.6690
                },
                'scale': {
                    'mean_ratio': 2.67,
                    'std_ratio': 0.12,
                    'min_ratio': 2.4,
                    'max_ratio': 3.0,
                    'is_consistent': True,
                    'per_sample_ratios': np.ones(100) * 2.67
                },
                'train_loads': {
                    'n_samples': 100,
                    'n_timesteps': 4000,
                    'means': np.zeros(100),
                    'stds': np.ones(100) * 0.014,
                    'mins': np.ones(100) * -0.3,
                    'maxs': np.ones(100) * 0.3,
                },
                'train_responses': {
                    'n_samples': 100,
                    'n_timesteps': 4000,
                    'means': np.zeros(100),
                    'stds': np.ones(100) * 0.038,
                    'mins': np.ones(100) * -0.5,
                    'maxs': np.ones(100) * 0.6,
                },
                'test_loads': {
                    'n_samples': 44,
                    'n_timesteps': 4000,
                    'means': np.zeros(44),
                    'stds': np.ones(44) * 0.014,
                    'mins': np.ones(44) * -0.3,
                    'maxs': np.ones(44) * 0.3,
                },
                'test_responses': {
                    'n_samples': 44,
                    'n_timesteps': 4000,
                    'means': np.zeros(44),
                    'stds': np.ones(44) * 0.038,
                    'mins': np.ones(44) * -0.5,
                    'maxs': np.ones(44) * 0.6,
                }
            }

            output_path = os.path.join(tmpdir, 'test_report.md')
            generate_normalization_report(stats, output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            # Check that key values appear in report
            assert '0.0142' in content  # load_std
            assert '0.0379' in content  # response_std
            assert '2.6690' in content or '2.669' in content  # scale_ratio
            assert 'z-score normalization' in content.lower()


class TestIntegration:
    """Integration tests for full workflow"""

    def test_main_workflow_completes(self):
        """Integration test of full workflow on real data"""
        data_dir = "CDONData"

        if not os.path.exists(data_dir):
            pytest.skip("CDON data not available")

        # Run full workflow
        data = load_cdon_data(data_dir)
        train_load_stats = compute_per_sample_statistics(data['train_loads'], 'train_loads')
        train_response_stats = compute_per_sample_statistics(data['train_responses'], 'train_responses')
        test_load_stats = compute_per_sample_statistics(data['test_loads'], 'test_loads')
        test_response_stats = compute_per_sample_statistics(data['test_responses'], 'test_responses')

        global_stats = compute_global_statistics(data['train_loads'], data['train_responses'])
        scale_stats = analyze_scale_consistency(data['train_loads'], data['train_responses'])

        all_stats = {
            'global': global_stats,
            'scale': scale_stats,
            'train_loads': train_load_stats,
            'train_responses': train_response_stats,
            'test_loads': test_load_stats,
            'test_responses': test_response_stats
        }

        # Generate report
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'scale_analysis_report.md')
            generate_normalization_report(all_stats, report_path)

            # Assert report created
            assert os.path.exists(report_path)

            # Save JSON
            json_path = os.path.join(tmpdir, 'cdon_stats.json')
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

            with open(json_path, 'w') as f:
                json.dump(norm_params, f, indent=2)

            # Assert JSON created and contains required keys
            assert os.path.exists(json_path)

            with open(json_path, 'r') as f:
                loaded = json.load(f)

            required_keys = ['load_mean', 'load_std', 'load_min', 'load_max',
                           'response_mean', 'response_std', 'response_min', 'response_max', 'scale_ratio']
            for key in required_keys:
                assert key in loaded

    def test_analysis_on_dummy_data(self):
        """Verify analysis pipeline works with dummy data directory"""
        dummy_dir = "data/dummy_cdon"

        if not os.path.exists(dummy_dir):
            pytest.skip("Dummy data not available yet (Phase 2)")

        data = load_cdon_data(dummy_dir)

        # Verify shapes match real data
        assert data['train_loads'].shape == (100, 4000)
        assert data['test_loads'].shape == (44, 4000)

        # Run full analysis pipeline
        global_stats = compute_global_statistics(data['train_loads'], data['train_responses'])

        # Assert statistics are reasonable
        assert 0.001 < global_stats['load_std'] < 0.1
        assert 0.001 < global_stats['response_std'] < 0.2

    def test_dummy_vs_real_data_pipeline_compatibility(self):
        """Ensure same code works for both dummy and real data"""
        real_dir = "CDONData"
        dummy_dir = "data/dummy_cdon"

        if not os.path.exists(real_dir) or not os.path.exists(dummy_dir):
            pytest.skip("Both real and dummy data required")

        # Run on real data
        real_data = load_cdon_data(real_dir)
        real_stats = compute_global_statistics(real_data['train_loads'], real_data['train_responses'])

        # Run on dummy data
        dummy_data = load_cdon_data(dummy_dir)
        dummy_stats = compute_global_statistics(dummy_data['train_loads'], dummy_data['train_responses'])

        # Verify both produce same structure
        assert set(real_stats.keys()) == set(dummy_stats.keys())

        # Verify scale ratios are in similar ranges (within factor of 2)
        ratio_real = real_stats['scale_ratio']
        ratio_dummy = dummy_stats['scale_ratio']
        assert 0.5 < ratio_dummy / ratio_real < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
