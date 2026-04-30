"""
Unit tests for utils/postprocessing.py

Tests the empirical measure functionals and resampling procedures.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.postprocessing import (
    empirical_measure_functional,
    cumulative_empirical_measure_functional,
    weighted_empirical_measure_functional,
    cumulative_mean_weighted_empirical_measure,
    resample_simple_weighted_empirical_measure,
    resample_weighted_empirical_measure
)


class TestEmpiricalMeasureFunctional:
    """Tests for empirical_measure_functional function."""

    def test_constant_function(self):
        """Test that constant function gives that constant."""
        path = np.array([[0], [1], [2], [3], [4]])
        times = np.array([0, 1, 2, 3, 4])
        f = lambda x: np.ones_like(x[:, 0]) * 5.0

        result = empirical_measure_functional(path, f, times)

        assert np.isclose(result, 5.0), \
            "Constant function should give that constant"

    def test_identity_function_uniform_time(self):
        """Test identity function with uniform time spacing."""
        path = np.array([[0], [1], [2], [3]])
        times = np.array([0, 1, 2, 3])
        f = lambda x: x[:, 0]

        result = empirical_measure_functional(path, f, times)

        # dt = [1, 1, 1, 0] (last dt is 0)
        # weighted mean = (0*1 + 1*1 + 2*1 + 3*0) / 3 = 1.0
        expected = (0*1 + 1*1 + 2*1) / 3
        assert np.isclose(result, expected), \
            f"Expected {expected}, got {result}"

    def test_variable_time_spacing(self):
        """Test with variable time spacing."""
        path = np.array([[0], [1], [0]])
        times = np.array([0, 1, 10])  # much longer time at position 1
        f = lambda x: x[:, 0]

        result = empirical_measure_functional(path, f, times)

        # dt = [1, 9, 0]
        # weighted mean = (0*1 + 1*9 + 0*0) / 10 = 0.9
        expected = (0*1 + 1*9) / 10
        assert np.isclose(result, expected), \
            f"Expected {expected}, got {result}"

    def test_quadratic_function(self):
        """Test with quadratic function."""
        path = np.array([[0], [1], [2]])
        times = np.array([0, 1, 2])
        f = lambda x: x[:, 0]**2

        result = empirical_measure_functional(path, f, times)

        # dt = [1, 1, 0]
        # f(path) = [0, 1, 4]
        # weighted mean = (0*1 + 1*1) / 2 = 0.5
        expected = (0*1 + 1*1) / 2
        assert np.isclose(result, expected), \
            f"Expected {expected}, got {result}"


class TestCumulativeEmpiricalMeasureFunctional:
    """Tests for cumulative_empirical_measure_functional function."""

    def test_cumulative_structure(self):
        """Test that cumulative returns array of correct length."""
        path = np.array([[0], [1], [2], [3], [4]])
        times = np.array([0, 1, 2, 3, 4])
        f = lambda x: x[:, 0]

        result = cumulative_empirical_measure_functional(path, f, times)

        assert len(result) == len(path), \
            "Should return array of same length as input"

    def test_cumulative_converges(self):
        """Test that cumulative converges to final value."""
        path = np.array([[1], [1], [1], [1], [1]])
        times = np.array([0, 1, 2, 3, 4])
        f = lambda x: x[:, 0]

        cumulative = cumulative_empirical_measure_functional(path, f, times)
        final = empirical_measure_functional(path, f, times)

        # Last non-inf cumulative value should match final
        # Note: last element has dt=0, so cumulative[-2] is more meaningful
        assert np.isclose(cumulative[-2], final, rtol=0.1), \
            "Cumulative should converge to final value"


class TestWeightedEmpiricalMeasureFunctional:
    """Tests for weighted_empirical_measure_functional function."""

    def test_rho_half_symmetric(self):
        """Test that rho=0.5 gives average of forward and backward."""
        x_path = np.array([[0], [1], [2]])
        y_path = np.array([[2], [1], [0]])
        rho = np.array([0.5, 0.5, 0.5])
        times = np.array([0, 1, 2])

        # f(x,y) = x, should give same result as vanilla
        f = lambda x, y: x[:, 0]

        result = weighted_empirical_measure_functional(
            x_path, y_path, rho, f, times
        )

        # At each point: 0.5*x + 0.5*y = 0.5*(x+y) = 1.0 for all points
        # So weighted average should be 1.0
        expected = 1.0
        assert np.isclose(result, expected), \
            f"Expected {expected}, got {result}"

    def test_rho_one_uses_only_forward(self):
        """Test that rho=1 uses only forward trajectory."""
        x_path = np.array([[1], [1], [1]])
        y_path = np.array([[0], [0], [0]])
        rho = np.array([1.0, 1.0, 1.0])
        times = np.array([0, 1, 2])

        f = lambda x, y: x[:, 0]

        result = weighted_empirical_measure_functional(
            x_path, y_path, rho, f, times
        )

        assert np.isclose(result, 1.0), \
            "rho=1 should give result from forward trajectory only"

    def test_rho_zero_uses_only_backward(self):
        """Test that rho=0 uses only backward trajectory."""
        x_path = np.array([[1], [1], [1]])
        y_path = np.array([[2], [2], [2]])
        rho = np.array([0.0, 0.0, 0.0])
        times = np.array([0, 1, 2])

        f = lambda x, y: x[:, 0]  # Note: f(x,y) evaluated at x

        result = weighted_empirical_measure_functional(
            x_path, y_path, rho, f, times
        )

        # When rho=0: we get feval_fwd*0 + feval_bwd*1 = f(y,x)
        # f(y,x) = y = 2
        assert np.isclose(result, 2.0), \
            "rho=0 should give result from backward trajectory"


class TestResampleSimpleWeightedEmpiricalMeasure:
    """Tests for resample_simple_weighted_empirical_measure function."""

    def test_returns_correct_length(self):
        """Test that resampling returns array of correct length."""
        path = np.array([[0], [1], [2], [3], [4]])
        times = np.array([0, 1, 2, 3, 4])

        samples = resample_simple_weighted_empirical_measure(path, times)

        assert len(samples) == len(path), \
            "Should return same number of samples"

    def test_samples_from_path(self):
        """Test that all samples come from the original path."""
        path = np.array([[0], [1], [2], [3], [4]])
        times = np.array([0, 1, 2, 3, 4])

        samples = resample_simple_weighted_empirical_measure(path, times)

        # All samples should be values from path
        for sample in samples:
            assert sample[0] in path[:, 0], \
                f"Sample {sample} not in original path"

    def test_longer_time_more_samples(self):
        """Test that positions with longer residence time get more samples."""
        # Position 1 has 90% of the total time
        path = np.array([[0], [1], [2]])
        times = np.array([0, 1, 10])  # pos 1 has dt=9, others have dt=1 and 0

        # Run multiple times to get statistics
        sample_counts = {0: 0, 1: 0, 2: 0}
        n_trials = 1000

        for _ in range(n_trials):
            samples = resample_simple_weighted_empirical_measure(path, times)
            for s in samples:
                sample_counts[int(s[0])] += 1

        # Position 1 should have most samples (about 90%)
        total = sum(sample_counts.values())
        pos1_fraction = sample_counts[1] / total

        assert pos1_fraction > 0.8, \
            f"Position 1 should dominate samples, got {pos1_fraction:.2%}"


class TestResampleWeightedEmpiricalMeasure:
    """Tests for resample_weighted_empirical_measure function."""

    def test_returns_correct_length(self):
        """Test that resampling returns array of correct length."""
        path = np.array([[0], [1], [2], [3], [4]])
        rho = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        times = np.array([0, 1, 2, 3, 4])

        samples = resample_weighted_empirical_measure(path, rho, times)

        assert len(samples) == len(path), \
            "Should return same number of samples"

    def test_high_rho_more_samples(self):
        """Test that positions with high rho get more samples."""
        # All equal time, but position 1 has rho=0.9, others rho=0.05
        path = np.array([[0], [1], [2]])
        rho = np.array([0.05, 0.9, 0.05])
        times = np.array([0, 1, 2])  # dt = [1, 1, 0]

        # Run multiple times to get statistics
        sample_counts = {0: 0, 1: 0, 2: 0}
        n_trials = 1000

        for _ in range(n_trials):
            samples = resample_weighted_empirical_measure(path, rho, times)
            for s in samples:
                sample_counts[int(s[0])] += 1

        # Position 1 should have most samples
        total = sum(sample_counts.values())
        pos1_fraction = sample_counts[1] / total

        # With rho = [0.05, 0.9, 0.05] and dt = [1, 1, 0]
        # weights = [0.05, 0.9, 0] -> normalized [0.053, 0.947, 0]
        assert pos1_fraction > 0.85, \
            f"High-rho position should dominate, got {pos1_fraction:.2%}"


class TestEmpiricalMeasureEdgeCases:
    """Edge case tests for empirical measure functions."""

    def test_single_point(self):
        """Test with single point trajectory."""
        path = np.array([[1.0]])
        times = np.array([0.0])
        f = lambda x: x[:, 0]

        # This might produce nan or inf due to dt=0
        # The function should handle this gracefully
        result = empirical_measure_functional(path, f, times)

        # With only one point, dt=0, so sum(dt)=0 -> division by zero
        # This is an edge case - result might be nan
        assert not np.isinf(result) or np.isnan(result), \
            "Single point should not produce inf (nan is acceptable)"

    def test_two_points(self):
        """Test with two point trajectory."""
        path = np.array([[0], [1]])
        times = np.array([0, 1])
        f = lambda x: x[:, 0]

        result = empirical_measure_functional(path, f, times)

        # dt = [1, 0], weighted mean = 0*1 / 1 = 0
        assert np.isclose(result, 0.0), \
            f"Expected 0.0, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
