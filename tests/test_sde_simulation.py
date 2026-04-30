"""
Unit tests for utils/sde_simulation.py

Tests the jump process discretization functions used for approximating
diffusion processes.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.sde_simulation import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion


class TestSDETransitionRates:
    """Tests for sde_transition_rates function."""

    def test_basic_1d_rates(self):
        """Test transition rates for simple 1D case."""
        x = np.array([0.0])
        b = 1.0  # positive drift
        a = 1.0  # unit diffusion
        h = 0.1  # step size

        rates = sde_transition_rates(x, b, a, h)

        # Should return rates for forward and backward jumps
        assert len(rates) == 2, "Should have 2 rates for 1D (forward and backward)"
        assert all(r >= 0 for r in rates), "All rates should be non-negative"

    def test_zero_drift(self):
        """Test that zero drift gives symmetric rates."""
        x = np.array([0.5])
        b = 0.0  # zero drift
        a = 1.0
        h = 0.1

        rates = sde_transition_rates(x, b, a, h)

        # With zero drift, forward and backward rates should be equal
        assert np.isclose(rates[0], rates[1], rtol=1e-10), \
            "Zero drift should give symmetric rates"

    def test_positive_drift_bias(self):
        """Test that positive drift biases rates forward."""
        x = np.array([0.0])
        b = 2.0  # positive drift
        a = 1.0
        h = 0.1

        rates = sde_transition_rates(x, b, a, h)

        # Positive drift should make forward rate higher
        assert rates[0] > rates[1], "Positive drift should bias forward"

    def test_rates_scale_with_diffusion(self):
        """Test that rates scale appropriately with diffusion coefficient."""
        x = np.array([0.0])
        b = 0.0
        h = 0.1

        rates_small = sde_transition_rates(x, b, 1.0, h)
        rates_large = sde_transition_rates(x, b, 4.0, h)

        # Larger diffusion should give larger total rate
        assert np.sum(rates_large) > np.sum(rates_small), \
            "Larger diffusion should increase total rate"


class TestOneStepSDE:
    """Tests for one_step_sde function."""

    def test_moves_particle(self):
        """Test that one_step_sde moves the particle."""
        x = np.array([0.0])
        rates = np.array([1.0, 1.0])  # equal forward/backward
        h = 0.1

        # Run multiple times to check particle moves
        positions = [one_step_sde(x.copy(), rates, h)[0] for _ in range(100)]

        # Should have moved to different positions
        unique_positions = len(set(positions))
        assert unique_positions > 1, "Particle should move to different positions"

    def test_step_size(self):
        """Test that steps are approximately h in size."""
        x = np.array([0.0])
        rates = np.array([1.0, 1.0])
        h = 0.1

        steps = []
        for _ in range(100):
            new_x = one_step_sde(x.copy(), rates, h)
            steps.append(abs(new_x[0] - x[0]))

        # All steps should be approximately h
        assert all(np.isclose(s, h, rtol=0.1) for s in steps), \
            "Steps should be approximately h"

    def test_biased_rates(self):
        """Test that biased rates lead to biased movement."""
        x = np.array([0.0])
        rates_forward = np.array([10.0, 0.1])  # heavily forward biased
        h = 0.1

        final_positions = [one_step_sde(x.copy(), rates_forward, h)[0] for _ in range(100)]

        # Most should be positive (forward)
        forward_count = sum(1 for p in final_positions if p > 0)
        assert forward_count > 80, "Heavily forward biased rates should mostly move forward"


class TestPureJumpApproxDiffusion:
    """Tests for pure_jump_approx_diffusion function."""

    def test_returns_trajectory(self):
        """Test that function returns trajectory and time arrays."""
        T = 1.0
        b = lambda x: -x  # OU drift
        a = lambda x: np.ones_like(x)
        h0 = 0.1
        x0 = np.array([0.5])

        X, t = pure_jump_approx_diffusion(T, b, a, h0, x0)

        assert len(X) == len(t), "X and t should have same length"
        assert len(t) > 1, "Should have multiple time points"
        assert t[0] == 0, "Should start at time 0"
        # Stochastic timing may not reach exactly T, allow small tolerance
        assert t[-1] >= T - 0.1, "Should approximately reach terminal time"

    def test_starts_at_initial_position(self):
        """Test trajectory starts at initial position."""
        T = 1.0
        b = lambda x: np.zeros_like(x)
        a = lambda x: np.ones_like(x)
        h0 = 0.1
        x0 = np.array([0.75])

        X, t = pure_jump_approx_diffusion(T, b, a, h0, x0)

        assert np.allclose(X[0], x0), "Should start at initial position"

    def test_ou_process_mean_reversion(self):
        """Test that OU process shows mean reversion behavior."""
        T = 5.0
        theta = 1.0  # mean reversion rate
        mu = 0.0     # mean
        b = lambda x: -theta * (x - mu)  # OU drift
        a = lambda x: 0.5 * np.ones_like(x)
        h0 = 0.05
        x0 = np.array([2.0])  # start far from mean

        X, t = pure_jump_approx_diffusion(T, b, a, h0, x0)

        # Final position should be closer to mean than initial
        assert abs(X[-1][0] - mu) < abs(x0[0] - mu), \
            "OU process should revert towards mean"

    def test_time_monotonic(self):
        """Test that time array is strictly increasing."""
        T = 1.0
        b = lambda x: np.zeros_like(x)
        a = lambda x: np.ones_like(x)
        h0 = 0.1
        x0 = np.array([0.0])

        X, t = pure_jump_approx_diffusion(T, b, a, h0, x0)

        assert all(t[i] < t[i+1] for i in range(len(t)-1)), \
            "Time should be strictly increasing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
