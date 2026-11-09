"""
Unit tests for utils/fleming_viot.py

Tests the Fleming-Viot particle system functions including infinite swapping,
symmetrized dynamics, and the main simulation functions.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.fleming_viot import (
    inf_swap_rate, symmetrized_dynamics, symmetrized_kill_clone_rate,
    event_rates, fleming_viot, fleming_viot_vanilla
)


class TestInfSwapRate:
    """Tests for inf_swap_rate function."""

    def test_symmetric_positions(self):
        """Test that equal positions give rho = 0.5."""
        x = np.array([0.5])
        y = np.array([0.5])
        V = lambda z: z[0]**2  # quadratic potential
        epsilon = 1.0

        rho = inf_swap_rate(x, y, V, epsilon)

        assert np.isclose(rho, 0.5, rtol=1e-10), \
            "Equal positions should give rho = 0.5"

    def test_rho_bounds(self):
        """Test that rho is always in [0, 1]."""
        V = lambda z: np.cos(2 * np.pi * z[0])
        epsilon = 0.1

        for _ in range(100):
            x = np.random.uniform(-1, 1, size=(1,))
            y = np.random.uniform(-1, 1, size=(1,))
            rho = inf_swap_rate(x, y, V, epsilon)

            assert 0 <= rho <= 1, f"rho={rho} should be in [0, 1]"

    def test_antisymmetry(self):
        """Test that rho(x,y) + rho(y,x) = 1."""
        x = np.array([0.3])
        y = np.array([0.7])
        V = lambda z: z[0]**2
        epsilon = 0.5

        rho_xy = inf_swap_rate(x, y, V, epsilon)
        rho_yx = inf_swap_rate(y, x, V, epsilon)

        assert np.isclose(rho_xy + rho_yx, 1.0, rtol=1e-10), \
            "rho(x,y) + rho(y,x) should equal 1"

    def test_lower_potential_higher_rho(self):
        """Test that particle at lower potential gets higher rho."""
        V = lambda z: z[0]**2  # minimum at 0
        epsilon = 0.5

        x_low = np.array([0.1])   # low potential
        x_high = np.array([0.9])  # high potential

        rho = inf_swap_rate(x_low, x_high, V, epsilon)

        assert rho > 0.5, "Particle at lower potential should have rho > 0.5"

    def test_temperature_effect(self):
        """Test that higher temperature makes rho closer to 0.5."""
        x = np.array([0.2])
        y = np.array([0.8])
        V = lambda z: z[0]**2

        rho_cold = inf_swap_rate(x, y, V, epsilon=0.1)
        rho_hot = inf_swap_rate(x, y, V, epsilon=10.0)

        # Higher temperature should bring rho closer to 0.5
        assert abs(rho_hot - 0.5) < abs(rho_cold - 0.5), \
            "Higher temperature should make rho closer to 0.5"


class TestSymmetrizedDynamics:
    """Tests for symmetrized_dynamics function."""

    def test_rho_half_gives_zero(self):
        """Test that rho=0.5 gives zero symmetrized drift."""
        x = np.array([0.5])
        rho_eps = 0.5
        DV = lambda z: 2 * z  # gradient of quadratic

        bsymm = symmetrized_dynamics(x, rho_eps, DV)

        assert np.allclose(bsymm, 0), \
            "rho=0.5 should give zero symmetrized drift"

    def test_rho_zero_gives_positive(self):
        """Test that rho=0 gives drift in direction of -DV."""
        x = np.array([1.0])
        rho_eps = 0.0
        DV = lambda z: 2 * z

        bsymm = symmetrized_dynamics(x, rho_eps, DV)

        # (1 - 2*0) * DV = DV
        expected = DV(x)
        assert np.allclose(bsymm, expected), \
            "rho=0 should give bsymm = DV"

    def test_rho_one_gives_negative(self):
        """Test that rho=1 gives drift in opposite direction."""
        x = np.array([1.0])
        rho_eps = 1.0
        DV = lambda z: 2 * z

        bsymm = symmetrized_dynamics(x, rho_eps, DV)

        # (1 - 2*1) * DV = -DV
        expected = -DV(x)
        assert np.allclose(bsymm, expected), \
            "rho=1 should give bsymm = -DV"


class TestSymmetrizedKillCloneRate:
    """Tests for symmetrized_kill_clone_rate function."""

    def test_basic_computation(self):
        """Test basic kill/clone rate computation."""
        x = np.array([0.5])
        rhoeps = 0.3
        D2V = lambda z: 2 * np.ones_like(z[0])  # constant Laplacian
        c = lambda z: 1.0  # constant kill rate

        r_kf, r_kb = symmetrized_kill_clone_rate(x, rhoeps, D2V, c)

        # r_kf = rhoeps * c(x); r_kb = (1-rhoeps)*(c(x) - D2V(x))
        expected_kf = 0.3 * 1.0
        expected_kb = (1 - 0.3) * (1.0 - 2.0)
        assert np.isclose(r_kf, expected_kf), f"Expected r_kf {expected_kf}, got {r_kf}"
        assert np.isclose(r_kb, expected_kb), f"Expected r_kb {expected_kb}, got {r_kb}"
        # their sum is the symmetrized kill/clone rate c(x) - (1-rhoeps)*D2V(x)
        assert np.isclose(r_kf + r_kb, 1.0 - (1 - 0.3) * 2.0)


class TestEventRates:
    """Tests for event_rates function."""

    def test_returns_correct_structure(self):
        """Test that event_rates returns the correct structure."""
        x = np.array([0.5])
        rhoeps = 0.5
        DV = lambda z: np.zeros_like(z)
        D2V = lambda z: np.zeros_like(z[0])
        a = lambda z: np.ones_like(z)
        c = lambda z: 0.0
        h = 0.1

        result = event_rates(x, rhoeps, DV, D2V, a, c, h)

        assert len(result) == 5, "Should return 5 values"
        net_rate, dynamics_rates, r_kf, r_kb, h_step = result

        assert isinstance(net_rate, (int, float, np.floating)), \
            "net_rate should be scalar"
        assert len(dynamics_rates) == 2, \
            "dynamics_rates should have 2 elements for 1D"

    def test_positive_net_rate(self):
        """Test that net event rate is always positive."""
        x = np.array([0.5])
        rhoeps = 0.5
        DV = lambda z: z
        D2V = lambda z: np.ones_like(z[0])
        a = lambda z: np.ones_like(z)
        c = lambda z: 0.5
        h = 0.1

        net_rate, _, _, _, _ = event_rates(x, rhoeps, DV, D2V, a, c, h)

        assert net_rate > 0, "Net event rate should be positive"


class TestFlemingViot:
    """Integration tests for fleming_viot function."""

    def test_returns_correct_shapes(self):
        """Test that simulation returns arrays with correct shapes."""
        T = 0.5
        V = lambda z: np.cos(2 * np.pi * z[0]) / (2 * np.pi)
        DV = lambda z: -np.sin(2 * np.pi * z)
        D2V = lambda z: -2 * np.pi * np.cos(2 * np.pi * z[0])
        a = lambda z: np.ones_like(z)
        epsilon = 0.5
        c = lambda z: 0.0
        h0 = 0.1

        Nparticles = 3
        dim = 1
        x_init = np.zeros((Nparticles, dim)) + 0.5
        y_init = np.zeros((Nparticles, dim)) + 0.5

        allx, ally, alltime, allrho = fleming_viot(
            T, V, DV, D2V, a, epsilon, c, h0, x_init, y_init
        )

        n_steps = len(alltime)
        assert allx.shape == (n_steps, Nparticles, dim), \
            f"allx shape mismatch: {allx.shape}"
        assert ally.shape == (n_steps, Nparticles, dim), \
            f"ally shape mismatch: {ally.shape}"
        assert allrho.shape == (n_steps, Nparticles), \
            f"allrho shape mismatch: {allrho.shape}"

    def test_time_reaches_horizon(self):
        """Test that simulation reaches time horizon."""
        T = 1.0
        V = lambda z: z[0]**2
        DV = lambda z: 2 * z
        D2V = lambda z: 2 * np.ones_like(z[0])
        a = lambda z: np.ones_like(z)
        epsilon = 1.0
        c = lambda z: 0.0
        h0 = 0.1

        Nparticles = 2
        dim = 1
        x_init = np.zeros((Nparticles, dim))
        y_init = np.zeros((Nparticles, dim))

        _, _, alltime, _ = fleming_viot(
            T, V, DV, D2V, a, epsilon, c, h0, x_init, y_init
        )

        assert alltime[-1] >= T, "Simulation should reach time horizon"

    def test_rho_in_bounds(self):
        """Test that all rho values are in [0, 1]."""
        T = 0.5
        V = lambda z: np.cos(2 * np.pi * z[0])
        DV = lambda z: -2 * np.pi * np.sin(2 * np.pi * z)
        D2V = lambda z: -4 * np.pi**2 * np.cos(2 * np.pi * z[0])
        a = lambda z: np.ones_like(z)
        epsilon = 0.5
        c = lambda z: 0.0
        h0 = 0.1

        Nparticles = 3
        dim = 1
        x_init = np.random.uniform(-1, 1, (Nparticles, dim))
        y_init = np.random.uniform(-1, 1, (Nparticles, dim))

        _, _, _, allrho = fleming_viot(
            T, V, DV, D2V, a, epsilon, c, h0, x_init, y_init
        )

        assert np.all(allrho >= 0) and np.all(allrho <= 1), \
            "All rho values should be in [0, 1]"


class TestFlemingViotVanilla:
    """Integration tests for fleming_viot_vanilla function."""

    def test_returns_correct_shapes(self):
        """Test that vanilla simulation returns correct shapes."""
        T = 0.5
        b = lambda z: -z  # drift towards origin
        a = lambda z: np.ones_like(z)
        c = lambda z: 0.0
        h0 = 0.1

        Nparticles = 3
        dim = 1
        x_init = np.random.uniform(-1, 1, (Nparticles, dim))

        allpositions, alltime = fleming_viot_vanilla(T, b, a, c, h0, x_init)

        n_steps = len(alltime)
        assert allpositions.shape == (n_steps, Nparticles, dim), \
            f"Shape mismatch: {allpositions.shape}"

    def test_starts_at_initial_positions(self):
        """Test that simulation starts at initial positions."""
        T = 0.5
        b = lambda z: np.zeros_like(z)
        a = lambda z: np.ones_like(z)
        c = lambda z: 0.0
        h0 = 0.1

        Nparticles = 2
        dim = 1
        x_init = np.array([[0.3], [0.7]])

        # Use .copy() because the simulation modifies the array in place
        allpositions, _ = fleming_viot_vanilla(T, b, a, c, h0, x_init.copy())

        assert np.allclose(allpositions[0], x_init), \
            "Should start at initial positions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
