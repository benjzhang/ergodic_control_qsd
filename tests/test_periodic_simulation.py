"""
Unit tests for examples/periodic/periodic_simulation.py

Tests the periodic potential simulation functions.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../examples/periodic'))

from examples.periodic.periodic_simulation import (
    _define_periodic_potential,
    run_vanilla_simulation,
    run_fleming_viot_simulation
)


class TestPeriodicPotential:
    """Tests for the periodic potential definition."""

    def test_potential_periodic(self):
        """Test that V(x) is periodic with period 1."""
        V, DV, D2V, a, c = _define_periodic_potential()

        x_vals = np.linspace(-2, 2, 100)
        for x in x_vals:
            assert np.isclose(V(np.array([x])), V(np.array([x + 1])), rtol=1e-10), \
                f"V should be periodic: V({x}) != V({x+1})"

    def test_potential_extrema(self):
        """Test that V has correct extrema locations."""
        V, DV, D2V, a, c = _define_periodic_potential()

        # V(x) = cos(2πx) / (2π)
        # Minimum at x = 0.5 (where cos = -1)
        # Maximum at x = 0 (where cos = 1)

        V_at_0 = V(np.array([0.0]))
        V_at_half = V(np.array([0.5]))

        assert V_at_0 > V_at_half, "V(0) should be greater than V(0.5)"

    def test_gradient_at_extrema(self):
        """Test that gradient is zero at extrema."""
        V, DV, D2V, a, c = _define_periodic_potential()

        # DV = -sin(2πx) * 2π / (2π) = -sin(2πx)
        # Zero at x = 0 and x = 0.5

        DV_at_0 = DV(np.array([0.0]))
        DV_at_half = DV(np.array([0.5]))

        assert np.isclose(DV_at_0, 0, atol=1e-10), "DV(0) should be 0"
        assert np.isclose(DV_at_half, 0, atol=1e-10), "DV(0.5) should be 0"

    def test_laplacian_sign(self):
        """Test Laplacian sign at extrema."""
        V, DV, D2V, a, c = _define_periodic_potential()

        # D2V = -2π * cos(2πx)
        # At x=0: D2V = -2π (negative -> maximum)
        # At x=0.5: D2V = 2π (positive -> minimum)

        D2V_at_0 = D2V(np.array([0.0]))
        D2V_at_half = D2V(np.array([0.5]))

        assert D2V_at_0 < 0, "D2V(0) should be negative (maximum)"
        assert D2V_at_half > 0, "D2V(0.5) should be positive (minimum)"

    def test_diffusion_constant(self):
        """Test that diffusion is constant."""
        V, DV, D2V, a, c = _define_periodic_potential()

        x_vals = np.linspace(-1, 1, 20)
        a_vals = [a(np.array([x]))[0] for x in x_vals]

        assert all(np.isclose(av, a_vals[0]) for av in a_vals), \
            "Diffusion coefficient should be constant"

    def test_c_is_zero(self):
        """Test that c(x) = 0."""
        V, DV, D2V, a, c = _define_periodic_potential()

        for x in np.linspace(-1, 1, 10):
            assert c(np.array([x])) == 0, "c(x) should be 0"


class TestVanillaSimulation:
    """Tests for run_vanilla_simulation function."""

    def test_returns_correct_structure(self):
        """Test that vanilla simulation returns correct structure."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        assert len(times) == Nparticles, \
            f"Should have {Nparticles} time arrays"
        assert len(positions) == Nparticles, \
            f"Should have {Nparticles} position arrays"

    def test_each_particle_has_trajectory(self):
        """Test that each particle has a non-empty trajectory."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        for i in range(Nparticles):
            assert len(times[i]) > 1, f"Particle {i} should have trajectory"
            assert len(positions[i]) == len(times[i]), \
                f"Particle {i} times and positions should match"

    def test_time_reaches_horizon(self):
        """Test that all particles reach time horizon."""
        T = 1.0
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 2

        times, _ = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        for i, t in enumerate(times):
            # Stochastic timing may not reach exactly T, allow small tolerance
            assert t[-1] >= T - 0.1, f"Particle {i} should approximately reach time T"

    def test_initial_position_in_range(self):
        """Test that initial positions are in [-1, 1]."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 10

        _, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        for i, pos in enumerate(positions):
            x0 = pos[0][0] if pos[0].ndim > 0 else pos[0]
            assert -1 <= x0 <= 1, f"Particle {i} initial position out of range"


class TestFlemingViotSimulation:
    """Tests for run_fleming_viot_simulation function."""

    def test_returns_correct_structure(self):
        """Test that Fleming-Viot simulation returns correct structure."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        allx, ally, alltime, allrho = run_fleming_viot_simulation(
            T, epsilon, h0, Nparticles
        )

        n_steps = len(alltime)
        assert allx.shape == (n_steps, Nparticles, 1), \
            f"allx shape mismatch: {allx.shape}"
        assert ally.shape == (n_steps, Nparticles, 1), \
            f"ally shape mismatch: {ally.shape}"
        # allrho has an extra dimension for compatibility with multi-dimensional systems
        assert allrho.shape[:2] == (n_steps, Nparticles), \
            f"allrho shape mismatch: {allrho.shape}"

    def test_time_monotonic(self):
        """Test that time is non-decreasing."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 2

        _, _, alltime, _ = run_fleming_viot_simulation(T, epsilon, h0, Nparticles)

        # Use non-strict inequality due to floating-point precision
        for i in range(len(alltime) - 1):
            assert alltime[i] <= alltime[i+1], "Time should be non-decreasing"

    def test_rho_in_bounds(self):
        """Test that rho values are in [0, 1]."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        _, _, _, allrho = run_fleming_viot_simulation(T, epsilon, h0, Nparticles)

        assert np.all(allrho >= 0), "rho should be >= 0"
        assert np.all(allrho <= 1), "rho should be <= 1"

    def test_initial_positions_at_zero(self):
        """Test that initial positions are at 0 (center of periodic domain)."""
        T = 0.5
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        allx, ally, _, _ = run_fleming_viot_simulation(T, epsilon, h0, Nparticles)

        # With periodic_bc=True, initial positions are at 0 (center of [-0.5, 0.5])
        assert np.allclose(allx[0], 0.0), "Initial x should be 0.0"
        assert np.allclose(ally[0], 0.0), "Initial y should be 0.0"


class TestSimulationPhysics:
    """Physics-based tests for simulations."""

    def test_low_temperature_concentrates_at_minimum(self):
        """Test that low temperature concentrates particles at potential minimum."""
        T = 5.0  # longer time for equilibration
        epsilon = 0.1  # low temperature
        h0 = 0.05
        Nparticles = 5

        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        # Collect final positions
        final_positions = [pos[-1][0] for pos in positions]

        # At low temperature, particles should concentrate near x = 0.5 (minimum)
        # Use modular distance since potential is periodic
        distances_from_min = []
        for x in final_positions:
            # Distance to nearest 0.5 + k for integer k
            d = min(abs(x - 0.5), abs(x + 0.5), abs(x - 1.5))
            distances_from_min.append(d)

        avg_distance = np.mean(distances_from_min)
        assert avg_distance < 0.3, \
            f"Low temp should concentrate near minimum, avg dist = {avg_distance}"

    def test_high_temperature_explores(self):
        """Test that high temperature explores more of the domain."""
        T = 2.0
        epsilon = 2.0  # high temperature
        h0 = 0.1
        Nparticles = 3

        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        # Collect all positions after burn-in
        all_positions = []
        for i, pos in enumerate(positions):
            # Take positions after t > 0.5
            t = np.array(times[i])
            burn_in_mask = t > 0.5
            all_positions.extend(np.array(pos).flatten()[burn_in_mask])

        # Check that we've explored a range of positions
        pos_range = np.max(all_positions) - np.min(all_positions)
        assert pos_range > 0.5, \
            f"High temp should explore, range = {pos_range}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
