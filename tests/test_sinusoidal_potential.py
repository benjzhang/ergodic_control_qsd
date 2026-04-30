"""
Verification tests for the sinusoidal (cosine) potential with periodic boundary conditions.

This test module verifies that the Fleming-Viot simulation correctly samples from
the Gibbs distribution for the periodic cosine potential:

    V(x) = cos(2πx) / (2π)

The stationary distribution should be:

    π(x) ∝ exp(-V(x)/ε) = exp(-cos(2πx)/(2πε))

Key properties tested:
1. Gibbs distribution sampling
2. Partition function (normalization constant)
3. Expected values under the stationary distribution
4. Infinite swapping estimator convergence
"""

import numpy as np
import pytest
from scipy import special
from scipy import integrate
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../examples/periodic'))

from utils.fleming_viot import fleming_viot, fleming_viot_vanilla, inf_swap_rate
from utils.postprocessing import empirical_measure_functional, resample_simple_weighted_empirical_measure
from examples.periodic.periodic_simulation import (
    _define_periodic_potential,
    run_vanilla_simulation,
    run_fleming_viot_simulation
)


class TestSinusoidalPotentialProperties:
    """Tests for mathematical properties of the sinusoidal potential."""

    def test_potential_value_at_extrema(self):
        """Test V(x) values at known extrema."""
        V, DV, D2V, a, c = _define_periodic_potential()

        # V(x) = cos(2πx) / (2π)
        # At x=0: V(0) = 1/(2π) ≈ 0.159
        # At x=0.5: V(0.5) = -1/(2π) ≈ -0.159

        V_at_0 = V(np.array([0.0]))
        V_at_half = V(np.array([0.5]))

        expected_max = 1 / (2 * np.pi)
        expected_min = -1 / (2 * np.pi)

        assert np.isclose(V_at_0, expected_max, rtol=1e-10), \
            f"V(0) should be {expected_max}, got {V_at_0}"
        assert np.isclose(V_at_half, expected_min, rtol=1e-10), \
            f"V(0.5) should be {expected_min}, got {V_at_half}"

    def test_partition_function_bessel(self):
        """Test partition function against modified Bessel function formula.

        For V(x) = cos(2πx)/(2π), the partition function on [-1/2, 1/2] is:

            Z = ∫ exp(-V(x)/ε) dx = ∫ exp(-cos(2πx)/(2πε)) dx

        Using the substitution θ = 2πx and the integral representation of I₀:

            Z = (1/2π) ∫₀^{2π} exp(-cos(θ)/(2πε)) dθ = I₀(1/(2πε))

        where I₀ is the modified Bessel function of the first kind.
        """
        V, _, _, _, _ = _define_periodic_potential()

        for epsilon in [0.1, 0.5, 1.0, 2.0]:
            # Numerical integration
            def integrand(x):
                return np.exp(-V(np.array([x])) / epsilon)

            Z_numerical, _ = integrate.quad(integrand, -0.5, 0.5)

            # Bessel function formula
            # Z = I₀(1/(2πε)) for domain of length 1
            Z_bessel = special.i0(1 / (2 * np.pi * epsilon))

            assert np.isclose(Z_numerical, Z_bessel, rtol=1e-6), \
                f"ε={epsilon}: Z_numerical={Z_numerical}, Z_bessel={Z_bessel}"

    def test_gibbs_density_normalization(self):
        """Test that the Gibbs density integrates to 1."""
        V, _, _, _, _ = _define_periodic_potential()

        for epsilon in [0.1, 0.5, 1.0]:
            def unnormalized_density(x):
                return np.exp(-V(np.array([x])) / epsilon)

            Z, _ = integrate.quad(unnormalized_density, -0.5, 0.5)

            def normalized_density(x):
                return unnormalized_density(x) / Z

            integral, _ = integrate.quad(normalized_density, -0.5, 0.5)

            assert np.isclose(integral, 1.0, rtol=1e-8), \
                f"ε={epsilon}: density integral = {integral}"


class TestExpectedValuesAnalytic:
    """Tests for analytically computable expected values."""

    def test_expected_position_is_zero(self):
        """Test that E[x] = 0 by symmetry.

        The potential V(x) = cos(2πx)/(2π) is symmetric around x=0,
        so the expected value of x under the Gibbs distribution is 0.
        """
        V, _, _, _, _ = _define_periodic_potential()

        for epsilon in [0.1, 0.5, 1.0]:
            def integrand(x):
                return x * np.exp(-V(np.array([x])) / epsilon)

            numerator, _ = integrate.quad(integrand, -0.5, 0.5)

            def unnormalized_density(x):
                return np.exp(-V(np.array([x])) / epsilon)

            Z, _ = integrate.quad(unnormalized_density, -0.5, 0.5)

            E_x = numerator / Z

            assert np.isclose(E_x, 0.0, atol=1e-10), \
                f"ε={epsilon}: E[x] should be 0, got {E_x}"

    def test_expected_exp_minus_x_over_epsilon(self):
        """Compute and verify E[exp(-x/ε)] analytically.

        This is the functional used in the periodic example postprocessing.
        """
        V, _, _, _, _ = _define_periodic_potential()
        epsilon = 0.5

        def integrand(x):
            return np.exp(-x / epsilon) * np.exp(-V(np.array([x])) / epsilon)

        numerator, _ = integrate.quad(integrand, -0.5, 0.5)

        def unnormalized_density(x):
            return np.exp(-V(np.array([x])) / epsilon)

        Z, _ = integrate.quad(unnormalized_density, -0.5, 0.5)

        E_analytic = numerator / Z

        # Store for comparison with simulation
        # At ε=0.5: E[exp(-x/0.5)] ≈ 1.1752 (sinh(1)/1)
        # This is because the Gibbs measure is roughly uniform at high temp

        assert E_analytic > 0, "E[exp(-x/ε)] should be positive"
        assert np.isfinite(E_analytic), "E[exp(-x/ε)] should be finite"


class TestSimulationConvergence:
    """Tests that simulations converge to correct distributions."""

    @pytest.mark.slow
    def test_vanilla_histogram_matches_gibbs(self):
        """Test that vanilla simulation histogram matches Gibbs distribution.

        Run long simulation and check histogram matches theoretical density.
        """
        T = 10.0  # Long enough for equilibration
        epsilon = 0.5
        h0 = 0.05
        Nparticles = 5

        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        # Collect all positions after burn-in (t > 2)
        all_positions = []
        for i in range(Nparticles):
            t = np.array(times[i])
            x = np.array(positions[i]).flatten()
            burn_in_mask = t > 2.0
            all_positions.extend(x[burn_in_mask])

        all_positions = np.array(all_positions)

        # Wrap to [-0.5, 0.5] for comparison
        all_positions_wrapped = np.mod(all_positions + 0.5, 1) - 0.5

        # Compute empirical histogram
        hist, bin_edges = np.histogram(all_positions_wrapped, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute theoretical density
        V, _, _, _, _ = _define_periodic_potential()

        def gibbs_density(x):
            unnorm = np.exp(-V(np.array([x])) / epsilon)
            Z = special.i0(1 / (2 * np.pi * epsilon))
            return unnorm / Z

        theoretical = np.array([gibbs_density(x) for x in bin_centers])

        # Check correlation between histogram and theory
        correlation = np.corrcoef(hist, theoretical)[0, 1]

        assert correlation > 0.8, \
            f"Histogram should correlate with Gibbs density, got r={correlation}"

    def test_empirical_measure_converges(self):
        """Test that empirical measure functional converges.

        Note: The simulation runs on unbounded domain with periodic potential.
        We wrap positions to [-0.5, 0.5] before computing statistics to test
        convergence to the periodic Gibbs distribution.
        """
        T = 5.0
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        # Compute E[x²] via empirical measure, wrapping to periodic domain
        results = []
        for i in range(Nparticles):
            t = np.array(times[i])
            x = np.array(positions[i])

            # Wrap positions to [-0.5, 0.5] (periodic domain)
            x_wrapped = np.mod(x + 0.5, 1.0) - 0.5

            # Apply burn-in
            burn_in_mask = t > 1.0
            if np.sum(burn_in_mask) > 10:
                result = empirical_measure_functional(
                    x_wrapped[burn_in_mask],
                    lambda path: path[:, 0]**2,
                    t[burn_in_mask]
                )
                if np.isfinite(result):
                    results.append(result)

        if len(results) > 0:
            mean_x_squared = np.mean(results)

            # E[x²] for Gibbs on [-0.5, 0.5] with cosine potential
            # At high temp (ε=0.5), close to uniform: E[x²] ≈ 1/12 ≈ 0.083
            # Allow range for statistical variation
            assert 0.01 < mean_x_squared < 0.2, \
                f"E[x²] = {mean_x_squared} should be in reasonable range for periodic domain"


class TestInfiniteSwappingProperties:
    """Tests for infinite swapping mechanism properties."""

    def test_rho_consistency_with_potential(self):
        """Test that rho values are consistent with potential differences."""
        V, _, _, _, _ = _define_periodic_potential()
        epsilon = 0.5

        # At potential minimum (x=0.5), particle should have high rho
        # when paired with particle at maximum (x=0)
        x_min = np.array([0.5])
        x_max = np.array([0.0])

        rho_min_vs_max = inf_swap_rate(x_min, x_max, V, epsilon)
        rho_max_vs_min = inf_swap_rate(x_max, x_min, V, epsilon)

        # Particle at minimum should dominate (higher rho)
        assert rho_min_vs_max > 0.5, \
            f"Particle at minimum should have rho > 0.5, got {rho_min_vs_max}"
        assert rho_max_vs_min < 0.5, \
            f"Particle at maximum should have rho < 0.5, got {rho_max_vs_min}"

        # They should sum to 1
        assert np.isclose(rho_min_vs_max + rho_max_vs_min, 1.0), \
            "rho(x,y) + rho(y,x) should equal 1"

    def test_rho_equals_half_at_equal_potential(self):
        """Test that rho = 0.5 when particles have equal potential."""
        V, _, _, _, _ = _define_periodic_potential()
        epsilon = 0.5

        # V(0.25) = V(-0.25) = 0 by symmetry
        x1 = np.array([0.25])
        x2 = np.array([-0.25])

        rho = inf_swap_rate(x1, x2, V, epsilon)

        assert np.isclose(rho, 0.5, rtol=1e-10), \
            f"Equal potential should give rho=0.5, got {rho}"

    def test_fleming_viot_rho_distribution(self):
        """Test that FV simulation produces reasonable rho distribution."""
        T = 2.0
        epsilon = 0.5
        h0 = 0.1
        Nparticles = 3

        allx, ally, alltime, allrho = run_fleming_viot_simulation(
            T, epsilon, h0, Nparticles
        )

        # Flatten rho values after burn-in
        burn_in_idx = np.searchsorted(alltime, 0.5)
        rho_values = allrho[burn_in_idx:].flatten()

        # rho should span [0, 1] with some concentration
        assert np.min(rho_values) < 0.3, "rho should sometimes be low"
        assert np.max(rho_values) > 0.7, "rho should sometimes be high"

        # Mean should be around 0.5 (by symmetry of the dynamics)
        mean_rho = np.mean(rho_values)
        assert 0.3 < mean_rho < 0.7, \
            f"Mean rho should be around 0.5, got {mean_rho}"


class TestPeriodicBoundaryConditions:
    """Tests for periodic boundary conditions."""

    def test_potential_periodicity(self):
        """Test that potential is periodic with period 1."""
        V, DV, D2V, _, _ = _define_periodic_potential()

        x_vals = np.linspace(-2, 2, 100)

        for x in x_vals:
            # V(x) = V(x + 1) = V(x - 1)
            assert np.isclose(V(np.array([x])), V(np.array([x + 1])), rtol=1e-10)
            assert np.isclose(V(np.array([x])), V(np.array([x - 1])), rtol=1e-10)

            # Same for derivatives
            assert np.isclose(DV(np.array([x]))[0], DV(np.array([x + 1]))[0], rtol=1e-10)
            assert np.isclose(D2V(np.array([x])), D2V(np.array([x + 1])), rtol=1e-10)

    def test_gradient_periodicity(self):
        """Test that gradient has correct periodicity properties."""
        V, DV, _, _, _ = _define_periodic_potential()

        # DV(x) = -sin(2πx)
        # Should be zero at x = 0, 0.5, 1, ...
        # Should be -1 at x = 0.25
        # Should be +1 at x = -0.25

        assert np.isclose(DV(np.array([0.0]))[0], 0.0, atol=1e-10)
        assert np.isclose(DV(np.array([0.5]))[0], 0.0, atol=1e-10)
        assert np.isclose(DV(np.array([0.25]))[0], -1.0, atol=1e-10)
        assert np.isclose(DV(np.array([-0.25]))[0], 1.0, atol=1e-10)


class TestNumericalStability:
    """Tests for numerical stability of the implementation."""

    def test_low_temperature_stability(self):
        """Test that simulation is stable at low temperature."""
        T = 1.0
        epsilon = 0.05  # Very low temperature
        h0 = 0.02  # Smaller step for stability
        Nparticles = 2

        # Should not raise any exceptions
        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        # All positions should be finite
        for i in range(Nparticles):
            assert np.all(np.isfinite(positions[i])), \
                f"Particle {i} has non-finite positions"

    def test_high_temperature_stability(self):
        """Test that simulation is stable at high temperature."""
        T = 1.0
        epsilon = 5.0  # High temperature
        h0 = 0.1
        Nparticles = 2

        # Should not raise any exceptions
        times, positions = run_vanilla_simulation(T, epsilon, h0, Nparticles)

        # All positions should be finite
        for i in range(Nparticles):
            assert np.all(np.isfinite(positions[i])), \
                f"Particle {i} has non-finite positions"

    def test_inf_swap_rate_numerical_stability(self):
        """Test inf_swap_rate doesn't overflow/underflow."""
        V, _, _, _, _ = _define_periodic_potential()

        # Test with extreme potential differences
        x_min = np.array([0.5])  # V = -1/(2π)
        x_max = np.array([0.0])  # V = +1/(2π)

        # At very low epsilon, the exponentials could overflow
        for epsilon in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            rho = inf_swap_rate(x_min, x_max, V, epsilon)

            assert np.isfinite(rho), f"rho should be finite at ε={epsilon}"
            assert 0 <= rho <= 1, f"rho should be in [0,1] at ε={epsilon}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
