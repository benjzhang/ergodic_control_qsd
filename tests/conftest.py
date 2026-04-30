"""
Pytest configuration and shared fixtures for ergodic_control_qsd tests.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset to random state after test
    np.random.seed(None)


@pytest.fixture
def simple_1d_path():
    """Simple 1D particle path for testing."""
    return np.array([[0], [1], [2], [1], [0], [1], [2]])


@pytest.fixture
def simple_times():
    """Simple time array for testing."""
    return np.array([0, 1, 2, 3, 4, 5, 6])


@pytest.fixture
def periodic_potential():
    """Periodic cosine potential and derivatives."""
    def V(x):
        return np.cos(2 * np.pi * x) / (2 * np.pi)

    def DV(x):
        return -np.sin(2 * np.pi * x)

    def D2V(x):
        return -2 * np.pi * np.cos(2 * np.pi * x)

    return V, DV, D2V


@pytest.fixture
def quadratic_potential():
    """Simple quadratic potential and derivatives."""
    def V(x):
        return 0.5 * x[0]**2

    def DV(x):
        return x

    def D2V(x):
        return np.ones_like(x[0])

    return V, DV, D2V


@pytest.fixture
def unit_diffusion():
    """Unit diffusion coefficient."""
    return lambda x: np.ones_like(x)


@pytest.fixture
def zero_kill_rate():
    """Zero kill rate function."""
    return lambda x: 0.0


@pytest.fixture
def simulation_params():
    """Default simulation parameters for testing."""
    return {
        'T': 1.0,
        'epsilon': 0.5,
        'h0': 0.1,
        'Nparticles': 3,
        'dim': 1
    }


# Configure pytest to show more output
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Make tests reproducible by default
@pytest.fixture(autouse=True)
def set_random_seed_auto():
    """Automatically set random seed before each test."""
    np.random.seed(12345)
    yield
