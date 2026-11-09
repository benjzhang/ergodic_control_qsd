"""Periodic diffusion example for forward-backward INS Fleming-Viot.

Target Gibbs measure:
    rho^eps(x)  proportional to  exp(-2 V(x)/eps) = exp(-cos(2 pi x)/(pi eps)),  x in [-1, 1].

Forward SDE:  dX = -DV(X) dt + sqrt(eps) dW      (drift = sin 2 pi x),  c(x) = 0.

Note the diffusion is sqrt(eps): the jump-process diffusion coefficient is a(x)=eps
(generator (eps/2) d^2/dx^2), so the invariant measure is exp(-2V/eps), matching the
infinite-swap weight rho = 1/(exp(2(V(x)-V(y))/eps)+1).
"""

import numpy as np

name = "periodic"
dim = 1
domain = (-1.0, 1.0)
wrap_period = 2.0  # domain [-1, 1]


def V(x):
    return np.cos(2 * np.pi * x) / (2 * np.pi)


def DV(x):
    return -np.sin(2 * np.pi * x)  # V'


def D2V(x):
    return -2 * np.pi * np.cos(2 * np.pi * x)  # V'' (Laplacian)


def c(x):
    return 0  # no forward killing; backward kill/clone comes from the D2V term


def diffusion(epsilon):
    """Jump-process diffusion coefficient a(x)=eps  =>  sqrt(eps) noise."""
    return lambda x: epsilon * np.ones_like(x)


def initial_x(epsilon, rng):
    return np.array([rng.uniform(*domain)])


# ---- analytic target rho^eps ∝ exp(-2V/eps) on the domain, for validation ----
def _target(epsilon, n=8001):
    xg = np.linspace(domain[0], domain[1], n)
    w = np.exp(-2.0 * V(xg) / epsilon)
    w /= np.trapezoid(w, xg)
    return xg, w


def analytic_density(epsilon, n=1000):
    return _target(epsilon, n)


def analytic_moment(g, epsilon, n=8001):
    xg, w = _target(epsilon, n)
    return np.trapezoid(g(xg) * w, xg)
