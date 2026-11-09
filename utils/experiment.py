"""Shared experiment helpers for the INS Fleming-Viot examples.

A "problem" module (e.g. examples/periodic/problem.py) provides:
    name, dim, domain, wrap_period
    V(x), DV(x), D2V(x), c(x)          # potential, gradient, Laplacian, kill rate
    diffusion(epsilon) -> a(x)          # jump-process diffusion coefficient callable
and (for validation) analytic_moment / analytic_density.

This module runs the simulations and reduces the Fleming-Viot output into the
weighted empirical measure used for validation and plotting. It is dimension-
agnostic where possible; the 1D density helpers assume a scalar coordinate.
"""

import numpy as np

from .sde_simulation import pure_jump_approx_diffusion
from .fleming_viot import fleming_viot, fleming_viot_vanilla


def wrap(x, period):
    """Wrap coordinates into [-period/2, period/2] (identity if period is None)."""
    x = np.asarray(x, dtype=float)
    if period is None:
        return x
    return np.mod(x + period / 2.0, period) - period / 2.0


def run_vanilla_independent(problem, T, epsilon, N, h0, rng=None):
    """N independent copies of the forward SDE dX = -DV dt + sqrt(eps) dW.

    This is the reference ("vanilla") sampler for pure-sampling problems (c=0).
    Returns lists of per-particle time and position arrays.
    """
    rng = rng or np.random.default_rng()
    a = problem.diffusion(epsilon)
    b = lambda x: -problem.DV(x)
    times, positions = [], []
    for _ in range(N):
        x0 = problem.initial_x(epsilon, rng)
        X, tt = pure_jump_approx_diffusion(T, b, a, h0, x0, wrap_period=problem.wrap_period)
        times.append(tt)
        positions.append(X)
    return times, positions


def run_fv(problem, T, epsilon, N, h0, x0=None, y0=None):
    """Infinite-swap Fleming-Viot run for the given problem."""
    a = problem.diffusion(epsilon)
    if x0 is None:
        x0 = np.zeros((N, problem.dim))
    if y0 is None:
        y0 = np.zeros((N, problem.dim))
    return fleming_viot(T, problem.V, problem.DV, problem.D2V, a, epsilon, problem.c, h0,
                        x0, y0, wrap_period=problem.wrap_period)


def _rho_col(all_rho, m, i):
    """Extract particle-i rho over mask m, handling (n,N) and (n,N,1) shapes."""
    r = np.asarray(all_rho)
    return r[m, i, 0] if r.ndim == 3 else r[m, i]


def fv_weighted_samples_1d(px, py, all_rho, all_time, wrap_period=None, burn_frac=0.0):
    """Flatten a 1D Fleming-Viot run into (positions, weights) for the weighted
    empirical measure: mass dt*rho at x, dt*(1-rho) at y, pooled over particles.
    """
    t = np.asarray(all_time)
    dt = np.append(np.diff(t), 0.0)
    m = t >= burn_frac * t[-1]
    dtm = dt[m]
    pos, wt = [], []
    for i in range(px.shape[1]):
        x = wrap(px[m, i, 0], wrap_period)
        y = wrap(py[m, i, 0], wrap_period)
        r = _rho_col(all_rho, m, i)
        pos += [x, y]
        wt += [dtm * r, dtm * (1.0 - r)]
    return np.concatenate(pos), np.concatenate(wt)


def weighted_moment(pos, wt, g):
    return np.sum(wt * g(pos)) / np.sum(wt)
