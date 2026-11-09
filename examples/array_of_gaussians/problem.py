"""Array-of-Gaussians example for forward-backward INS Fleming-Viot (2D).

Target Gibbs measure on the torus [0, 4)^2 with a 4x4 (K=4) array of sharp
Gaussian wells (sigma = 0.1) centered at {1,2,3,4}^2 -> 16 Gaussians:

    pi(x) prop. sum_{n=1}^K sum_{m=1}^K exp(-((x1-n)^2+(x2-m)^2)/(2 sigma^2)),
    V(x) = -log pi(x) = -logsumexp_k [ -|x - c_k|^2 / (2 sigma^2) ].

Periodicity on [0,4]^2 is enforced via minimum-image distances (see _delta).

Sharp wells (sigma = 0.1 -> D2V ~ 200) make the backward kill/clone event rate
large; the horizon T is correspondingly expensive.

Forward SDE:  dX = -DV(X) dt + sqrt(a) dW,  c(x) = 0.

NOTE on wrapping conventions:
    This problem wraps coordinates to [left, right) = [0, 4) via
        wrap(x) = np.mod(x - left, period) + left.
    That is the convention used inside V/DV/D2V and for post-processing the
    trajectories. It is DIFFERENT from utils.experiment.wrap (and from the
    wrap_period logic inside the simulators), which use the symmetric
    [-period/2, period/2] convention. Runners for this example should use the
    `wrap` function defined here, not the shared one.
"""

import numpy as np
from scipy.special import logsumexp

name = "aog"
dim = 2
left, right = 0.0, 4.0
domain = (left, right)
period = right - left          # 4.0
wrap_period = period           # domain width, for template consistency
sigma = 0.1

# K x K array of Gaussian centers (K=4) at {1,2,3,4}^2 -> 16 Gaussians, matching
# pi(x) prop. sum_{n=1}^K sum_{m=1}^K exp(-((x1-n)^2+(x2-m)^2)/(2 sigma^2)).
# Periodicity on [0,4]^2 is enforced by minimum-image distances in _delta().
K = 4
centers = np.array([(n, m) for n in range(1, K + 1) for m in range(1, K + 1)])


def wrap(x):
    """AOG-style wrap into [left, right) = [0, 4). NOT the symmetric convention."""
    return np.mod(x - left, period) + left


def _delta(x):
    """Minimum-image displacement (x - center) on the periodic torus, shape (..., K^2, dim)."""
    xw = wrap(x)
    dlt = xw[..., None, :] - centers
    return np.mod(dlt + period / 2.0, period) - period / 2.0


def V(x):
    dlt = _delta(x)
    d = -np.sum(dlt ** 2, axis=-1) / (2 * sigma ** 2)
    return -logsumexp(d, axis=-1)


def DV(x):
    dlt = _delta(x)
    d = -np.sum(dlt ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(d - logsumexp(d, axis=-1, keepdims=True))
    return np.sum(w[..., None] * dlt / (sigma ** 2), axis=-2)


def D2V(x):
    dlt = _delta(x)
    d = -np.sum(dlt ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(d - logsumexp(d, axis=-1, keepdims=True))
    lap = -np.sum(w * (-2 * d / (sigma ** 2) - dim / (sigma ** 2)), axis=-1) \
        + np.sum(DV(x) ** 2, axis=-1)
    return lap


def c(x):
    return 0  # no forward killing; backward kill/clone comes from the D2V term


def diffusion(epsilon):
    """Jump-process diffusion coefficient a(x) = 2*epsilon (AOG convention)."""
    return lambda x: 2 * epsilon * np.ones_like(x)


def initial_ensemble(N):
    """Initial forward (x0) and backward (y0) ensembles, shape (N, dim), all at 2.0."""
    x0 = np.zeros((N, dim)) + 2.0
    y0 = np.zeros((N, dim)) + 2.0
    return x0, y0
