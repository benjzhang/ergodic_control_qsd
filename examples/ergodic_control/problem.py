"""Ergodic-control / QSD example: 2D array-of-Gaussians mixture + killing.

Base dynamics (uncontrolled):  dX = -grad V(X) dt + sqrt(a) dW  on the torus [0,4]^2,
with  a = 2  and  V = -log(mixture)  so the base invariant measure is exactly the
16-Gaussian mixture:
    m(x) prop. sum_{k} exp(-|x - c_k|^2 / (2 sigma^2)),   V = -log m.

We add a killing rate c(x) >= 0 (periodic, non-uniform), turning this into a
quasi-stationary / ergodic-control problem.  Principal eigenpair (lambda, phi) of
the sub-generator L - c:
    (L - c) phi = -lambda phi,   L f = -grad V . grad f + (a/2) Laplacian f.
Associated measures:
    QSD                    nu   prop. phi   * m
    optimally-controlled   rho* prop. phi^2 * m

Ergodic-control reading: lambda = min_u lim (1/T) E int_0^T ( c + |u|^2/(2a) ) dt
subject to dX = (-grad V + u) dt + sqrt(a) dW; the optimal control is the score of
the eigenfunction, u*(x) = a grad log phi(x), and (using nu prop. phi m):
    u* = a grad log nu + 2 grad V,
so learning grad log nu (score of the QSD, sampled by Fleming-Viot / INS) yields
the control.  The optimally controlled process then samples rho* prop. phi^2 m.

Config chosen by exploration: sigma=0.2 wells on a 4x4 torus grid, "corner" killing
c(x) = kappa (2 - cos(2 pi x1/L) - cos(2 pi x2/L)) with kappa=2.0 -> QSD/rho*
concentrate in the 4 low-killing corner wells.
"""
import numpy as np
from scipy.special import logsumexp

name = "ergodic_control"
dim = 2
L = 4.0
domain = (0.0, L)
period = L
a_coeff = 2.0                      # diffusion a; base invariant m = mixture
sigma = 0.2
K = 4
centers = np.array([(i + 0.5, j + 0.5) for i in range(K) for j in range(K)])  # 16 wells, tile torus
kappa = 2.0
killing_pattern = "corner"


def wrap(x):
    return np.mod(x, L)


def _delta(x):
    """Minimum-image displacement x - center on the torus, shape (..., K^2, dim)."""
    d = wrap(x)[..., None, :] - centers
    return (d + L / 2.0) % L - L / 2.0


def log_mixture(x):
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)   # (..., K^2)
    return logsumexp(q, axis=-1)                       # log sum_k exp(...)


def V(x):
    return -log_mixture(x)                             # V = -log m


def DV(x):
    """grad V = -grad log m = sum_k softmax_k (x - c_k)/sigma^2."""
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(q - logsumexp(q, axis=-1, keepdims=True))
    return np.sum(w[..., None] * d / sigma ** 2, axis=-2)


def D2V(x):
    """Laplacian of V = -log m:  dim/sigma^2 - sum_k w_k |x-c_k|^2/sigma^4 + |grad V|^2."""
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(q - logsumexp(q, axis=-1, keepdims=True))
    r2 = np.sum(d ** 2, axis=-1)
    return dim / sigma ** 2 - np.sum(w * r2 / sigma ** 4, axis=-1) + np.sum(DV(x) ** 2, axis=-1)


def c(x):
    """Killing rate (periodic, non-uniform)."""
    xw = wrap(x)
    x1, x2 = xw[..., 0], xw[..., 1]
    if killing_pattern == "corner":
        g = 2 - np.cos(2 * np.pi * x1 / L) - np.cos(2 * np.pi * x2 / L)
    elif killing_pattern == "center":
        g = 2 + np.cos(2 * np.pi * x1 / L) + np.cos(2 * np.pi * x2 / L)
    else:
        raise ValueError(killing_pattern)
    return kappa * g


def diffusion(a=a_coeff):
    """Jump-process diffusion coefficient a(x) = a (constant)."""
    return lambda x: a * np.ones_like(x)


def mixture_density(x):
    """Normalized base invariant m(x) = mixture (per-Gaussian normalization on R^2)."""
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)
    return np.exp(logsumexp(q, axis=-1)) / (K * K * 2 * np.pi * sigma ** 2)
