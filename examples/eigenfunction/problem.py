"""Eigenfunction (c != 0) example for forward-backward INS Fleming-Viot.

Estimates the principal eigenvalue of the sub-Markov generator

    A = -V'(x) d/dx - eps_op d^2/dx^2 + c(x)

via E[c(X)] under the quasi-stationary distribution (QSD). This is the
c != 0 case, so it exercises BOTH kill/clone channels (forward and backward)
of the simulation scheme.

Convention: ``diffusion(epsilon)`` returns the jump-process diffusion coefficient
a(x) = epsilon (sqrt(epsilon) noise, matching the periodic template), so the
diffusion generator's second-order coefficient is a/2 = epsilon/2 = eps_op. This
matches the documented operator  -sin(2 pi x) phi' - (epsilon/2) phi'' + c phi.
The exact reference is a DIRECT finite-difference solve (not a Schrodinger
transform) with eps_op = epsilon/2.

Unlike the periodic problem, c(x) = x^2/pi depends on absolute position, so the
dynamics is genuinely unbounded and there is NO periodic wrapping
(``wrap_period = None``). ``domain`` is only a nominal bounded window used for
reference integrals / plot ranges.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import eigs

name = "eigenfunction"
dim = 1
domain = (-6.0, 6.0)   # nominal bounded window for reference integrals / plotting
wrap_period = None      # c depends on absolute position -> NO periodic wrapping


def V(x):
    return np.cos(2 * np.pi * x) / (2 * np.pi)


def DV(x):
    return -np.sin(2 * np.pi * x)  # V'


def D2V(x):
    return -2 * np.pi * np.cos(2 * np.pi * x)  # V'' (Laplacian)


def c(x):
    return x ** 2 / np.pi  # NON-ZERO killing rate


def diffusion(epsilon):
    """Jump-process diffusion coefficient a(x) = epsilon  =>  sqrt(epsilon) noise.

    Generator second-order coefficient a/2 = epsilon/2 = eps_op, matching the
    documented operator  -sin(2 pi x) phi' - (epsilon/2) phi'' + c phi. The
    infinite-swap run is driven with swap-temperature epsilon (= a).
    """
    return lambda x: epsilon * np.ones_like(x)


def initial_x(epsilon, rng=None):
    """Particles start at +1 (deterministic; rng is accepted but unused)."""
    return np.ones(dim)


# ---- exact reference: DIRECT finite-difference solve (no Schrodinger transform) ----
# Finite-difference the (non-symmetric) eigenvalue problem of the documented
# operator directly:
#     -sin(2 pi x) phi' - eps_op phi'' + c(x) phi = lambda phi,   eps_op = a/2,
# solved with a non-symmetric sparse eigensolver (eigs, shift-invert at 0).
# NOTE: eigsh is INVALID here (the operator is not symmetric).
# The right eigenfunction phi is NOT the density; the QSD is phi reweighted by the
# forward diffusion's reversible measure e^{-2V/a} = e^{-V/eps_op}:
#     nu ∝ phi(x) * e^{-V(x)/eps_op}.
def exact_eigen(eps_op, L=4.0, N=1001):
    """Return (lambda, xgrid, qsd). Direct FD of -sin(2 pi x) phi' - eps_op phi'' + c phi.

    eps_op is the generator's second-order coefficient (= a/2). Grid [-L, L] with
    N points matches the documented FD window ([-4, 4], dx ~= 0.008)."""
    xg = np.linspace(-L, L, N); dx = xg[1] - xg[0]
    D1 = (eye(N, k=1) - eye(N, k=-1)) / (2 * dx)
    D2 = (eye(N, k=1) - 2 * eye(N) + eye(N, k=-1)) / dx ** 2
    A = csr_matrix(-diags(np.sin(2 * np.pi * xg), 0) @ D1 - eps_op * D2 + diags(c(xg), 0))
    evals, evecs = eigs(A, k=5, sigma=0.0, which='LM')
    idx = np.argsort(evals.real); lam = float(evals.real[idx][0])
    phi = np.abs(evecs[:, idx][:, 0].real)
    nu = phi * np.exp(-V(xg) / eps_op); nu /= np.trapezoid(nu, xg)
    return lam, xg, nu


# ---- convenience wrappers (mirror periodic's analytic_* helpers) ----
def analytic_eigenvalue(epsilon):
    """Principal eigenvalue (eps_op = epsilon/2 = a/2)."""
    lam, _, _ = exact_eigen(epsilon / 2)
    return lam


def analytic_qsd(epsilon):
    """(xgrid, qsd density) of the quasi-stationary distribution."""
    _, xg, nu = exact_eigen(epsilon / 2)
    return xg, nu
