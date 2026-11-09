"""FD reference for the SMOOTH ergodic-control AoG (problem_smooth.py).

Same non-symmetric shift-invert eigensolve as fd_reference.py, but importing the
smooth problem.  Saves fd_smooth.npy (lambda + fields) + a figure.

    python fd_smooth.py [--N 256]
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem_smooth as P

L = P.L
a = P.a_coeff


def periodic_ops(N, h):
    o = np.ones(N - 1)
    D1 = sp.diags([o, -o], [1, -1], format="lil")
    D1[0, N - 1] = -1.0; D1[N - 1, 0] = 1.0
    D1 = (D1 / (2 * h)).tocsr()
    D2 = sp.diags([o, -2 * np.ones(N), o], [1, 0, -1], format="lil")
    D2[0, N - 1] = 1.0; D2[N - 1, 0] = 1.0
    D2 = (D2 / h ** 2).tocsr()
    return D1, D2


def fd_solve(N=256):
    h = L / N
    g = (np.arange(N) + 0.5) * h
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1)

    m = P.mixture_density(pts); m = m / (m.sum() * h ** 2)
    gradV = P.DV(pts)
    cc = P.c(pts)

    D1, D2 = periodic_ops(N, h)
    I = sp.identity(N, format="csr")
    d1 = sp.kron(D1, I); d2 = sp.kron(I, D1); lap = sp.kron(D2, I) + sp.kron(I, D2)
    Aop = (-sp.diags(gradV[:, 0]) @ d1 - sp.diags(gradV[:, 1]) @ d2
           + (a / 2) * lap - sp.diags(cc)).tocsc()

    vals, vecs = eigs(Aop, k=6, sigma=0.0, which="LM")
    idx = np.argsort(-vals.real)
    lam = float(-vals.real[idx[0]])
    phi = np.abs(vecs[:, idx[0]].real)

    nu = phi * m; nu /= nu.sum() * h ** 2
    rho = phi ** 2 * m; rho /= rho.sum() * h ** 2
    log_nu = np.log(np.clip(nu, 1e-300, None))
    grad_log_nu = np.stack([d1 @ log_nu, d2 @ log_nu], axis=1)
    return dict(N=N, h=h, g=g, pts=pts, lam=lam, m=m, c=cc, phi=phi, nu=nu,
                rho=rho, grad_log_nu=grad_log_nu, sigma=P.sigma, kappa=P.kappa, a=a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    args = ap.parse_args()
    here = os.path.dirname(__file__)
    r = fd_solve(args.N)
    print(f"lambda = {r['lam']:.5f}  (N={r['N']}, sigma={r['sigma']}, kappa={r['kappa']}, a={r['a']})")
    np.save(os.path.join(here, "fd_smooth.npy"), r, allow_pickle=True)
    N = r["N"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
    for a_, (fld, ttl) in zip(ax, [("m", "base mixture m"), ("c", "killing c(x)"),
                                   ("nu", r"QSD $\nu\propto\phi m$"),
                                   ("rho", r"controlled $\rho^*\propto\phi^2 m$")]):
        a_.imshow(r[fld].reshape(N, N).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
        a_.set_title(ttl); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"Array-of-Gaussians QSD problem  (exact eigenvalue $\\lambda$={r['lam']:.4f})")
    os.makedirs(os.path.join(here, "figures"), exist_ok=True)
    fig.savefig(os.path.join(here, "figures", "fd_smooth.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved fd_smooth.npy + figures/fd_smooth.png")


if __name__ == "__main__":
    main()
