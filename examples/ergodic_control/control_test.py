"""Stage 4: build the control from the learned score and test it.

Control (a = P.a_coeff):  u*(x) = a * s_theta(x) + 2 grad V(x)  ~ a grad log phi.
For speed the drift fields are precomputed once on a grid (a single torch forward
for the score), then the Euler-Maruyama loops run in pure numpy with PERIODIC
bilinear interpolation (scipy.ndimage.map_coordinates, mode='grid-wrap').

    T2  drift = (a/2) grad log nu = s_theta          -> should sample the QSD nu
    T3  drift = -grad V + u* = grad V + a s_theta      -> should sample rho* prop. phi^2 m
Tests:  T2 vs FD nu | T3 vs FD rho* | T4 ergodic cost E_rho*[c+|u*|^2/2a] vs lambda.

    python control_test.py [--walkers 4000 --steps 15000 --dt 1e-3 --gridN 256]
"""
import argparse
import os

import numpy as np
import scipy.ndimage as ndi
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem as P
from score_match import ScoreNet, score_np

L = P.L
a = P.a_coeff
HERE = os.path.dirname(__file__)


def interp(field, X, h):
    """Periodic bilinear interpolation of field (Ng,Ng,2) at points X (M,2)."""
    coords = np.stack([X[:, 0] / h - 0.5, X[:, 1] / h - 0.5])          # index space (2,M)
    d0 = ndi.map_coordinates(field[..., 0], coords, order=1, mode="grid-wrap")
    d1 = ndi.map_coordinates(field[..., 1], coords, order=1, mode="grid-wrap")
    return np.stack([d0, d1], axis=1)


def simulate(field, h, walkers, steps, dt, burn, rng):
    X = rng.uniform(0, L, size=(walkers, 2))
    sq = np.sqrt(a * dt)
    keep = []
    for k in range(steps):
        X = np.mod(X + interp(field, X, h) * dt + sq * rng.standard_normal(X.shape), L)
        if k > burn * steps and k % 10 == 0:
            keep.append(X.copy())
    return np.concatenate(keep, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkers", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--burn", type=float, default=0.3)
    ap.add_argument("--gridN", type=int, default=256)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    ck = torch.load(os.path.join(HERE, "score_model.pt"), map_location="cpu")
    model = ScoreNet(n_harm=ck["n_harm"]); model.load_state_dict(ck["state_dict"]); model.eval()
    fd = np.load(os.path.join(HERE, "fd_reference.npy"), allow_pickle=True).item()
    Nfd = fd["N"]

    # ---- precompute drift fields on a grid (one torch forward) ----
    Ng = args.gridN; h = L / Ng
    g = (np.arange(Ng) + 0.5) * h
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1)
    s_grid = score_np(model, pts).reshape(Ng, Ng, 2)          # s_theta on grid
    gv_grid = P.DV(pts).reshape(Ng, Ng, 2)                    # grad V on grid
    drift_nu = s_grid                                         # targets nu
    drift_rho = gv_grid + a * s_grid                          # -gradV + u* ; targets rho*

    print("simulating nu-targeting Langevin (T2) ...", flush=True)
    samp_nu = simulate(drift_nu, h, args.walkers, args.steps, args.dt, args.burn, rng)
    print("simulating rho*-targeting controlled SDE (T3) ...", flush=True)
    samp_rho = simulate(drift_rho, h, args.walkers, args.steps, args.dt, args.burn, rng)

    # T4: ergodic cost under rho*   (u* = a s + 2 gradV)
    u = a * interp(s_grid, samp_rho, h) + 2 * P.DV(samp_rho)
    cost = float(np.mean(P.c(samp_rho) + np.sum(u ** 2, axis=1) / (2 * a)))
    print(f"T4  ergodic cost E_rho*[c + |u*|^2/2a] = {cost:.4f}   (lambda_FD = {fd['lam']:.4f})", flush=True)

    def h2(pts_):
        H, _, _ = np.histogram2d(pts_[:, 0], pts_[:, 1], bins=Nfd // 2, range=[[0, L], [0, L]])
        return H

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(fd["nu"].reshape(Nfd, Nfd).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[0, 0].set_title(r"FD  $\nu$")
    ax[0, 1].imshow(h2(samp_nu).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[0, 1].set_title(r"controlled (drift $s_\theta$) $\to\nu$")
    ax[1, 0].imshow(fd["rho"].reshape(Nfd, Nfd).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[1, 0].set_title(r"FD  $\rho^*$")
    ax[1, 1].imshow(h2(samp_rho).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[1, 1].set_title(r"controlled ($u^*$) $\to\rho^*$")
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"Control test: learned score -> QSD (top) and rho* (bottom)\n"
                 f"T4 ergodic cost {cost:.3f} vs lambda {fd['lam']:.3f}")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "control_test.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/control_test.png", flush=True)


if __name__ == "__main__":
    main()
