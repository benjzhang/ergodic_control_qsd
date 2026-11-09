"""Clean method demo: score matching -> control, using samples drawn directly from
the FD QSD nu (isolates the score->control method from the FV/INS sampler bias).

Draws i.i.d. nu-samples via grid inverse-CDF, trains the score, then runs the same
controlled-SDE test as control_test.py.  Expectation: control recovers FD rho* and
the ergodic cost ~ lambda.

    python clean_demo.py [--nsamp 80000 --epochs 4000 --sigma 0.04]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem as P
from score_match import train, score_np
from control_test import interp, simulate

L = P.L
a = P.a_coeff
HERE = os.path.dirname(__file__)


def draw_from_field(field2d, g, h, nsamp, rng):
    """i.i.d. samples from a density given on a cell-centered grid (inverse-CDF + jitter)."""
    p = field2d.ravel().astype(float); p = np.clip(p, 0, None); p /= p.sum()
    idx = rng.choice(len(p), size=nsamp, p=p)
    Ng = len(g)
    ix, iy = idx // Ng, idx % Ng
    x = g[ix] + rng.uniform(-h / 2, h / 2, nsamp)
    y = g[iy] + rng.uniform(-h / 2, h / 2, nsamp)
    return np.mod(np.stack([x, y], 1), L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamp", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--harmonics", type=int, default=6)
    ap.add_argument("--walkers", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--dt", type=float, default=1e-3)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    fd = np.load(os.path.join(HERE, "fd_reference.npy"), allow_pickle=True).item()
    N = fd["N"]; g = fd["g"]; h = fd["h"]; pts = fd["pts"]

    # 1) i.i.d. samples from the TRUE QSD nu
    nu2d = fd["nu"].reshape(N, N)
    samples = draw_from_field(nu2d, g, h, args.nsamp, rng)
    print(f"drew {len(samples)} i.i.d. samples from FD nu")

    # 2) train score
    model = train(samples, epochs=args.epochs, sigma=args.sigma, n_harm=args.harmonics)
    s_exact = fd["grad_log_nu"]; s_learn = score_np(model, pts)
    wmask = fd["nu"] / fd["nu"].sum()
    rel = np.sqrt(np.sum(wmask * np.sum((s_learn - s_exact) ** 2, 1)) /
                  np.sum(wmask * np.sum(s_exact ** 2, 1)))
    print(f"nu-weighted score rel. error = {rel:.3f}")

    # 3) control test (numpy EM with grid drift)
    s_grid = s_learn.reshape(N, N, 2)
    gv_grid = P.DV(pts).reshape(N, N, 2)
    samp_nu = simulate(s_grid, h, args.walkers, args.steps, args.dt, 0.3, rng)
    samp_rho = simulate(gv_grid + a * s_grid, h, args.walkers, args.steps, args.dt, 0.3, rng)
    u = a * interp(s_grid, samp_rho, h) + 2 * P.DV(samp_rho)
    cost = float(np.mean(P.c(samp_rho) + np.sum(u ** 2, 1) / (2 * a)))
    print(f"T4 ergodic cost = {cost:.4f}  (lambda = {fd['lam']:.4f})")

    def h2(pp):
        H, _, _ = np.histogram2d(pp[:, 0], pp[:, 1], bins=N // 2, range=[[0, L], [0, L]]); return H
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(fd["nu"].reshape(N, N).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[0, 0].set_title(r"FD $\nu$")
    ax[0, 1].imshow(h2(samp_nu).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[0, 1].set_title(r"controlled ($s_\theta$) $\to\nu$")
    ax[1, 0].imshow(fd["rho"].reshape(N, N).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[1, 0].set_title(r"FD $\rho^*$")
    ax[1, 1].imshow(h2(samp_rho).T, origin="lower", extent=[0, L, 0, L], cmap="viridis"); ax[1, 1].set_title(r"controlled ($u^*$) $\to\rho^*$")
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"CLEAN demo (FD-drawn nu samples): score rel.err {rel:.3f}\n"
                 f"T4 ergodic cost {cost:.3f} vs lambda {fd['lam']:.3f}")
    fig.savefig(os.path.join(HERE, "figures", "clean_demo.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/clean_demo.png")


if __name__ == "__main__":
    main()
