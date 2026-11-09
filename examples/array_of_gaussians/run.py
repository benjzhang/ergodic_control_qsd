"""Headless runner for the array-of-Gaussians (2D, 5x5 wells on a torus) example.

Runs N-particle vanilla (independent forward SDE) reference trajectories and
infinite-swap Fleming-Viot, then saves the potential contour plus the vanilla
and Fleming-Viot (forward and rho-weighted) scatter figures, and the raw data.
Parameterized via CLI:

    python run.py --epsilon 0.4 --N 5 [--T 100 --h0 0.1]

Behavior mirrors the original run_aog.py exactly: the simulators are called
without wrap_period (unbounded lattice), the potential V/DV/D2V wrap internally
so the force is periodic, and trajectories are wrapped to [0, 4) for plotting
via problem.wrap (the AOG convention, NOT the symmetric utils.experiment.wrap).
Note fleming_viot receives `epsilon` while the diffusion coefficient is 2*epsilon.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import pure_jump_approx_diffusion, fleming_viot

import problem as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--epsilon", type=float, default=0.4)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"{P.name}_T{args.T:g}_eps{args.epsilon:g}_N{args.N}_h{args.h0:g}"
    out = args.out or os.path.join(os.path.dirname(__file__), "outputs", tag)
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)

    a = P.diffusion(args.epsilon)

    # ---- grid for contour ----
    gx = np.linspace(P.domain[0], P.domain[1], 200)
    X, Y = np.meshgrid(gx, gx)
    grid = np.stack([X, Y], axis=-1)
    Z = P.V(grid)

    def base_contour(title):
        plt.figure(figsize=(7, 6))
        # faint greyscale potential backdrop (light flats, grey wells); red points carry it
        cf = plt.contourf(X, Y, Z, levels=30, cmap="gray", vmin=0, vmax=10, alpha=0.55)
        plt.colorbar(cf, label="V(x)")
        plt.title(title)
        plt.xlabel("x"); plt.ylabel("y")

    # 1) potential contour
    base_contour("Array of Gaussians: potential V(x)")
    plt.savefig(os.path.join(figs, "potential_contour.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved potential_contour.png")

    # ---- vanilla reference ----
    init_x, init_y = P.initial_ensemble(args.N)
    simple_t, simple_x = [], []
    for i in range(2 * args.N):
        tx, tt = pure_jump_approx_diffusion(args.T, lambda x: -P.DV(x), a, args.h0, init_x[0])
        simple_t.append(tt)
        simple_x.append(P.wrap(np.asarray(tx)))
    print("vanilla reference done: %d trajectories" % len(simple_x))

    # 2) vanilla trajectories
    base_contour("Vanilla reference trajectories")
    for traj in simple_x:
        plt.scatter(traj[:, 0], traj[:, 1], s=0.5, alpha=0.5)
    plt.savefig(os.path.join(figs, "vanilla_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved vanilla_trajectories.png")

    # ---- Fleming-Viot (epsilon passed as epsilon; diffusion a(x) = 2*epsilon) ----
    apx, apy, at, arho = fleming_viot(args.T, P.V, P.DV, P.D2V, a, args.epsilon, P.c,
                                      args.h0, init_x, init_y)
    apx = P.wrap(apx)
    apy = P.wrap(apy)
    print("Fleming-Viot done: %d events" % len(at))

    # 3) Fleming-Viot forward (x) positions
    base_contour("Fleming-Viot forward (x) positions")
    plt.scatter(apx[:, :, 0].ravel(), apx[:, :, 1].ravel(), s=0.5, alpha=0.4, color="red")
    plt.savefig(os.path.join(figs, "fv_forward_positions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved fv_forward_positions.png")

    # 4) weighted empirical scatter: red points with ALPHA = weight (rho for the
    #    forward x, 1-rho for the backward y), so high-weight points are solid red
    #    and low-weight points fade out / disappear.
    def _red_alpha(weight):
        w = np.clip(np.asarray(weight), 0.0, 1.0)
        rgba = np.zeros((w.size, 4))
        rgba[:, 0] = 1.0          # red
        rgba[:, 3] = w            # alpha = weight
        return rgba

    base_contour("Fleming-Viot weighted empirical measure")
    plt.scatter(apx[:, :, 0].ravel(), apx[:, :, 1].ravel(), s=1.5, color=_red_alpha(arho.ravel()))
    plt.scatter(apy[:, :, 0].ravel(), apy[:, :, 1].ravel(), s=1.5, color=_red_alpha((1 - arho).ravel()))
    plt.savefig(os.path.join(figs, "fv_weighted_positions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved fv_weighted_positions.png")

    # 5) resampled points: DRAW (with replacement) from the weighted empirical measure
    #    (mass dt*rho at x, dt*(1-rho) at y), giving equal-weight approximate draws from
    #    the target. The reweighting defines the measure; THIS step resamples it.
    rng = np.random.default_rng()
    dt = np.append(np.diff(at), 0.0)
    pool_pos = np.concatenate([apx[:, i, :] for i in range(args.N)]
                              + [apy[:, i, :] for i in range(args.N)])
    pool_wt = np.concatenate([dt * arho[:, i] for i in range(args.N)]
                             + [dt * (1.0 - arho[:, i]) for i in range(args.N)])
    pool_wt = np.clip(pool_wt, 0.0, None)
    pool_wt = pool_wt / pool_wt.sum()
    samp = pool_pos[rng.choice(len(pool_wt), size=min(len(pool_wt), 40000), p=pool_wt)]
    base_contour("Fleming-Viot resampled from weighted measure")
    plt.scatter(samp[:, 0], samp[:, 1], s=1, alpha=0.4, color="red")
    plt.savefig(os.path.join(figs, "fv_resampled_samples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved fv_resampled_samples.png")

    # ---- save data ----
    def _obj(lst):
        arr = np.empty(len(lst), dtype=object)
        for i, v in enumerate(lst):
            arr[i] = np.asarray(v)
        return arr

    params = {"T": args.T, "epsilon": args.epsilon, "h0": args.h0, "Nparticles": args.N,
              "sigma": P.sigma, "dim": P.dim, "left": P.domain[0], "right": P.domain[1]}
    np.save(os.path.join(out, "aog_data.npy"), {
        "allpositionsx": apx, "allpositionsy": apy, "alltime": at, "all_rho": arho,
        "simple_t": _obj(simple_t), "simple_x": _obj(simple_x),
        "parameters": params,
    }, allow_pickle=True)
    with open(os.path.join(out, "parameters.json"), "w") as f:
        json.dump(params, f)
    print("saved data + figures to", out)


if __name__ == "__main__":
    main()
