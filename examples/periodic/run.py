"""Headless runner for the periodic INS Fleming-Viot example.

Runs N-particle vanilla (independent forward SDE) and infinite-swap Fleming-Viot,
saves data + figures, and validates the weighted empirical measure against the
analytic target exp(-2V/eps). Parameterized via CLI:

    python run.py --epsilon 0.2 --N 5 [--T 100 --h0 0.1 --reps 2]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.experiment import (run_vanilla_independent, run_fv, wrap,
                              fv_weighted_samples_1d, weighted_moment)
from utils.plotting import plot_periodic_trajectories, plot_periodic_trajectories_list

import problem as P


def one_replicate(T, epsilon, N, h0, rng):
    van_t, van_x = run_vanilla_independent(P, T, epsilon, N, h0, rng)
    px, py, ti, rho = run_fv(P, T, epsilon, N, h0)
    return dict(van_t=van_t, van_x=van_x, px=px, py=py, ti=ti, rho=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--burn", type=float, default=0.1, help="burn-in fraction of T")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"{P.name}_T{args.T:g}_eps{args.epsilon:g}_N{args.N}_h{args.h0:g}"
    out = args.out or os.path.join(os.path.dirname(__file__), "outputs", tag)
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)
    rng = np.random.default_rng()

    reps = [one_replicate(args.T, args.epsilon, args.N, args.h0, rng) for _ in range(args.reps)]
    print(f"{tag}: FV events/rep = {[len(r['ti']) for r in reps]}")

    # pooled weighted (FV) and vanilla samples across reps, with burn-in
    fv_pos, fv_wt, van_pos = [], [], []
    for r in reps:
        p, w = fv_weighted_samples_1d(r["px"], r["py"], r["rho"], r["ti"], P.wrap_period, args.burn)
        fv_pos.append(p)
        fv_wt.append(w)
        for i in range(len(r["van_t"])):
            t = np.asarray(r["van_t"][i])
            m = t >= args.burn * t[-1]
            van_pos.append(wrap(np.asarray(r["van_x"][i])[m, 0], P.wrap_period))
    fv_pos = np.concatenate(fv_pos)
    fv_wt = np.concatenate(fv_wt)
    van_pos = np.concatenate(van_pos)

    # ---- moment validation vs analytic target ----
    funcs = [("E[cos2pix]", lambda z: np.cos(2 * np.pi * z)), ("E[x^2]", lambda z: z ** 2)]
    print("  functional     FV-weighted    vanilla     analytic")
    for nm, g in funcs:
        print("  %-12s %10.4f %10.4f %10.4f"
              % (nm, weighted_moment(fv_pos, fv_wt, g), np.mean(g(van_pos)), P.analytic_moment(g, args.epsilon)))

    # ---- figures ----
    xg, wg = P.analytic_density(args.epsilon)
    r0 = reps[0]

    def _save_trace(fig, name, title):
        fig.suptitle(title)
        fig.savefig(os.path.join(figs, name), dpi=140, bbox_inches="tight")
        plt.close(fig)

    # 1) trace plots (replicate 0): ALL FV particles (forward + backward together), and vanilla Langevin
    fv_all = np.concatenate([r0["px"][:, :, 0], r0["py"][:, :, 0]], axis=1)  # (n_events, 2N)
    _save_trace(plot_periodic_trajectories(r0["ti"], fv_all, threshold=1.5),
                "fv_all_particles.png", f"FV all particles (forward + backward)  ({tag})")
    _save_trace(plot_periodic_trajectories_list(r0["van_t"], [np.asarray(x)[:, 0] for x in r0["van_x"]], threshold=1.5),
                "vanilla_trajectories.png", f"Vanilla (Langevin) trajectories  ({tag})")

    # 2) implied potential of the INS system only (single panel, NO colorbar),
    #    with the coupled (x,y) locations of ALL particle pairs overlaid in red.
    def Psi(x):
        return 2.0 * P.V(x)  # forward potential Psi = 2V  (Phi = 0)

    def implied(x, y, eps):
        return -eps * np.log(np.exp(-Psi(x) / eps) + np.exp(-Psi(y) / eps))

    gx = np.linspace(P.domain[0], P.domain[1], 200)
    XX, YY = np.meshgrid(gx, gx)
    Z_impl = implied(XX, YY, args.epsilon)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contourf(XX, YY, Z_impl, levels=40, cmap="viridis")
    for i in range(args.N):
        ax.scatter(r0["px"][:, i, 0], r0["py"][:, i, 0], s=1, alpha=0.35, color="red")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"Implied potential ($\varepsilon=%g$)  (%s)" % (args.epsilon, tag))
    fig.savefig(os.path.join(figs, "implied_potential.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3) empirical measure vs analytic target (separate panels)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ax[0].hist(fv_pos, bins=120, range=P.domain, weights=fv_wt, density=True,
               alpha=0.7, color="C1", label="FV weighted")
    ax[1].hist(van_pos, bins=120, range=P.domain, density=True,
               alpha=0.7, color="C0", label="vanilla")
    for a_, lbl in zip(ax, ["FV weighted", "vanilla"]):
        a_.plot(xg, wg, "r", lw=2, label=r"$\propto e^{-2V/\varepsilon}$")
        a_.set_xlabel("x")
        a_.legend(loc="upper center", fontsize="small")
    ax[0].set_ylabel("density")
    fig.suptitle(f"Empirical measure vs target  ({tag})")
    fig.savefig(os.path.join(figs, "density_comparison.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    def _obj(lst):
        arr = np.empty(len(lst), dtype=object)
        for i, v in enumerate(lst):
            arr[i] = np.asarray(v)
        return arr
    np.save(os.path.join(out, "data.npy"),
            {"params": vars(args), "fv_events": [len(r["ti"]) for r in reps],
             "px": r0["px"], "py": r0["py"], "ti": r0["ti"], "rho": r0["rho"],
             "van_t": _obj(r0["van_t"]), "van_x": _obj(r0["van_x"])}, allow_pickle=True)
    print("saved to", out)


if __name__ == "__main__":
    main()
