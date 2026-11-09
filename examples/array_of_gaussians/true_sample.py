"""Reference TRUE sample for the array-of-Gaussians target pi(x) ∝ exp(-V(x)/eps).

The 16 wells are well separated (sigma=0.1, spacing 1), so near each center
    exp(-V/eps) ≈ G_k^{1/eps} = Normal(center_k, (sigma*sqrt(eps))^2),
and by symmetry the wells carry equal mass. So we sample DIRECTLY from the
Gaussians: pick a center uniformly, draw N(center, sigma^2*eps), wrap to [0,4).

Saves the samples (true_samples.npy) + a standalone figure + a side-by-side vs
the FV resampled samples (if aog_data.npy is present), in the greyscale/red style.

    python true_sample.py [--eps 0.6 --nsamp 40000 --outdir outputs/<tag>]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import problem as P


def true_samples(eps, nsamp, seed=0):
    """Direct i.i.d. draws: pick a well uniformly, draw N(center, sigma^2 * eps), wrap."""
    rng = np.random.default_rng(seed)
    std = P.sigma * np.sqrt(eps)                 # width of G_k^{1/eps}
    idx = rng.integers(len(P.centers), size=nsamp)
    samp = P.centers[idx] + std * rng.standard_normal((nsamp, 2))
    return P.wrap(samp)


def _fv_resampled(outdir, nsamp, seed=1):
    """Rebuild FV resampled draws from saved aog_data.npy (same logic as run.py)."""
    path = os.path.join(outdir, "aog_data.npy")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True).item()
    apx, apy, at, arho = d["allpositionsx"], d["allpositionsy"], np.asarray(d["alltime"]), d["all_rho"]
    N = apx.shape[1]
    dt = np.append(np.diff(at), 0.0)
    pool = np.concatenate([apx[:, i, :] for i in range(N)] + [apy[:, i, :] for i in range(N)])
    w = np.concatenate([dt * arho[:, i] for i in range(N)] + [dt * (1 - arho[:, i]) for i in range(N)])
    w = np.clip(w, 0.0, None); w /= w.sum()
    rng = np.random.default_rng(seed)
    return pool[rng.choice(len(w), size=min(len(w), nsamp), p=w)]


def _backdrop(ax, Xg, Yg, Z):
    cf = ax.contourf(Xg, Yg, Z, levels=30, cmap="gray", vmin=0, vmax=10, alpha=0.55)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(P.domain); ax.set_ylim(P.domain)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--nsamp", type=int, default=40000)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                                                     "outputs", "aog_T100_eps0.6_N5_h0.1"))
    args = ap.parse_args()
    figs = os.path.join(args.outdir, "figures")
    os.makedirs(figs, exist_ok=True)

    samp = true_samples(args.eps, args.nsamp)
    near = samp[np.hypot(samp[:, 0] - 2.0, samp[:, 1] - 2.0) < 0.4]
    print("well(2,2): %d samples, empirical std = %.4f  (analytic sigma*sqrt(eps) = %.4f)"
          % (len(near), near.std(axis=0).mean(), P.sigma * np.sqrt(args.eps)))

    gx = np.linspace(P.domain[0], P.domain[1], 200)
    Xg, Yg = np.meshgrid(gx, gx)
    Z = P.V(np.stack([Xg, Yg], axis=-1))

    # standalone true-sample figure
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = _backdrop(ax, Xg, Yg, Z)
    fig.colorbar(cf, ax=ax, label="V(x)")
    ax.scatter(samp[:, 0], samp[:, 1], s=1, alpha=0.4, color="red")
    ax.set_title(r"True target samples  $\pi \propto e^{-V/\varepsilon}$  ($\varepsilon=%g$)" % args.eps)
    fig.savefig(os.path.join(figs, "aog_true_samples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved aog_true_samples.png")

    # side-by-side vs FV resampled
    fv = _fv_resampled(args.outdir, args.nsamp)
    if fv is not None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for a_, pts_, ttl in [(ax[0], samp, "True target  " + r"$e^{-V/\varepsilon}$"),
                              (ax[1], fv, "FV resampled")]:
            _backdrop(a_, Xg, Yg, Z)
            a_.scatter(pts_[:, 0], pts_[:, 1], s=1, alpha=0.4, color="red")
            a_.set_title(ttl)
        fig.suptitle(r"AoG reference vs Fleming-Viot  ($\varepsilon=%g$)" % args.eps)
        fig.savefig(os.path.join(figs, "aog_true_vs_fv.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved aog_true_vs_fv.png")

    np.save(os.path.join(args.outdir, "true_samples.npy"),
            {"samples": samp, "eps": args.eps, "std": P.sigma * np.sqrt(args.eps),
             "nsamp": args.nsamp}, allow_pickle=True)
    print("saved true_samples.npy")


if __name__ == "__main__":
    main()
