"""Two-well ergodic-control comparison: plain Fleming-Viot vs INS.

Bimodal potential V with two deep, well-separated, symmetry-equivalent wells on the
torus. The true QSD is symmetric between the two wells, so the well-A/well-B mass
split measures each sampler's mixing: plain FV (respawn, no swap) is expected to
FIXATE to one well at a high barrier; INS should keep both (a pair x in A, y in B is
stable and the swap symmetrizes it).

Killing types (the shaping of the QSD):
    const      c = kappa                        (QSD = base bimodal measure)
    indicator  c = kappa (1 - sum_k Gauss_k)    (near-0 at wells, kappa elsewhere:
                                                  confines to wells + penalizes the barrier)
    barrier    c = kappa * (well-barrier bump)  (high specifically in the crossing region)

    python compare_2mode.py --a 1.0 --killing indicator --kappa 2 [--T 100 --N 20]
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import fleming_viot, fleming_viot_vanilla

L = 4.0
dim = 2
sigma = 0.3
centers = np.array([[1.0, 2.0], [3.0, 2.0]])   # two wells, symmetric under x1 -> 4-x1
HERE = os.path.dirname(__file__)


def wrap(x):
    return np.mod(x, L)


def _delta(x):
    d = wrap(x)[..., None, :] - centers
    return (d + L / 2) % L - L / 2


def V(x):
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)
    return -logsumexp(q, axis=-1)


def DV(x):
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(q - logsumexp(q, axis=-1, keepdims=True))
    return np.sum(w[..., None] * d / sigma ** 2, axis=-2)


def D2V(x):
    d = _delta(x)
    q = -np.sum(d ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(q - logsumexp(q, axis=-1, keepdims=True))
    r2 = np.sum(d ** 2, axis=-1)
    return dim / sigma ** 2 - np.sum(w * r2 / sigma ** 4, axis=-1) + np.sum(DV(x) ** 2, axis=-1)


def make_c(kind, kappa):
    def c(x):
        d = _delta(x)
        r2 = np.sum(d ** 2, axis=-1)                       # dist^2 to each well
        near = np.exp(-r2 / (2 * (1.3 * sigma) ** 2)).sum(-1)   # ~1 at a well, ~0 away
        if kind == "const":
            return kappa * np.ones_like(near)
        if kind == "indicator":
            return kappa * np.clip(1 - near, 0, None)     # low at wells, kappa elsewhere
        if kind == "barrier":
            xw = wrap(x)[..., 0]
            bump = np.exp(-((((xw - 2 + L / 2) % L - L / 2)) ** 2) / (2 * 0.4 ** 2))   # peak at x1=2 barrier
            bump += np.exp(-(((xw + L / 2) % L - L / 2) ** 2) / (2 * 0.4 ** 2))        # peak at x1=0/4 barrier
            return kappa * bump + 0.2 * kappa
        if kind == "confine":
            # LOW inside the well-strip (incl. the x1=2 barrier so INS can ratchet through),
            # grows outside -> confines without penalizing the transition region.
            xw = wrap(x)
            dx1 = (xw[..., 0] - 2 + L / 2) % L - L / 2
            dx2 = (xw[..., 1] - 2 + L / 2) % L - L / 2
            return kappa * (np.clip(np.abs(dx1) - 1.1, 0, None) ** 2 + dx2 ** 2)
        raise ValueError(kind)
    return c


def well_split(s):
    d = _delta(s); lab = np.sum(d ** 2, -1).argmin(1)
    return np.array([np.mean(lab == 0), np.mean(lab == 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--killing", default="indicator")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--burn", type=float, default=0.25)
    ap.add_argument("--init", default="uniform", help="uniform | wellA (all start in one well)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    a = lambda x: args.a * np.ones_like(x)
    c = make_c(args.killing, args.kappa)
    tag = "%s_a%.2g_k%.2g_%s" % (args.killing, args.a, args.kappa, args.init)

    if args.init == "wellA":
        x0 = centers[0] + 0.1 * rng.standard_normal((args.N, dim))   # all start in well A
    else:
        x0 = rng.uniform(0, L, (args.N, dim))
    print(f"[{tag}] plain Fleming-Viot ...", flush=True)
    pv, tv = fleming_viot_vanilla(args.T, lambda x: -DV(x), a, c, args.h0, x0.copy())
    mv = tv > args.burn * args.T; pv, tv = pv[mv], tv[mv]
    dtv = np.append(np.diff(tv), 0.0)
    wv = np.clip(np.concatenate([dtv] * args.N), 0, None); wv /= wv.sum()
    sv = wrap(np.concatenate([pv[:, i, :] for i in range(args.N)])[rng.choice(len(wv), 60000, p=wv)])

    y0 = x0.copy() if args.init == "wellA" else rng.uniform(0, L, (args.N, dim))
    print(f"[{tag}] INS ...", flush=True)
    px, py, ti, rho = fleming_viot(args.T, V, DV, D2V, a, args.a, c, args.h0, x0.copy(), y0.copy())
    mi = ti > args.burn * args.T; px, py, ti, rho = px[mi], py[mi], ti[mi], rho[mi]
    dti = np.append(np.diff(ti), 0.0)
    pool = np.concatenate([px[:, i, :] for i in range(args.N)] + [py[:, i, :] for i in range(args.N)])
    wi = np.concatenate([dti * rho[:, i] for i in range(args.N)] + [dti * (1 - rho[:, i]) for i in range(args.N)])
    wi = np.clip(wi, 0, None); wi /= wi.sum()
    si = wrap(pool[rng.choice(len(wi), 60000, p=wi)])

    fv_s, ins_s = well_split(sv), well_split(si)
    def imbalance(sp): return float(abs(sp[0] - sp[1]))
    print(f"\n=== 2-well ({tag}), T={args.T} N={args.N}  (symmetric QSD -> 0.50/0.50) ===")
    print("  plain FV : well split %.2f / %.2f   imbalance %.2f" % (fv_s[0], fv_s[1], imbalance(fv_s)))
    print("  INS      : well split %.2f / %.2f   imbalance %.2f" % (ins_s[0], ins_s[1], imbalance(ins_s)))

    def h2(s):
        H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=100, range=[[0, L], [0, L]], density=True); return H
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].imshow(h2(sv).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[0].set_title("plain FV (no swap)\nwell split %.2f/%.2f" % (fv_s[0], fv_s[1]))
    ax[1].imshow(h2(si).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[1].set_title("infinite-swap (INS)\nwell split %.2f/%.2f" % (ins_s[0], ins_s[1]))
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Two-well QSD (killing=%s, a=%g): does the sampler keep BOTH wells?  (T=%g, N=%d)"
                 % (args.killing, args.a, args.T, args.N))
    fig.savefig(os.path.join(HERE, "figures", "compare_2mode_%s.png" % tag), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/compare_2mode_%s.png" % tag)


if __name__ == "__main__":
    main()
