"""Cold-regime comparison: plain Fleming-Viot vs INS.

Lowering the diffusion a raises the barrier/noise ratio, so the base measure
e^{-2V/a} sharpens and crossing between the four (symmetry-equivalent) corner wells
gets hard. The true QSD stays 4-fold symmetric by construction, so the per-corner
coefficient-of-variation (CV) measures how badly each sampler breaks that symmetry:
plain FV is expected to get trapped (high CV), INS to keep mixing (low CV).

    python compare_cold.py --a 0.7 [--T 100 --N 20 --h0 0.1 --burn 0.25]
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
from utils import fleming_viot, fleming_viot_vanilla

L = P.L
HERE = os.path.dirname(__file__)
CORNERS = [0, 3, 12, 15]


def well_labels(pos):
    d = pos[:, None, :] - P.centers[None]
    d = (d + L / 2) % L - L / 2
    return (d ** 2).sum(-1).argmin(1)


def per_corner(s):
    lab = well_labels(P.wrap(s))
    return np.array([np.mean(lab == k) for k in CORNERS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--burn", type=float, default=0.25)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    a = P.diffusion(args.a)

    x0 = rng.uniform(0, L, (args.N, P.dim))
    print(f"a={args.a}: running plain Fleming-Viot ...", flush=True)
    pv, tv = fleming_viot_vanilla(args.T, lambda x: -P.DV(x), a, P.c, args.h0, x0.copy())
    mv = tv > args.burn * args.T; pv, tv = pv[mv], tv[mv]
    dtv = np.append(np.diff(tv), 0.0)
    wv = np.clip(np.concatenate([dtv] * args.N), 0, None); wv /= wv.sum()
    sv = P.wrap(np.concatenate([pv[:, i, :] for i in range(args.N)])[rng.choice(len(wv), 60000, p=wv)])

    y0 = rng.uniform(0, L, (args.N, P.dim))
    print(f"a={args.a}: running INS ...", flush=True)
    px, py, ti, rho = fleming_viot(args.T, P.V, P.DV, P.D2V, a, args.a, P.c, args.h0, x0.copy(), y0.copy())
    mi = ti > args.burn * args.T; px, py, ti, rho = px[mi], py[mi], ti[mi], rho[mi]
    dti = np.append(np.diff(ti), 0.0)
    pool = np.concatenate([px[:, i, :] for i in range(args.N)] + [py[:, i, :] for i in range(args.N)])
    wi = np.concatenate([dti * rho[:, i] for i in range(args.N)] + [dti * (1 - rho[:, i]) for i in range(args.N)])
    wi = np.clip(wi, 0, None); wi /= wi.sum()
    si = P.wrap(pool[rng.choice(len(wi), 60000, p=wi)])

    def cv(x): return float(np.std(x) / (np.mean(x) + 1e-12))
    pv_c, pi_c = per_corner(sv), per_corner(si)
    print(f"\n=== a={args.a}, T={args.T}, N={args.N}  (true QSD: 4 corners equal, CV=0) ===")
    print("  plain FV : per-corner [%s]  CV = %.2f" % (", ".join("%.3f" % v for v in pv_c), cv(pv_c)))
    print("  INS      : per-corner [%s]  CV = %.2f" % (", ".join("%.3f" % v for v in pi_c), cv(pi_c)))
    np.save(os.path.join(HERE, "cold_data_a%.2g.npy" % args.a),
            {"sv": sv, "si": si, "pv_c": pv_c, "pi_c": pi_c, "a": args.a}, allow_pickle=True)

    def h2(s):
        H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=100, range=[[0, L], [0, L]], density=True); return H
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].imshow(h2(sv).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[0].set_title("plain FV (no swap)\nper-corner CV = %.2f  (trapped)" % cv(pv_c))
    ax[1].imshow(h2(si).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[1].set_title("infinite-swap (INS)\nper-corner CV = %.2f  (mixes)" % cv(pi_c))
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Cold regime (a=%g): does the sampler keep the 4-corner symmetry?  (T=%g, N=%d)"
                 % (args.a, args.T, args.N))
    fig.savefig(os.path.join(HERE, "figures", "compare_cold_a%.2g.png" % args.a), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/compare_cold_a%.2g.png" % args.a)


if __name__ == "__main__":
    main()
