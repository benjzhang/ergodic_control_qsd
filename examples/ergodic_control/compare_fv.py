"""Compare QSD sampling: plain Fleming-Viot (no swap) vs infinite-swap (INS).

Both run the killed process (drift -grad V, killing c) with Fleming-Viot respawn;
INS additionally symmetrizes via the x<->y swap. We compare how well each recovers
the QSD nu against the FD reference: eigenvalue estimate E_nu[c] (per-particle
mean +/- std), corner-well mass, and the QSD histograms.

    python compare_fv.py [--T 100 --N 20 --h0 0.1 --burn 0.2]
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
CORNERS = [0, 3, 12, 15]   # indices of the 4 corner wells in P.centers


def well_labels(pos):
    d = pos[:, None, :] - P.centers[None]
    d = (d + L / 2) % L - L / 2
    return (d ** 2).sum(-1).argmin(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--burn", type=float, default=0.2)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    a = P.diffusion(P.a_coeff)
    fd = np.load(os.path.join(HERE, "fd_reference.npy"), allow_pickle=True).item()
    Nfd = fd["N"]; lam = fd["lam"]

    # ---------- plain Fleming-Viot (no swap) ----------
    x0 = rng.uniform(0, L, (args.N, P.dim))
    print("running plain Fleming-Viot (no swap) ...", flush=True)
    pv, tv = fleming_viot_vanilla(args.T, lambda x: -P.DV(x), a, P.c, args.h0, x0.copy())
    mv = tv > args.burn * args.T; pv, tv = pv[mv], tv[mv]
    dtv = np.append(np.diff(tv), 0.0)
    ec_v = np.array([np.sum(dtv * P.c(pv[:, i, :])) / np.sum(dtv) for i in range(args.N)])
    # time-weighted resample of forward positions -> QSD samples
    wv = np.concatenate([dtv] * args.N); posv = np.concatenate([pv[:, i, :] for i in range(args.N)])
    wv = np.clip(wv, 0, None); wv /= wv.sum()
    sv = P.wrap(posv[rng.choice(len(wv), size=60000, p=wv)])

    # ---------- infinite-swap Fleming-Viot ----------
    y0 = rng.uniform(0, L, (args.N, P.dim))
    print("running infinite-swap Fleming-Viot ...", flush=True)
    px, py, ti, rho = fleming_viot(args.T, P.V, P.DV, P.D2V, a, P.a_coeff, P.c, args.h0, x0.copy(), y0.copy())
    mi = ti > args.burn * args.T; px, py, ti, rho = px[mi], py[mi], ti[mi], rho[mi]
    dti = np.append(np.diff(ti), 0.0)
    ec_i = np.array([np.sum(dti * (P.c(px[:, i, :]) * rho[:, i] + P.c(py[:, i, :]) * (1 - rho[:, i]))) / np.sum(dti)
                     for i in range(args.N)])
    pool = np.concatenate([px[:, i, :] for i in range(args.N)] + [py[:, i, :] for i in range(args.N)])
    wi = np.concatenate([dti * rho[:, i] for i in range(args.N)] + [dti * (1 - rho[:, i]) for i in range(args.N)])
    wi = np.clip(wi, 0, None); wi /= wi.sum()
    si = P.wrap(pool[rng.choice(len(wi), size=60000, p=wi)])

    # ---------- metrics ----------
    def per_corner(s):
        lab = well_labels(P.wrap(s))
        return np.array([np.mean(lab == k) for k in CORNERS])   # mass in each of the 4 corners
    cm_fd_each = np.array([fd["nu"].ravel()[well_labels(fd["pts"]) == k].sum() * fd["h"] ** 2 for k in CORNERS])
    pv_c, pi_c = per_corner(sv), per_corner(si)
    # symmetry: coefficient of variation across the 4 (should be ~0 for the symmetric QSD)
    def cv(x): return float(np.std(x) / (np.mean(x) + 1e-12))
    print("\n=== QSD recovery vs FD (lambda = %.4f) ===" % lam)
    print("  FD nu:  per-corner mass = [%s]  total = %.3f" %
          (", ".join("%.3f" % v for v in cm_fd_each), cm_fd_each.sum()))
    print("  plain FV (no swap):  E[c] = %.3f +/- %.3f | corners [%s] tot %.3f | asym(cv) %.2f" %
          (ec_v.mean(), ec_v.std(), ", ".join("%.3f" % v for v in pv_c), pv_c.sum(), cv(pv_c)))
    print("  infinite-swap (INS): E[c] = %.3f +/- %.3f | corners [%s] tot %.3f | asym(cv) %.2f" %
          (ec_i.mean(), ec_i.std(), ", ".join("%.3f" % v for v in pi_c), pi_c.sum(), cv(pi_c)))
    np.save(os.path.join(HERE, "compare_data.npy"),
            {"sv": sv, "si": si, "ec_v": ec_v, "ec_i": ec_i,
             "pv_c": pv_c, "pi_c": pi_c, "T": args.T, "N": args.N}, allow_pickle=True)

    def corner_mass(s):
        return float(np.mean(np.isin(well_labels(P.wrap(s)), CORNERS)))

    # ---------- figure ----------
    def h2(s):
        H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=Nfd // 2, range=[[0, L], [0, L]], density=True); return H
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    ax[0].imshow(fd["nu"].reshape(Nfd, Nfd).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[0].set_title(r"FD  $\nu$   (E[c]=%.2f)" % lam)
    ax[1].imshow(h2(sv).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[1].set_title("plain FV (no swap)\nE[c]=%.2f$\\pm$%.2f  corner %.2f" % (ec_v.mean(), ec_v.std(), corner_mass(sv)))
    ax[2].imshow(h2(si).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[2].set_title("infinite-swap (INS)\nE[c]=%.2f$\\pm$%.2f  corner %.2f" % (ec_i.mean(), ec_i.std(), corner_mass(si)))
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("QSD sampling: plain Fleming-Viot vs infinite-swap  (T=%g, N=%d, h0=%g)" % (args.T, args.N, args.h0))
    fig.savefig(os.path.join(HERE, "figures", "compare_fv.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/compare_fv.png")


if __name__ == "__main__":
    main()
