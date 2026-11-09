"""Stage 2: sample the QSD nu via forward-backward INS Fleming-Viot, then validate
against the FD reference.

Runs the coupled particle system with killing c(x), resamples the FULL symmetrized
weighted measure (x with weight dt*rho, y with weight dt*(1-rho)) -> approximate
i.i.d. draws from the QSD nu prop. phi m.  Saves the samples for score matching
(Stage 3) and a comparison figure + the eigenvalue estimate E_nu[c] vs FD lambda.

    python sample_qsd.py --T 50 --N 20 [--h0 0.1 --burn 0.2 --nsamp 60000]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import fleming_viot

import problem as P

L = P.L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--burn", type=float, default=0.2, help="burn-in fraction of T")
    ap.add_argument("--nsamp", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    here = os.path.dirname(__file__)
    rng = np.random.default_rng(args.seed)

    a = P.diffusion(P.a_coeff)
    x0 = rng.uniform(0, L, size=(args.N, P.dim))            # spread init -> faster equilibration
    y0 = rng.uniform(0, L, size=(args.N, P.dim))
    print(f"running INS Fleming-Viot: T={args.T} N={args.N} h0={args.h0} ...")
    px, py, ti, rho = fleming_viot(args.T, P.V, P.DV, P.D2V, a, P.a_coeff, P.c,
                                   args.h0, x0.copy(), y0.copy())
    print("events:", len(ti))

    # ---- burn-in ----
    m = ti > args.burn * args.T
    px, py, ti, rho = px[m], py[m], ti[m], rho[m]
    dt = np.append(np.diff(ti), 0.0)
    N = args.N

    # ---- eigenvalue estimate E_nu[c] (full weighted measure) vs FD ----
    num = np.sum([dt * (P.c(px[:, i, :]) * rho[:, i] + P.c(py[:, i, :]) * (1 - rho[:, i]))
                  for i in range(N)])
    den = N * np.sum(dt)
    lam_est = num / den

    # ---- resample the QSD nu: pool both replicas, weights dt*rho and dt*(1-rho) ----
    pool = np.concatenate([px[:, i, :] for i in range(N)] + [py[:, i, :] for i in range(N)])
    w = np.concatenate([dt * rho[:, i] for i in range(N)] + [dt * (1 - rho[:, i]) for i in range(N)])
    w = np.clip(w, 0.0, None); w /= w.sum()
    idx = rng.choice(len(w), size=min(args.nsamp, len(w)), p=w)
    nu_samples = P.wrap(pool[idx])

    np.save(os.path.join(here, "qsd_samples.npy"),
            {"samples": nu_samples, "lam_est": lam_est,
             "params": {"T": args.T, "N": args.N, "h0": args.h0, "burn": args.burn}},
            allow_pickle=True)

    # ---- validation vs FD ----
    fd = np.load(os.path.join(here, "fd_reference.npy"), allow_pickle=True).item()
    Nfd = fd["N"]
    print(f"  eigenvalue  E_nu[c]:  FV/INS = {lam_est:.4f}   FD = {fd['lam']:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    ax[0].imshow(fd["nu"].reshape(Nfd, Nfd).T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
    ax[0].set_title(r"FD QSD $\nu$")
    ax[1].hist2d(nu_samples[:, 0], nu_samples[:, 1], bins=120, range=[[0, L], [0, L]], cmap="viridis")
    ax[1].set_title("FV/INS resampled QSD (both replicas)")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"QSD: FV/INS vs FD   (E[c]: {lam_est:.3f} vs {fd['lam']:.3f})")
    os.makedirs(os.path.join(here, "figures"), exist_ok=True)
    fig.savefig(os.path.join(here, "figures", "qsd_samples_vs_fd.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved qsd_samples.npy + figures/qsd_samples_vs_fd.png  (%d samples)" % len(nu_samples))


if __name__ == "__main__":
    main()
