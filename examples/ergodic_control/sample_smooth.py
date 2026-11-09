"""Run INS Fleming-Viot on the smooth AoG and save (a) resampled QSD nu draws for
score matching and (b) a snapshot of the raw INS particle locations (both replicas)
for the 'what the sampler holds' figure.

    python sample_smooth.py --T 50 --N 20 --h0 0.1
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import fleming_viot

import problem_smooth as P

L = P.L
HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--burn", type=float, default=0.3)
    ap.add_argument("--nsamp", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    a = P.diffusion(P.a_coeff)
    x0 = rng.uniform(0, L, (args.N, P.dim)); y0 = rng.uniform(0, L, (args.N, P.dim))
    print(f"INS on smooth AoG: T={args.T} N={args.N} h0={args.h0} ...", flush=True)
    px, py, ti, rho = fleming_viot(args.T, P.V, P.DV, P.D2V, a, P.a_coeff, P.c, args.h0,
                                   x0.copy(), y0.copy())
    print("events:", len(ti))
    m = ti > args.burn * args.T
    px, py, ti, rho = px[m], py[m], ti[m], rho[m]
    dt = np.append(np.diff(ti), 0.0)
    N = args.N

    # snapshot of raw INS particle locations at the final recorded event (both replicas)
    snap_x = P.wrap(px[-1]); snap_y = P.wrap(py[-1])

    # resampled QSD nu: pool both replicas, weights dt*rho and dt*(1-rho)
    pool = np.concatenate([px[:, i, :] for i in range(N)] + [py[:, i, :] for i in range(N)])
    w = np.concatenate([dt * rho[:, i] for i in range(N)] + [dt * (1 - rho[:, i]) for i in range(N)])
    w = np.clip(w, 0.0, None); w /= w.sum()
    idx = rng.choice(len(w), size=min(args.nsamp, len(w)), p=w)
    nu_samples = P.wrap(pool[idx])

    out = dict(samples=nu_samples, snap_x=snap_x, snap_y=snap_y, rho_last=rho[-1],
               params=dict(T=args.T, N=args.N, h0=args.h0))
    np.save(os.path.join(HERE, "qsd_samples_smooth.npy"), out, allow_pickle=True)
    print(f"saved qsd_samples_smooth.npy  ({len(nu_samples)} nu samples, "
          f"snapshot {snap_x.shape[0]}+{snap_y.shape[0]} particles)")


if __name__ == "__main__":
    main()
