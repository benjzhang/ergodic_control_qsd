"""One INS or plain-FV run on the SMOOTH AoG -> eigenvalue estimate E_nu[c].

Writes a small json per (sampler, seed) so a SLURM array can fan out seeds; the
aggregator (aggregate_smooth.py) turns the replicate E[c]'s into bias / variance /
MSE for INS vs plain FV, against the FD lambda.

    python run_smooth.py --sampler ins --seed 0 --h0 0.1 --T 50 --N 20
    python run_smooth.py --sampler fv  --seed 0 --h0 0.1 --T 50 --N 20
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import fleming_viot, fleming_viot_vanilla

import problem_smooth as P

L = P.L
HERE = os.path.dirname(__file__)


def ins_estimate(T, N, h0, burn, seed):
    rng = np.random.default_rng(seed)
    a = P.diffusion(P.a_coeff)
    x0 = rng.uniform(0, L, (N, P.dim)); y0 = rng.uniform(0, L, (N, P.dim))
    px, py, ti, rho = fleming_viot(T, P.V, P.DV, P.D2V, a, P.a_coeff, P.c, h0, x0.copy(), y0.copy())
    m = ti > burn * T
    px, py, ti, rho = px[m], py[m], ti[m], rho[m]
    dt = np.append(np.diff(ti), 0.0)
    num = np.sum([dt * (P.c(px[:, i, :]) * rho[:, i] + P.c(py[:, i, :]) * (1 - rho[:, i]))
                  for i in range(N)])
    den = N * np.sum(dt)
    return num / den, len(ti)


def fv_estimate(T, N, h0, burn, seed):
    rng = np.random.default_rng(seed)
    a = P.diffusion(P.a_coeff)
    x0 = rng.uniform(0, L, (N, P.dim))
    pv, tv = fleming_viot_vanilla(T, lambda x: -P.DV(x), a, P.c, h0, x0.copy())
    m = tv > burn * T
    pv, tv = pv[m], tv[m]
    dt = np.append(np.diff(tv), 0.0)
    num = np.sum([dt * P.c(pv[:, i, :]) for i in range(N)])
    den = N * np.sum(dt)
    return num / den, len(tv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["ins", "fv"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--burn", type=float, default=0.3)
    args = ap.parse_args()

    if args.sampler == "ins":
        lam, nev = ins_estimate(args.T, args.N, args.h0, args.burn, args.seed)
    else:
        lam, nev = fv_estimate(args.T, args.N, args.h0, args.burn, args.seed)

    out = dict(sampler=args.sampler, seed=args.seed, h0=args.h0, T=args.T, N=args.N,
               burn=args.burn, lam_est=float(lam), events=int(nev))
    d = os.path.join(HERE, "smooth_out"); os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f"{args.sampler}_h{args.h0}_s{args.seed}.json")
    with open(fn, "w") as f:
        json.dump(out, f)
    print(f"[{args.sampler} seed={args.seed} h0={args.h0}] E[c]={lam:.4f}  ({nev} events) -> {fn}")


if __name__ == "__main__":
    main()
