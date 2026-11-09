"""Aggregate smooth-AoG replicate runs -> bias / variance / MSE, INS vs plain FV."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)


def load(sampler, h0):
    rows = []
    for fn in sorted(glob.glob(os.path.join(HERE, "smooth_out", f"{sampler}_h{h0}_s*.json"))):
        with open(fn) as f:
            rows.append(json.load(f))
    return rows


def main():
    fd = np.load(os.path.join(HERE, "fd_smooth.npy"), allow_pickle=True).item()
    lam = fd["lam"]
    print(f"FD lambda (smooth AoG, sigma={fd['sigma']}) = {lam:.4f}\n")
    print(f"{'sampler':8s} {'h0':>6s} {'n':>3s} {'mean E[c]':>10s} {'bias':>9s} "
          f"{'std':>8s} {'MSE':>10s}")
    for h0 in [0.1, 0.067, 0.05]:
        for s in ["fv", "ins"]:
            rows = load(s, h0)
            if not rows:
                continue
            e = np.array([r["lam_est"] for r in rows])
            bias = e.mean() - lam
            std = e.std(ddof=1) if len(e) > 1 else float("nan")
            mse = bias ** 2 + (e.var(ddof=1) if len(e) > 1 else 0.0)
            name = "plain FV" if s == "fv" else "INS"
            print(f"{name:8s} {h0:6.3f} {len(e):3d} {e.mean():10.4f} {bias:+9.4f} "
                  f"{std:8.4f} {mse:10.2e}")
    print("\n(smaller |bias|, |std|, MSE all better; true = FD lambda above)")


if __name__ == "__main__":
    main()
