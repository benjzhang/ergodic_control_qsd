"""Results figure for the smooth-AoG example: INS vs plain FV eigenvalue estimate.

Left  : mean E[c] +/- SE (6 seeds) with the exact FD lambda as a dashed line.
Right : MSE (bias^2 + var) per sampler, log scale, annotated.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
LAM = np.load(os.path.join(HERE, "fd_smooth.npy"), allow_pickle=True).item()["lam"]


def stats(sampler, h0=0.1):
    e = np.array([json.load(open(fn))["lam_est"]
                  for fn in sorted(glob.glob(os.path.join(HERE, "smooth_out", f"{sampler}_h{h0}_s*.json")))])
    n = len(e)
    bias = e.mean() - LAM
    var = e.var(ddof=1)
    return dict(mean=e.mean(), se=e.std(ddof=1) / np.sqrt(n), mse=bias ** 2 + var, n=n)

fv, ins = stats("fv"), stats("ins")
labels = ["plain FV\n(no swap)", "INS\n(infinite swap)"]
cols = ["0.55", "0.15"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))

# --- left: mean E[c] +/- SE vs exact ---
means = [fv["mean"], ins["mean"]]; ses = [fv["se"], ins["se"]]
ax[0].bar([0, 1], means, 0.55, yerr=ses, capsize=6, color=cols, edgecolor="k")
ax[0].axhline(LAM, color="red", ls="--", lw=1.4, label=f"exact FD $\\lambda$ = {LAM:.3f}")
for i, (m, s) in enumerate(zip(means, ses)):
    ax[0].annotate(f"{m:.3f}\n(+{100*(m-LAM)/LAM:.1f}%)", (i, m + s), ha="center", va="bottom", fontsize=9)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(labels)
ax[0].set_ylabel(r"eigenvalue estimate  $E_\nu[c]$")
ax[0].set_ylim(LAM - 0.05, max(means) + 0.18)
ax[0].set_title("Estimate vs exact (closer = better)")
ax[0].legend(fontsize=9, loc="upper left")

# --- right: MSE (log) ---
mses = [fv["mse"], ins["mse"]]
b = ax[1].bar([0, 1], mses, 0.55, color=cols, edgecolor="k")
ax[1].set_yscale("log")
for i, m in enumerate(mses):
    ax[1].annotate(f"{m:.1e}", (i, m), ha="center", va="bottom", fontsize=9)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("MSE  (bias$^2$ + variance)")
ax[1].set_title(f"Mean-squared error  (INS {mses[0]/mses[1]:.1f}$\\times$ lower)")

fig.suptitle(f"Array-of-Gaussians QSD: INS beats plain Fleming-Viot   "
             f"(h0=0.1, T=50, N=20, {fv['n']} seeds)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "figures", "results_smooth.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out, f"| MSE ratio FV/INS = {mses[0]/mses[1]:.2f}")
