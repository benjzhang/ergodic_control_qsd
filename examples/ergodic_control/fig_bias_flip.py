"""Money figure: INS's eigenvalue bias flips from worst (sharp AoG) to best (smooth
AoG) once |D2V| is brought down to the eigenfunction level.  All points at h0=0.1.

  sharp  AoG (sigma=0.2, |D2V|~262): plain FV 2.118, INS 2.758 ; FD lambda 2.141
  smooth AoG (sigma=0.4, |D2V|~7  ): plain FV 2.788, INS 2.707 ; FD lambda 2.671
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

# (label, FD lambda, plainFV E[c], INS E[c], maxD2V)
cases = [
    ("sharp AoG\n$|D^2V|\\approx262$", 2.141, 2.118, 2.758, 262),
    ("smooth AoG\n$|D^2V|\\approx7$",  2.671, 2.788, 2.707, 7),
]

fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8))
x = np.arange(len(cases)); w = 0.36
fv_bias = [100 * (c[2] - c[1]) / c[1] for c in cases]
ins_bias = [100 * (c[3] - c[1]) / c[1] for c in cases]

b1 = ax.bar(x - w / 2, fv_bias, w, label="plain FV (no swap)", color="0.55", edgecolor="k")
b2 = ax.bar(x + w / 2, ins_bias, w, label="INS (infinite swap)", color="0.15", edgecolor="k")
ax.axhline(0, color="red", ls="--", lw=1.2, label="exact (FD $\\lambda$)")

for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:+.0f}%", (b.get_x() + b.get_width() / 2, h),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases])
ax.set_ylabel(r"eigenvalue bias  $(E[c]-\lambda)/\lambda$")
ax.set_title("INS bias flips worst$\\to$best when the potential is smoothed\n"
             "(h0=0.1, T=50, N=20)", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.margins(y=0.18)
fig.tight_layout()
out = os.path.join(HERE, "figures", "bias_flip.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
