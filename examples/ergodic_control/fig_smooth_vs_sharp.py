"""Figure: why INS needs a smooth potential.

Top row = sharp AoG (problem.py, sigma=0.2, centers on half-integers -> seam on a
barrier).  Bottom row = smooth AoG (problem_smooth.py, sigma=0.4, centers on
integers -> seam on a well).  Columns: V, |D2V| (the INS backward-channel curvature),
and a 1D slice of |D2V| across the torus seam.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem as SHARP
import problem_smooth as SMOOTH

HERE = os.path.dirname(__file__)
L = SHARP.L
g = np.linspace(0, L, 301)
X, Y = np.meshgrid(g, g)
pts = np.stack([X.ravel(), Y.ravel()], 1)

fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6))
for row, (P, tag) in enumerate([(SHARP, "sharp  sigma=0.2  (seam on barrier)"),
                                (SMOOTH, "smooth  sigma=0.4  (seam on well)")]):
    Vv = P.V(pts).reshape(X.shape)
    d2 = np.abs(P.D2V(pts)).reshape(X.shape)
    maxd2 = d2.max()

    # col 0: potential (greyscale)
    ax[row, 0].imshow(Vv.T, origin="lower", extent=[0, L, 0, L], cmap="gray")
    ax[row, 0].set_title(f"V(x)   [{tag}]")

    # col 1: |D2V| (shared perceptual scale is misleading across rows, so annotate max)
    im = ax[row, 1].imshow(d2.T, origin="lower", extent=[0, L, 0, L], cmap="gray")
    ax[row, 1].set_title(r"$|D^2V|$   max = %.0f" % maxd2)
    fig.colorbar(im, ax=ax[row, 1], fraction=0.046, pad=0.04)

    # col 2: |D2V| along the seam x1=0 (where sharp AoG peaks)
    seam = np.abs(P.D2V(np.stack([np.zeros_like(g), g], 1)))
    ax[row, 2].plot(g, seam, color="k", lw=1.6)
    ax[row, 2].axhline(2 * np.pi, color="red", ls="--", lw=1.2, label=r"$2\pi$ (eigenfunc. level)")
    ax[row, 2].set_title(r"$|D^2V|$ along seam $x_1=0$")
    ax[row, 2].set_xlabel(r"$x_2$"); ax[row, 2].set_ylim(0, max(30, seam.max() * 1.1))
    ax[row, 2].legend(fontsize=8, loc="upper right")

    for c in (0, 1):
        ax[row, c].set_xticks([]); ax[row, c].set_yticks([])

fig.suptitle("INS backward-channel rate ~ (1-rho)(c - D2V): the sharp AoG's |D2V| ~ 262 biases INS;\n"
             "the smooth AoG keeps |D2V| ~ 7 (eigenfunction level), where INS is unbiased.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "smooth_vs_sharp_D2V.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
