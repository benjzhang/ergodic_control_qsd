"""Sign-duality check for the ergodic control problem.

The QSD nu is sampled from the BASE killed process (drift -grad V). The optimally
controlled process that realizes rho* = phi^2 m uses drift  +grad V + a grad log nu
(the grad V term FLIPS SIGN). Using -grad V instead drives the process to the WRONG
measure phi^2 m^3.

This script uses the EXACT FD score (grad log nu) to isolate the sign question from
score-matching error, simulates both drifts, and plots them against the two targets.

    python sign_check.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem as P
from control_test import interp, simulate

L = P.L
a = P.a_coeff
HERE = os.path.dirname(__file__)


def main():
    fd = np.load(os.path.join(HERE, "fd_reference.npy"), allow_pickle=True).item()
    N, h, pts = fd["N"], fd["h"], fd["pts"]
    gln = fd["grad_log_nu"]                 # EXACT score of the QSD nu
    gv = P.DV(pts)                          # +grad V

    field_pos = (gv + a * gln).reshape(N, N, 2)     # +gradV + a*score  -> rho* = phi^2 m
    field_neg = (-gv + a * gln).reshape(N, N, 2)    # -gradV + a*score  -> phi^2 m^3 (wrong)

    rng = np.random.default_rng(0)
    s_pos = simulate(field_pos, h, 4000, 12000, 1e-3, 0.3, rng)
    s_neg = simulate(field_neg, h, 4000, 12000, 1e-3, 0.3, rng)

    rho = fd["rho"] / (fd["rho"].sum() * h ** 2)                          # phi^2 m

    def emp(samps, bins=N):
        H, _, _ = np.histogram2d(samps[:, 0], samps[:, 1], bins=bins, range=[[0, L], [0, L]], density=True)
        return H

    def l1(p, q):
        p = p / p.sum(); q = q / q.sum(); return 0.5 * np.sum(np.abs(p - q))

    l1_pos = l1(emp(s_pos).ravel(), rho); l1_neg = l1(emp(s_neg).ravel(), rho)
    print(f"+gradV controlled  L1 to rho* = {l1_pos:.3f}")
    print(f"-gradV controlled  L1 to rho* = {l1_neg:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = [
        (rho.reshape(N, N), r"target  $\rho^*=\phi^2 m$"),
        (emp(s_pos, N // 2), r"controlled  $+\nabla V + a\,\nabla\log\nu$" + f"\nL1 to $\\rho^*$ = {l1_pos:.3f}   (correct sign) ✓"),
        (emp(s_neg, N // 2), r"controlled  $-\nabla V + a\,\nabla\log\nu$" + f"\nL1 to $\\rho^*$ = {l1_neg:.3f}   (flipped sign) ✗"),
    ]
    for a_, (F, ttl) in zip(ax, panels):
        a_.imshow(F.T, origin="lower", extent=[0, L, 0, L], cmap="viridis")
        a_.set_title(ttl, fontsize=11); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Sign duality: controlled process needs +grad V (base sampler uses -grad V) "
                 r"$\to$ only +grad V recovers $\rho^*$", fontsize=12, y=1.06)
    fig.subplots_adjust(top=0.82)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "sign_check.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/sign_check.png")


if __name__ == "__main__":
    main()
