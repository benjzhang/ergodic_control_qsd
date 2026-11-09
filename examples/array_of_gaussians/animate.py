"""Make FV and vanilla animations (GIF) + snapshot PNGs from a saved AOG run.

Frames are subsampled evenly over [0, T] and rendered at low dpi with a coarse
(flat-band) contour background so the GIF stays small.

    python animate.py --data outputs/<tag>/aog_data.npy [--frames 80 --fps 20]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import problem as P

_GIF_DPI = 72


def _bg(ax, Xg, Yg, Z):
    ax.contourf(Xg, Yg, Z, levels=15, cmap="gray", vmin=0, vmax=10)
    ax.set_xlim(P.domain)
    ax.set_ylim(P.domain)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _idx(times, t):
    return max(np.searchsorted(np.asarray(times), t, side="right") - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True).item()
    apx, apy, at = d["allpositionsx"], d["allpositionsy"], np.asarray(d["alltime"])
    simple_t, simple_x = d.get("simple_t"), d.get("simple_x")
    figs = os.path.join(os.path.dirname(args.data), "figures")
    os.makedirs(figs, exist_ok=True)
    T = at[-1]
    snaps = np.linspace(0.0, T, args.frames)

    gx = np.linspace(P.domain[0], P.domain[1], 200)
    Xg, Yg = np.meshgrid(gx, gx)
    Z = P.V(np.stack([Xg, Yg], axis=-1))

    # ---- FV animation (forward X red, backward Y cyan) ----
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    def fv_upd(k):
        ax.clear()
        _bg(ax, Xg, Yg, Z)
        i = _idx(at, snaps[k])
        ax.scatter(apx[i, :, 0], apx[i, :, 1], s=30, color="red", edgecolors="k", linewidths=0.3, label="X")
        ax.scatter(apy[i, :, 0], apy[i, :, 1], s=30, color="cyan", edgecolors="k", linewidths=0.3, label="Y")
        ax.set_title("Fleming-Viot   t = %.1f" % snaps[k])

    animation.FuncAnimation(fig, fv_upd, frames=args.frames).save(
        os.path.join(figs, "fv_animation.gif"), writer="pillow", fps=args.fps, dpi=_GIF_DPI)
    plt.close(fig)
    print("saved fv_animation.gif")

    # ---- vanilla animation ----
    if simple_t is not None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        def van_upd(k):
            ax.clear()
            _bg(ax, Xg, Yg, Z)
            for i in range(len(simple_t)):
                xi = np.asarray(simple_x[i])
                j = _idx(simple_t[i], snaps[k])
                ax.scatter(xi[j, 0], xi[j, 1], s=30, color="red", edgecolors="k", linewidths=0.3)
            ax.set_title("Vanilla   t = %.1f" % snaps[k])

        animation.FuncAnimation(fig, van_upd, frames=args.frames).save(
            os.path.join(figs, "vanilla_animation.gif"), writer="pillow", fps=args.fps, dpi=_GIF_DPI)
        plt.close(fig)
        print("saved vanilla_animation.gif")

    # ---- FV snapshot PNGs ----
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = _idx(at, frac * T)
        fig, ax = plt.subplots(figsize=(5, 5))
        _bg(ax, Xg, Yg, Z)
        ax.scatter(apx[i, :, 0], apx[i, :, 1], s=32, color="red", edgecolors="k", linewidths=0.3, label="X")
        ax.scatter(apy[i, :, 0], apy[i, :, 1], s=32, color="cyan", edgecolors="k", linewidths=0.3, label="Y")
        ax.set_title("FV snapshot   t = %.1f" % (frac * T))
        ax.legend(loc="upper right", fontsize="small")
        fig.savefig(os.path.join(figs, "fv_snapshot_t%02d.png" % round(frac * T)), dpi=130, bbox_inches="tight")
        plt.close(fig)
    print("saved FV snapshots")


if __name__ == "__main__":
    main()
