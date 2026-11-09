"""Score-based control on the smooth AoG.

Denoising score matching (a score-based generative model) learns s_theta ~ grad log nu
from the INS QSD samples; the ergodic-control law is then
    u*(x) = a * grad log nu + 2 grad V  ~  a * s_theta(x) + 2 * DV(x).
Validates s_theta against the FD reference and draws two presentation figures:
  (1) INS particle locations (the QSD sample cloud), greyscale V backdrop + red points
  (2) learned vs exact control field u*(x) (quiver over greyscale nu) + rel. error

    python score_control_smooth.py [--epochs 4000 --sigma 0.05 --harmonics 4]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem_smooth as P

L = P.L
a = P.a_coeff
HERE = os.path.dirname(__file__)


def fourier_features(x, n_harm):
    feats = []
    for k in range(1, n_harm + 1):
        feats += [torch.cos(2 * np.pi * k * x / L), torch.sin(2 * np.pi * k * x / L)]
    return torch.cat(feats, dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, n_harm=4, hidden=256, depth=3):
        super().__init__()
        self.n_harm = n_harm
        layers = [nn.Linear(4 * n_harm, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(fourier_features(x, self.n_harm))


def train(samples, epochs, sigma, n_harm, batch=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(samples, dtype=torch.float32)
    model = ScoreNet(n_harm=n_harm)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    M = X.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, M, (batch,))
        x = X[idx]; eps = torch.randn_like(x)
        xt = torch.remainder(x + sigma * eps, L)
        s = model(xt); target = -eps / sigma
        loss = ((s - target) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 8) == 0:
            print(f"  epoch {ep+1:5d}/{epochs}  dsm loss {loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def score_np(model, pts):
    return model(torch.tensor(P.wrap(pts), dtype=torch.float32)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--harmonics", type=int, default=4)
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, "qsd_samples_smooth.npy"), allow_pickle=True).item()
    samples = d["samples"]
    fd = np.load(os.path.join(HERE, "fd_smooth.npy"), allow_pickle=True).item()
    N = fd["N"]; pts = fd["pts"]; g = fd["g"]

    print(f"training DSM on {len(samples)} INS QSD samples ...", flush=True)
    model = train(samples, args.epochs, args.sigma, args.harmonics)

    # ---- score validation (nu-weighted) ----
    s_learn = score_np(model, pts); s_exact = fd["grad_log_nu"]; nu = fd["nu"]
    wm = nu / nu.sum()
    rel = np.sqrt(np.sum(wm * np.sum((s_learn - s_exact) ** 2, 1)) /
                  np.sum(wm * np.sum(s_exact ** 2, 1)))
    print(f"nu-weighted relative score error = {rel:.3f}")

    # ---- control fields  u* = a s + 2 DV ----
    gV = P.DV(pts)
    u_learn = a * s_learn + 2 * gV
    u_exact = a * s_exact + 2 * gV

    Vg = P.V(pts).reshape(N, N)
    nug = nu.reshape(N, N)

    # ===== Figure 1: INS particle locations (greyscale V + red points) =====
    pcloud = samples[np.random.default_rng(0).choice(len(samples), min(3500, len(samples)), replace=False)]
    fig1, ax1 = plt.subplots(figsize=(5.4, 5.2))
    ax1.imshow(nug.T, origin="lower", extent=[0, L, 0, L], cmap="gray")   # QSD nu backdrop
    ax1.scatter(pcloud[:, 0], pcloud[:, 1], s=4, c="red", alpha=0.35, linewidths=0)
    ax1.set_title("INS particle locations over the QSD $\\nu$")
    ax1.set_xticks([]); ax1.set_yticks([])
    fig1.tight_layout()
    fig1.savefig(os.path.join(HERE, "figures", "ins_locations_smooth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ===== Figure 2: control field u*, learned vs exact (quiver over greyscale nu) =====
    step = max(1, N // 22); sub = (slice(None, None, step), slice(None, None, step))
    XX, YY = np.meshgrid(g, g, indexing="ij")
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5.4))
    for a_, (U, ttl) in zip(ax2, [(u_exact.reshape(N, N, 2), "exact control $u^*$ (FD)"),
                                  (u_learn.reshape(N, N, 2), "learned control $u^*$ (score model)")]):
        a_.imshow(nug.T, origin="lower", extent=[0, L, 0, L], cmap="gray")
        a_.quiver(XX[sub], YY[sub], U[sub][..., 0], U[sub][..., 1], color="red", scale=90, width=0.004)
        a_.set_title(ttl); a_.set_xticks([]); a_.set_yticks([])
    fig2.suptitle("Ergodic-control law $u^*=a\\,\\nabla\\log\\nu+2\\nabla V$ from a score model on INS samples "
                  f"(rel. error {rel:.2f})", fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(HERE, "figures", "control_smooth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("saved figures/ins_locations_smooth.png + figures/control_smooth.png")
    with open(os.path.join(HERE, "score_control_smooth.txt"), "w") as f:
        f.write(f"nu-weighted relative score error = {rel:.4f}\n")


if __name__ == "__main__":
    main()
