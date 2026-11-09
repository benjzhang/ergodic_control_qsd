"""Stage 3: denoising score matching (PyTorch MLP) on the FV/INS QSD samples.

Learns s_theta(x) ~ grad log nu(x) on the torus.  Periodic Fourier features make
the network respect the [0,4]^2 periodicity.  Validates the learned score against
the FD reference grad log nu, and saves the model for the control stage.

    python score_match.py [--epochs 4000 --sigma 0.05 --harmonics 4]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem as P

L = P.L
HERE = os.path.dirname(__file__)


def fourier_features(x, n_harm):
    """x: (M,2) in [0,L). -> periodic features (M, 4*n_harm)."""
    feats = []
    for k in range(1, n_harm + 1):
        feats += [torch.cos(2 * np.pi * k * x / L), torch.sin(2 * np.pi * k * x / L)]
    return torch.cat(feats, dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, n_harm=4, hidden=256, depth=3):
        super().__init__()
        self.n_harm = n_harm
        din = 4 * n_harm
        layers = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(fourier_features(x, self.n_harm))


def wrap_t(x):
    return torch.remainder(x, L)


def train(samples, epochs=4000, sigma=0.05, n_harm=4, batch=4096, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    X = torch.tensor(samples, dtype=torch.float32, device=device)
    model = ScoreNet(n_harm=n_harm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    M = X.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, M, (batch,), device=device)
        x = X[idx]
        eps = torch.randn_like(x)
        xt = wrap_t(x + sigma * eps)                 # perturbed (wrapped to torus)
        target = -eps / sigma                        # (x - xt)/sigma^2  = -eps/sigma
        s = model(xt)
        loss = ((s - target) ** 2).sum(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {ep+1:5d}/{epochs}  dsm loss {loss.item():.4f}")
    return model


@torch.no_grad()
def score_np(model, pts):
    """Query s_theta at numpy points (M,2) -> (M,2)."""
    x = torch.tensor(P.wrap(pts), dtype=torch.float32)
    return model(x).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--harmonics", type=int, default=4)
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, "qsd_samples.npy"), allow_pickle=True).item()
    samples = d["samples"]
    print(f"training DSM on {len(samples)} QSD samples (sigma={args.sigma}, harmonics={args.harmonics})")
    model = train(samples, epochs=args.epochs, sigma=args.sigma, n_harm=args.harmonics)
    torch.save({"state_dict": model.state_dict(), "n_harm": args.harmonics, "sigma": args.sigma},
               os.path.join(HERE, "score_model.pt"))

    # ---- validate vs FD grad log nu ----
    fd = np.load(os.path.join(HERE, "fd_reference.npy"), allow_pickle=True).item()
    N = fd["N"]; pts = fd["pts"]; s_exact = fd["grad_log_nu"]
    s_learn = score_np(model, pts)
    nu = fd["nu"]
    # error weighted by where the QSD has mass (score only matters where nu>0)
    wmask = nu / nu.sum()
    rel = np.sqrt(np.sum(wmask * np.sum((s_learn - s_exact) ** 2, axis=1)) /
                  np.sum(wmask * np.sum(s_exact ** 2, axis=1)))
    print(f"nu-weighted relative score error ||s_learn - s_exact|| = {rel:.3f}")

    # quiver comparison (subsampled) + error map
    g = fd["g"]; step = max(1, N // 24)
    XX, YY = np.meshgrid(g, g, indexing="ij")
    sub = (slice(None, None, step), slice(None, None, step))
    sl = s_learn.reshape(N, N, 2); se = s_exact.reshape(N, N, 2)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a_, (S, ttl) in zip(ax[:2], [(se, "FD  grad log nu"), (sl, "learned  s_theta")]):
        a_.imshow(fd["nu"].reshape(N, N).T, origin="lower", extent=[0, L, 0, L], cmap="Greys", alpha=0.6)
        a_.quiver(XX[sub], YY[sub], S[sub][..., 0], S[sub][..., 1], color="red", scale=200)
        a_.set_title(ttl); a_.set_xticks([]); a_.set_yticks([])
    err = np.sqrt(np.sum((sl - se) ** 2, axis=2))
    im = ax[2].imshow(err.T, origin="lower", extent=[0, L, 0, L], cmap="magma")
    ax[2].set_title("|s_theta - grad log nu|"); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.colorbar(im, ax=ax[2])
    fig.suptitle(f"Score matching vs FD  (nu-weighted rel. error {rel:.3f})")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "score_vs_fd.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved score_model.pt + figures/score_vs_fd.png")


if __name__ == "__main__":
    main()
