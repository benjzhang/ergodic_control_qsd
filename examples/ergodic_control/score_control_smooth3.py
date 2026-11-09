"""Score-based control on the smooth AoG, v3: ANALYTIC-BASELINE score.

Control is u* = a grad log phi, and grad log nu = grad log phi - grad V.  Learning
grad log nu directly forces the net to fit the large fine-scale grad V wiggle (16
wells), whose errors get amplified into u*.  Instead supply grad V analytically and
let the net learn only the SMOOTH envelope h_theta ~ grad log phi:
    s_theta(x) = h_theta(x) - grad V(x)   (DSM score, matches grad log nu)
    u*(x)      = a * h_theta(x)           (control, clean by construction)
h_theta = grad Phi_theta (curl-free potential net).
"""
import argparse, os
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import problem_smooth as P

L = P.L; a = P.a_coeff; SW = P.sigma; HERE = os.path.dirname(__file__)
CENT = torch.tensor(P.centers, dtype=torch.float32)


def fourier_features(x, n):
    f = []
    for k in range(1, n + 1):
        f += [torch.cos(2*np.pi*k*x/L), torch.sin(2*np.pi*k*x/L)]
    return torch.cat(f, -1)


def DV_torch(x):
    """analytic grad V = softmax-weighted (x-c)/sigma^2, min-image; detached baseline."""
    d = x[:, None, :] - CENT
    d = (d + L/2) % L - L/2
    q = -(d**2).sum(-1) / (2*SW**2)
    w = torch.softmax(q, dim=1)
    return (w[..., None] * d / SW**2).sum(1)


class PotentialNet(nn.Module):
    def __init__(self, n_harm=5, hidden=256, depth=3):
        super().__init__()
        self.n_harm = n_harm
        ly = [nn.Linear(4*n_harm, hidden), nn.SiLU()]
        for _ in range(depth-1): ly += [nn.Linear(hidden, hidden), nn.SiLU()]
        ly += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*ly)
    def phi(self, x): return self.net(fourier_features(x, self.n_harm)).squeeze(-1)
    def hgrad(self, x):
        x = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(self.phi(x).sum(), x, create_graph=self.training)
        return g


def train(samples, epochs, sigma, n_harm, batch=8192, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(samples, dtype=torch.float32)
    m = PotentialNet(n_harm=n_harm); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr); M = X.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, M, (batch,)); x = X[idx]; eps = torch.randn_like(x)
        xt = torch.remainder(x + sigma*eps, L)
        s = m.hgrad(xt) - DV_torch(xt).detach()          # score = grad log phi - grad V
        loss = ((s - (-eps/sigma))**2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % max(1, epochs//8) == 0:
            print(f"  epoch {ep+1:5d}/{epochs}  dsm loss {loss.item():.4f}", flush=True)
    m.eval(); return m


def h_np(m, pts):
    return m.hgrad(torch.tensor(P.wrap(pts), dtype=torch.float32)).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--harmonics", type=int, default=5)
    args = ap.parse_args()
    d = np.load(os.path.join(HERE, "qsd_samples_smooth.npy"), allow_pickle=True).item()
    samples = d["samples"]
    fd = np.load(os.path.join(HERE, "fd_smooth.npy"), allow_pickle=True).item()
    N = fd["N"]; pts = fd["pts"]; g = fd["g"]; nu = fd["nu"]
    print(f"training analytic-baseline DSM on {len(samples)} INS QSD samples ...", flush=True)
    model = train(samples, args.epochs, args.sigma, args.harmonics)

    gV = P.DV(pts)
    h_learn = h_np(model, pts)                       # ~ grad log phi (smooth)
    h_exact = fd["grad_log_nu"] + gV                 # exact grad log phi = grad log nu + grad V
    wm = nu / nu.sum()
    rel = np.sqrt(np.sum(wm*np.sum((h_learn-h_exact)**2,1)) / np.sum(wm*np.sum(h_exact**2,1)))
    ml = np.linalg.norm(h_learn,axis=1); me = np.linalg.norm(h_exact,axis=1)
    good = me > 0.1*me.max()
    cos = np.sum(h_learn*h_exact,1)/(ml*me+1e-9)
    cos_w = np.sum((wm*good)*cos)/np.sum(wm*good)
    print(f"nu-weighted rel error (grad log phi) = {rel:.3f}   |   dir cos = {cos_w:.3f}")

    u_learn = a*h_learn; u_exact = a*h_exact
    nug = nu.reshape(N, N)

    pc = samples[np.random.default_rng(0).choice(len(samples), min(3500,len(samples)), replace=False)]
    f1, x1 = plt.subplots(figsize=(5.4,5.2))
    x1.imshow(nug.T, origin="lower", extent=[0,L,0,L], cmap="gray")
    x1.scatter(pc[:,0], pc[:,1], s=4, c="red", alpha=0.35, linewidths=0)
    x1.set_title("INS particle locations over the QSD $\\nu$"); x1.set_xticks([]); x1.set_yticks([])
    f1.tight_layout(); f1.savefig(os.path.join(HERE,"figures","ins_locations_smooth.png"), dpi=150, bbox_inches="tight"); plt.close(f1)

    step = max(1, N//22); sub = (slice(None,None,step), slice(None,None,step))
    XX, YY = np.meshgrid(g, g, indexing="ij")
    f2, x2 = plt.subplots(1,2, figsize=(11,5.4))
    for a_, (U, t) in zip(x2, [(u_exact.reshape(N,N,2),"exact control $u^*$ (FD)"),
                               (u_learn.reshape(N,N,2),"learned control $u^*$ (score model)")]):
        a_.imshow(nug.T, origin="lower", extent=[0,L,0,L], cmap="gray")
        a_.quiver(XX[sub], YY[sub], U[sub][...,0], U[sub][...,1], color="red", scale=90, width=0.004)
        a_.set_title(t); a_.set_xticks([]); a_.set_yticks([])
    f2.suptitle("Ergodic-control law $u^*=a\\,\\nabla\\log\\phi$ from a score model on INS samples "
                f"(dir. cos {cos_w:.2f}, rel. err {rel:.2f})", fontsize=11)
    f2.tight_layout(rect=[0,0,1,0.95]); f2.savefig(os.path.join(HERE,"figures","control_smooth.png"), dpi=150, bbox_inches="tight"); plt.close(f2)
    print("saved figures/ins_locations_smooth.png + figures/control_smooth.png")
    open(os.path.join(HERE,"score_control_smooth.txt"),"w").write(f"v3 analytic-baseline: rel error (grad log phi) = {rel:.4f}, dir cos = {cos_w:.4f}\n")


if __name__ == "__main__":
    main()
