"""
Headless runner for the eigenfunction (c != 0) example.

Estimates the principal eigenvalue of  A = -V'(x) d/dx - eps d^2/dx^2 + c(x)
via E[c(X)] under the quasi-stationary distribution, comparing the vanilla
Fleming-Viot estimator against the infinite-swap (INS) weighted estimator, and
against an exact (finely-discretized operator) reference.

Mirrors main_eigenfunction.py but is cluster-friendly: Agg backend, exact truth
recomputed for the run parameters, figures saved to disk, no plt.show().
This is the c != 0 case, so it exercises BOTH kill/clone channels (forward and
backward) of the simulation scheme.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import eigsh

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import (fleming_viot, fleming_viot_vanilla,
                   empirical_measure_functional, weighted_empirical_measure_functional,
                   cumulative_empirical_measure_functional, cumulative_mean_weighted_empirical_measure,
                   resample_weighted_empirical_measure)

# ---- parameters (match main_eigenfunction.py) ----
T = 25
epsilon = 0.0625   # lower temperature -> sharper, harder-to-cross modes
h0 = 0.1
Nparticles = 50
dim = 1
burn_in = 10

def V(x):   return np.cos(2 * np.pi * x) / (2 * np.pi)
def DV(x):  return -np.sin(2 * np.pi * x)
def D2V(x): return -2 * np.pi * np.cos(2 * np.pi * x)
def a(x):   return 2 * epsilon * np.ones_like(x)   # diffusion coeff -> generator (a/2)=epsilon
def c(x):   return x ** 2 / np.pi                   # NON-ZERO killing rate

out_dir = os.path.join(os.path.dirname(__file__), 'eig_output')
os.makedirs(out_dir, exist_ok=True)

# ---- exact reference via the symmetric Schrodinger form ----
# A = -V' d/dx - eps d^2/dx^2 + c is non-symmetric, so eigsh on it is invalid.
# It is self-adjoint in L^2(e^{-V/eps}); the ground-state transform gives the
# symmetric operator H = -eps d^2/dx^2 + Veff with
#     Veff = c + V'^2/(4 eps) - V''/2,
# whose smallest eigenvalue is the principal eigenvalue. The QSD density is then
#     nu ∝ psi0 * e^{-V/(2 eps)}   (reduces to e^{-V/eps} when c=0).
def exact_eigen(eps_op, L=8.0, N=6000):
    xg = np.linspace(-L, L, N); dx = xg[1] - xg[0]
    Veff = c(xg) + DV(xg) ** 2 / (4 * eps_op) - D2V(xg) / 2
    D2 = (eye(N, k=1) - 2 * eye(N) + eye(N, k=-1)) / dx ** 2
    H = csr_matrix(-eps_op * D2 + diags(Veff, 0))
    evals, evecs = eigsh(H, k=3, sigma=0, which='LM')
    idx = np.argsort(evals); lam = evals[idx][0]; psi0 = np.abs(evecs[:, idx][:, 0])
    nu = psi0 * np.exp(-V(xg) / (2 * eps_op)); nu /= np.trapezoid(nu, xg)
    return lam, xg, nu

truth, xg, f0 = exact_eigen(epsilon)   # eps_op = a/2 = epsilon = 0.125
print(f"exact principal eigenvalue (eps_op={epsilon}): {truth:.8f}")

# ---- simulations ----
ix = np.zeros((Nparticles, dim)) + 1
iy = np.zeros((Nparticles, dim)) + 1
print("running vanilla Fleming-Viot...")
pv, tv = fleming_viot_vanilla(T, lambda x: -DV(x), a, c, h0, ix.copy())
print("vanilla events:", len(tv))
print("running infinite-swap Fleming-Viot...")
px, py, ti, rho = fleming_viot(T, V, DV, D2V, a, 2 * epsilon, c, h0, ix.copy(), iy.copy())
print("INS events:", len(ti))

# ---- burn-in ----
mv = tv > burn_in; pv = pv[mv]; tv = tv[mv]
mi = ti > burn_in; px = px[mi]; py = py[mi]; ti = ti[mi]; rho = rho[mi]

def functional(x, y): return c(x)

# ---- eigenvalue estimates E[c(X)] ----
res_van = np.array([empirical_measure_functional(pv[:, i, 0], c, tv) for i in range(Nparticles)])
res_ins = np.array([weighted_empirical_measure_functional(px[:, i, 0], py[:, i, 0], rho[:, i, 0], functional, ti)
                    for i in range(Nparticles)])
print("\n=== principal eigenvalue estimate (mean over particles) ===")
print(f"  truth     : {truth:.6f}")
print(f"  vanilla   : {res_van.mean():.6f} +/- {res_van.std():.6f}")
print(f"  inf-swap  : {res_ins.mean():.6f} +/- {res_ins.std():.6f}")

# ---- cumulative estimates ----
cum_van = np.array([cumulative_empirical_measure_functional(pv[:, i, 0], c, tv) for i in range(Nparticles)])
cum_ins = np.array([cumulative_mean_weighted_empirical_measure(px[:, i, 0], py[:, i, 0], rho[:, i, 0], functional, ti)
                    for i in range(Nparticles)])

def save(fig, name):
    fig.savefig(os.path.join(out_dir, name), dpi=140, bbox_inches='tight'); plt.close(fig)

# ensemble trajectories
for tag, tt, pp in [('vanilla', tv, pv), ('ins', ti, px)]:
    fig = plt.figure(figsize=(9, 5))
    for i in range(Nparticles): plt.plot(tt, pp[:, i, 0], alpha=0.4, lw=0.6)
    plt.ylim(-5, 5); plt.xlabel('Time'); plt.ylabel('Position')
    plt.title(f'Ensemble trajectories ({tag})'); save(fig, f'trajectories_{tag}.png')

# cumulative eigenvalue estimate
for tag, tt, cum in [('vanilla', tv, cum_van), ('ins', ti, cum_ins)]:
    fig = plt.figure(figsize=(9, 5))
    for i in range(Nparticles): plt.plot(tt, cum[i], alpha=0.4, lw=0.6)
    plt.axhline(truth, color='k', ls='--', label=f'truth {truth:.4f}')
    plt.ylim(0, 0.3); plt.xlabel('Time'); plt.ylabel('cumulative E[c]')
    plt.title(f'Cumulative eigenvalue estimate ({tag})'); plt.legend(); save(fig, f'cumulative_{tag}.png')

# variance of cumulative estimate (log-log)
var_van = np.var(cum_van, axis=0); var_ins = np.var(cum_ins, axis=0)
fig = plt.figure(figsize=(9, 5))
plt.plot(tv, var_van, label='vanilla', color='C0'); plt.plot(ti, var_ins, label='infinite swap', color='C1')
plt.xscale('log'); plt.yscale('log'); plt.xlabel('Time'); plt.ylabel('variance')
plt.title('Variance of cumulative eigenvalue estimate'); plt.legend(); save(fig, 'variance_comparison.png')

# MSE vs truth (log-log)
mse_van = np.mean((cum_van - truth) ** 2, axis=0); mse_ins = np.mean((cum_ins - truth) ** 2, axis=0)
fig = plt.figure(figsize=(9, 5))
plt.plot(tv, mse_van, label='vanilla', color='C0'); plt.plot(ti, mse_ins, label='infinite swap', color='C1')
plt.xscale('log'); plt.yscale('log'); plt.grid(True, which='both', ls='--', alpha=0.4)
plt.xlabel('Time'); plt.ylabel('MSE'); plt.title('Cumulative MSE vs truth'); plt.legend()
save(fig, 'mse_comparison.png')

# resampled INS positions vs exact eigenfunction (separate panels, not overlaid)
resampled = np.concatenate([resample_weighted_empirical_measure(px[:, i, 0], rho[:, i, 0], ti) for i in range(Nparticles)])
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
panels = [(axes[0], resampled, 'resampled INS', 'C1'),
          (axes[1], pv[:, :, 0].ravel(), 'vanilla positions', 'C0')]
for ax, data, lbl, col in panels:
    ax.hist(data, bins=120, density=True, alpha=0.7, color=col, label=lbl)
    ax.plot(xg, f0, color='red', lw=2, label='exact eigenfunction')
    ax.set_xlim(-3, 3); ax.set_xlabel('x'); ax.legend(loc='upper left', fontsize='small')
axes[0].set_ylabel('density')
fig.suptitle('Empirical measure vs exact eigenfunction')
save(fig, 'eigenfunction_histogram.png')

np.save(os.path.join(out_dir, 'eig_data.npy'), {
    'truth': truth, 'res_van': res_van, 'res_ins': res_ins,
    'parameters': {'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles},
}, allow_pickle=True)
print('\nsaved figures + data to', out_dir)
