"""Headless CLI runner for the eigenfunction (c != 0) INS Fleming-Viot example.

Estimates the principal eigenvalue of  A = -V'(x) d/dx - eps d^2/dx^2 + c(x)
via E[c(X)] under the quasi-stationary distribution, comparing the vanilla
Fleming-Viot estimator against the infinite-swap (INS) weighted estimator, and
against an exact (finely-discretized, symmetric Schrodinger form) reference.

This is the c != 0 case, so it exercises BOTH kill/clone channels (forward and
backward) of the simulation scheme. Agg backend, no plt.show(), figures saved
to outputs/<tag>/figures/. Parameterized via CLI:

    python run.py --epsilon 0.25 --N 50 --T 25 [--h0 0.1 --burn 10]

Convention: a(x) = epsilon (sqrt(epsilon) noise), so the generator's second-order
coefficient is a/2 = epsilon/2 and the operator is
    -sin(2 pi x) phi' - (epsilon/2) phi'' + c(x) phi = lambda phi.
The exact reference uses a direct finite-difference solve with eps_op = epsilon/2.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import (fleming_viot, fleming_viot_vanilla,
                   empirical_measure_functional, weighted_empirical_measure_functional,
                   cumulative_empirical_measure_functional, cumulative_mean_weighted_empirical_measure,
                   resample_weighted_empirical_measure)

import problem as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=25.0)
    ap.add_argument('--epsilon', type=float, default=0.25)
    ap.add_argument('--N', type=int, default=50)
    ap.add_argument('--h0', type=float, default=0.1)
    ap.add_argument('--burn', type=float, default=10.0, help='burn-in time')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    T, epsilon, Nparticles, h0, burn_in = args.T, args.epsilon, args.N, args.h0, args.burn
    dim = P.dim

    tag = f"{P.name}_T{T:g}_eps{epsilon:g}_N{Nparticles}_h{h0:g}"
    out = args.out or os.path.join(os.path.dirname(__file__), 'outputs', tag)
    figs = os.path.join(out, 'figures')
    os.makedirs(figs, exist_ok=True)

    # problem functions
    V, DV, D2V, c = P.V, P.DV, P.D2V, P.c
    a = P.diffusion(epsilon)   # a(x) = epsilon -> generator (a/2) = epsilon/2

    # ---- exact reference: direct finite-difference solve (eps_op = epsilon/2 = a/2) ----
    truth, xg, f0 = P.exact_eigen(epsilon / 2)
    print(f"exact principal eigenvalue (eps_op={epsilon / 2}): {truth:.8f}")

    # ---- simulations ----
    ix = np.array([P.initial_x(epsilon) for _ in range(Nparticles)])   # particles start at +1
    iy = ix.copy()
    print("running vanilla Fleming-Viot...")
    pv, tv = fleming_viot_vanilla(T, lambda x: -DV(x), a, c, h0, ix.copy())
    print("vanilla events:", len(tv))
    print("running infinite-swap Fleming-Viot...")
    px, py, ti, rho = fleming_viot(T, V, DV, D2V, a, epsilon, c, h0, ix.copy(), iy.copy())
    print("INS events:", len(ti))

    # ---- burn-in ----
    mv = tv > burn_in; pv = pv[mv]; tv = tv[mv]
    mi = ti > burn_in; px = px[mi]; py = py[mi]; ti = ti[mi]; rho = rho[mi]

    def functional(x, y):
        return c(x)

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

    def save(fig, fname):
        fig.savefig(os.path.join(figs, fname), dpi=140, bbox_inches='tight'); plt.close(fig)

    # ensemble trajectories
    for lbl, tt, pp in [('vanilla', tv, pv), ('ins', ti, px)]:
        fig = plt.figure(figsize=(9, 5))
        for i in range(Nparticles):
            plt.plot(tt, pp[:, i, 0], alpha=0.4, lw=0.6)
        plt.ylim(-5, 5); plt.xlabel('Time'); plt.ylabel('Position')
        plt.title(f'Ensemble trajectories ({lbl})'); save(fig, f'trajectories_{lbl}.png')

    # cumulative eigenvalue estimate
    for lbl, tt, cum in [('vanilla', tv, cum_van), ('ins', ti, cum_ins)]:
        fig = plt.figure(figsize=(9, 5))
        for i in range(Nparticles):
            plt.plot(tt, cum[i], alpha=0.4, lw=0.6)
        plt.axhline(truth, color='k', ls='--', label=f'truth {truth:.4f}')
        plt.ylim(0, 0.3); plt.xlabel('Time'); plt.ylabel('cumulative E[c]')
        plt.title(f'Cumulative eigenvalue estimate ({lbl})'); plt.legend(); save(fig, f'cumulative_{lbl}.png')

    # variance of cumulative estimate (log-log)
    var_van = np.var(cum_van, axis=0); var_ins = np.var(cum_ins, axis=0)
    fig = plt.figure(figsize=(9, 5))
    plt.plot(tv, var_van, label='vanilla', color='C0'); plt.plot(ti, var_ins, label='infinite swap', color='C1')
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('Time'); plt.ylabel('variance')
    plt.title('Variance of cumulative eigenvalue estimate'); plt.legend(); save(fig, 'variance_comparison.png')

    # MSE vs truth (log-log)
    mse_van = np.mean((cum_van - truth) ** 2, axis=0); mse_ins = np.mean((cum_ins - truth) ** 2, axis=0)
    fig = plt.figure(figsize=(9, 5))
    plt.plot(tv, mse_van, label='standard', color='C0'); plt.plot(ti, mse_ins, label='infinite swap', color='C1')
    plt.xscale('log'); plt.yscale('log'); plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.xlabel('Time'); plt.ylabel('MSE'); plt.title('Cumulative MSE vs truth'); plt.legend()
    save(fig, 'mse_comparison.png')

    # resampled INS positions vs exact eigenfunction (separate panels, not overlaid)
    # NOTE: pool BOTH replicas (x weight dt*rho, y weight dt*(1-rho)) -> full symmetrized
    # measure, consistent with the eigenvalue estimator (forward-only would look asymmetric).
    resampled = np.concatenate([resample_weighted_empirical_measure(px[:, i, 0], rho[:, i, 0], ti, py[:, i, 0])
                                for i in range(Nparticles)])
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

    np.save(os.path.join(out, 'eig_data.npy'), {
        'truth': truth, 'res_van': res_van, 'res_ins': res_ins,
        'xg': xg, 'f0': f0,                       # exact FD grid + QSD (for replotting overlays)
        'px': px, 'py': py, 'ti': ti, 'rho': rho, # INS trajectories (post burn-in)
        'pv': pv, 'tv': tv,                       # vanilla trajectories (post burn-in)
        'parameters': {'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles, 'burn_in': burn_in},
    }, allow_pickle=True)
    print('\nsaved figures + data to', out)


if __name__ == '__main__':
    main()
