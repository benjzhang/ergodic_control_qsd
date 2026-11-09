"""
Headless runner for the array-of-Gaussians (2D, 5x5 wells on a torus) example.

Mirrors main_aog.py but is cluster-friendly: Agg backend, local output dir,
figures saved to disk instead of plt.show(), no hardcoded paths.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import pure_jump_approx_diffusion, fleming_viot

# ---- parameters ----
# Sharp wells (sigma=0.1 -> D2V~200) make the kill/clone event rate large
# (~1e4 events/unit time); fleming_viot now accumulates snapshots in O(n) so
# longer horizons are tractable.
T = 100
epsilon = 0.4
h0 = 0.1
Nparticles = 5
sigma = 0.1
dim = 2
left, right = 0, 4
period = right - left

centers = np.array([(i, j) for i in range(5) for j in range(5)])


def _wrap(x):
    return np.mod(x - left, period) + left


def V(x):
    xw = _wrap(x)
    d = -np.sum((xw[..., None, :] - centers) ** 2, axis=-1) / (2 * sigma ** 2)
    return -logsumexp(d, axis=-1)


def DV(x):
    xw = _wrap(x)
    d = -np.sum((xw[..., None, :] - centers) ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(d - logsumexp(d, axis=-1, keepdims=True))
    return -np.sum(w[..., None] * (centers - xw[..., None, :]) / (sigma ** 2), axis=-2)


def D2V(x):
    xw = _wrap(x)
    d = -np.sum((xw[..., None, :] - centers) ** 2, axis=-1) / (2 * sigma ** 2)
    w = np.exp(d - logsumexp(d, axis=-1, keepdims=True))
    lap = -np.sum(w * (-2 * d / (sigma ** 2) - dim / (sigma ** 2)), axis=-1) \
        + np.sum(DV(x) ** 2, axis=-1)
    return lap


def a(x):
    return 2 * epsilon * np.ones_like(x)


def c(x):
    return 0


out_dir = os.path.join(os.path.dirname(__file__), 'aog_output',
                       f'T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}')
fig_dir = os.path.join(out_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ---- grid for contour ----
gx = np.linspace(left, right, 200)
X, Y = np.meshgrid(gx, gx)
grid = np.stack([X, Y], axis=-1)
Z = V(grid)


def base_contour(title):
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=10)
    plt.colorbar(cf, label='V(x)')
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')


base_contour('Array of Gaussians: potential V(x)')
plt.savefig(os.path.join(fig_dir, 'potential_contour.png'), dpi=150, bbox_inches='tight')
plt.close()
print('saved potential_contour.png')

# ---- vanilla reference ----
init_x = np.zeros((Nparticles, dim)) + 2
init_y = np.zeros((Nparticles, dim)) + 2
simple_t, simple_x = [], []
for i in range(2 * Nparticles):
    tx, tt = pure_jump_approx_diffusion(T, lambda x: -DV(x), a, h0, init_x[0])
    simple_t.append(tt)
    simple_x.append(_wrap(np.asarray(tx)))
print('vanilla reference done: %d trajectories' % len(simple_x))

base_contour('Vanilla reference trajectories')
for traj in simple_x:
    plt.scatter(traj[:, 0], traj[:, 1], s=1, alpha=0.5)
plt.savefig(os.path.join(fig_dir, 'vanilla_trajectories.png'), dpi=150, bbox_inches='tight')
plt.close()
print('saved vanilla_trajectories.png')

# ---- Fleming-Viot ----
apx, apy, at, arho = fleming_viot(T, V, DV, D2V, a, epsilon, c, h0, init_x, init_y)
apx = _wrap(apx)
apy = _wrap(apy)
print('Fleming-Viot done: %d events' % len(at))

base_contour('Fleming-Viot forward (x) positions')
plt.scatter(apx[:, :, 0].ravel(), apx[:, :, 1].ravel(), s=1, alpha=0.4, color='red')
plt.savefig(os.path.join(fig_dir, 'fv_forward_positions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('saved fv_forward_positions.png')

# weighted (rho) empirical scatter: forward weighted by rho, backward by 1-rho
base_contour('Fleming-Viot weighted empirical measure')
plt.scatter(apx[:, :, 0].ravel(), apx[:, :, 1].ravel(), s=2,
            c=arho.ravel(), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
plt.scatter(apy[:, :, 0].ravel(), apy[:, :, 1].ravel(), s=2,
            c=(1 - arho).ravel(), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
plt.savefig(os.path.join(fig_dir, 'fv_weighted_positions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('saved fv_weighted_positions.png')

# ---- save data ----
np.save(os.path.join(out_dir, 'aog_data.npy'), {
    'allpositionsx': apx, 'allpositionsy': apy, 'alltime': at, 'all_rho': arho,
    'parameters': {'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles,
                   'sigma': sigma, 'dim': dim, 'left': left, 'right': right},
}, allow_pickle=True)
with open(os.path.join(out_dir, 'parameters.json'), 'w') as f:
    json.dump({'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles,
               'sigma': sigma, 'dim': dim, 'left': left, 'right': right}, f)
print('saved data + figures to', out_dir)
