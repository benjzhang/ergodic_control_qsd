import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate, weighted_empirical_measure_functional, empirical_measure_functional
from utils.plotting import plot_periodic_trajectories, plot_periodic_trajectories_list
import os
import json


T = 10
epsilon = 0.05
h0 = 0.05
Nparticles = 20



# def periodic_cos(T, epsilon, h0, Nparticles):
# periodic potential problem
def V(x): 
    return np.cos(2 * np.pi * x) / (2 * np.pi)

def DV(x):
    return -  np.sin(2 * np.pi * x) 

def D2V(x):
    return - 2 * np.pi * np.cos(2 * np.pi * x)

def a(x): 
    return  2 * epsilon * np.ones_like(x)

def c(x):
    return 0

dim = 1


initial_positionsx = np.zeros((Nparticles,dim)) 
initial_positionsy = np.zeros((Nparticles,dim)) 

# Reference simulations
simple_trajectories_times = []
simple_trajectories_x = []

for i in range(2 * Nparticles):
    traj_x, times = pure_jump_approx_diffusion(T, lambda x: -DV(x), a, h0, np.array([0.]))
    simple_trajectories_times.append(times)
    traj_x = -1 + np.mod(traj_x + 1, 2)
    simple_trajectories_x.append(traj_x)

fig0 = plot_periodic_trajectories_list(simple_trajectories_times, simple_trajectories_x)
fig0.suptitle(f'Simple Trajectories: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')

# fig01 =  plot_periodic_trajectories_list(simple_trajectories_times[0], simple_trajectories_x[0])
# fig0.suptitle(f'Simple Trajectories: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')




# Fleming-Viot simulation
allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T,V,DV,D2V,a,epsilon,c,h0,initial_positionsx,initial_positionsy)

allpositionsx = -1 + np.mod(allpositionsx + 1, 2)
allpositionsy = -1 + np.mod(allpositionsy + 1, 2)


# plotting

fig1 = plot_periodic_trajectories(alltime, allpositionsx[:, 0, 0])
fig1.suptitle(f'All Positions X[0]: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
# fig1.savefig(os.path.join(output_dir, 'allpositionsx_0_plot.png'))

fig2 = plot_periodic_trajectories(alltime, allpositionsy[:, 0, 0])
fig2.suptitle(f'All Positions Y[0]: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
# fig2.savefig(os.path.join(output_dir, 'allpositionsy_0_plot.png'))

fig3 = plot_periodic_trajectories(alltime, allpositionsx)
fig3.suptitle(f'All Positions X: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
# fig3.savefig(os.path.join(output_dir, 'allpositionsx_plot.png'))

fig4 = plot_periodic_trajectories(alltime, allpositionsy)
fig4.suptitle(f'All Positions Y: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
# fig4.savefig(os.path.join(output_dir, 'allpositionsy_plot.png'))



def functional(x,y):
    return np.exp(-x)



# compute expectations

# reference trajectories


results_reference = []
for i in range(Nparticles):
    result = empirical_measure_functional(simple_trajectories_x[i], lambda x: np.exp(-x), simple_trajectories_times[i])
    results_reference.append(result)
results_reference = np.array(results_reference)

results_IS_FV = []
for i in range(Nparticles):
    result = weighted_empirical_measure_functional(
        allpositionsx[:, i, 0], allpositionsy[:, i, 0], all_rho[:, i, 0], functional, alltime
    )
    results_IS_FV.append(result)
results_IS_FV = np.array(results_IS_FV)
