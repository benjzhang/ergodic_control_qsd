import numpy as np
import importlib
import matplotlib.pyplot as plt
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate
import plot_utils
importlib.reload(plot_utils)
from plot_utils import plot_periodic_trajectories, plot_periodic_trajectories_list
import os
import json




def periodic_cos(T, epsilon, h0, Nparticles):
    # periodic potential problem
    def V(x): 
        return np.cos(2 * np.pi * x) / (2 * np.pi)

    def DV(x):
        return -np.sin(2 * np.pi * x) 

    def D2V(x):
        return -2 * np.pi * np.cos(2 * np.pi * x)

    def a(x): 
        return  2 * epsilon * np.ones_like(x)

    def c(x):
        return 0

    dim = 1

    # Create the directory if it doesn't exist
    output_dir = f'periodic/T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}'
    os.makedirs(output_dir, exist_ok=True)

    initial_positionsx = np.zeros((Nparticles,dim)) + 0.5 
    initial_positionsy = np.zeros((Nparticles,dim)) + 0.5


    # Reference simulations
    simple_trajectories_times = []
    simple_trajectories_x = []

    for replicate in range(100):  # Replicate the process 10 times
        replicate_times = []
        replicate_trajectories = []
        for i in range(2 * Nparticles):
            traj_x, times = pure_jump_approx_diffusion(T, lambda x: -DV(x), a, h0, np.array([0.5]))
            replicate_times.append(times)
            traj_x = -1 + np.mod(traj_x + 1, 2)
            replicate_trajectories.append(traj_x)
        simple_trajectories_times.append(replicate_times)
        simple_trajectories_x.append(replicate_trajectories)

    fig0 = plot_periodic_trajectories_list(simple_trajectories_times[0], simple_trajectories_x[0])
    fig0.suptitle(f'Simple Trajectories: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig0.savefig(os.path.join(output_dir, 'simple_trajectories_plot.png'))

    plt.close(fig0)

    # Fleming-Viot simulation
    allpositionsx_replicates = []
    allpositionsy_replicates = []
    alltime_replicates = []
    all_rho_replicates = []

    for replicate in range(100):  # Replicate the process 10 times
        allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T, V, DV, D2V, a, 2*epsilon, c, h0, initial_positionsx, initial_positionsy)
        allpositionsx = -1 + np.mod(allpositionsx + 1, 2)
        allpositionsy = -1 + np.mod(allpositionsy + 1, 2)
        allpositionsx_replicates.append(allpositionsx)
        allpositionsy_replicates.append(allpositionsy)
        alltime_replicates.append(alltime)
        all_rho_replicates.append(all_rho)

    # Plot only the first replicate
    fig1 = plot_periodic_trajectories(alltime_replicates[0], allpositionsx_replicates[0][:, 0, 0])
    fig1.suptitle(f'All Positions X[0] (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig1.savefig(os.path.join(output_dir, 'allpositionsx_0_first_replicate_plot.png'))

    fig2 = plot_periodic_trajectories(alltime_replicates[0], allpositionsy_replicates[0][:, 0, 0])
    fig2.suptitle(f'All Positions Y[0] (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig2.savefig(os.path.join(output_dir, 'allpositionsy_0_first_replicate_plot.png'))

    fig3 = plot_periodic_trajectories(alltime_replicates[0], allpositionsx_replicates[0])
    fig3.suptitle(f'All Positions X (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig3.savefig(os.path.join(output_dir, 'allpositionsx_first_replicate_plot.png'))

    fig4 = plot_periodic_trajectories(alltime_replicates[0], allpositionsy_replicates[0])
    fig4.suptitle(f'All Positions Y (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig4.savefig(os.path.join(output_dir, 'allpositionsy_first_replicate_plot.png'))

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

    # Save parameters and results 
    params = {
        'T': T,
        'epsilon': epsilon,
        'h0': h0,
        'Nparticles': Nparticles
    }

    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f)

    # Save simulation results using numpy save function
    np.save(os.path.join(output_dir, 'alltime_replicates.npy'), np.array(alltime_replicates, dtype=object))
    np.save(os.path.join(output_dir, 'allpositionsx_replicates.npy'), np.array(allpositionsx_replicates, dtype=object))
    np.save(os.path.join(output_dir, 'allpositionsy_replicates.npy'), np.array(allpositionsy_replicates, dtype=object))
    np.save(os.path.join(output_dir, 'all_rho_replicates.npy'), np.array(all_rho_replicates, dtype=object))
    np.save(os.path.join(output_dir, 'simple_trajectories_times.npy'), np.array(simple_trajectories_times, dtype=object))
    np.save(os.path.join(output_dir, 'simple_trajectories_x.npy'), np.array(simple_trajectories_x, dtype=object))




# periodic_cos(25, 0.1, 0.025, 10)


T_values = [100]
epsilon_values = [0.05, 0.1]
h0_values = [0.1]
Nparticles_values = [20, 25]

for T in T_values:
    for epsilon in epsilon_values:
        for h0 in h0_values:
            for Nparticles in Nparticles_values:
                periodic_cos(T, epsilon, h0, Nparticles)


print("done")
