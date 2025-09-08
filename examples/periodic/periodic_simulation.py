"""
Main simulation file for periodic potential Fleming-Viot system.

This file contains the core simulation function that runs a single
Fleming-Viot simulation with specified parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate
import json


def periodic_cos(T, epsilon, h0, Nparticles):
    """
    Run Fleming-Viot simulation for periodic cosine potential.
    
    Parameters:
    - T: time horizon
    - epsilon: temperature parameter  
    - h0: mean step size
    - Nparticles: number of particles
    
    Returns:
    - Dictionary containing all simulation results and metadata
    """
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

    for replicate in range(3):  # Reduced replicates for quick test
        replicate_times = []
        replicate_trajectories = []
        
        for i in range(Nparticles):
            initial_position = np.array([np.random.uniform(-1, 1)])  # Random initial position between -1 and 1
            X_reference, t_reference = pure_jump_approx_diffusion(T, lambda x: -DV(x) / epsilon, a, h0, initial_position)
            replicate_times.append(t_reference)
            replicate_trajectories.append(X_reference)
        
        simple_trajectories_times.append(replicate_times)
        simple_trajectories_x.append(replicate_trajectories)


    # Infinite swapping simulations

    alltime_replicates = []
    allpositionsx_replicates = []
    allpositionsy_replicates = []
    all_rho_replicates = []

    for replicate in range(3):  # Reduced replicates for quick test
        allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T, V, DV, D2V, a, epsilon, c, h0, initial_positionsx, initial_positionsy)
        
        alltime_replicates.append(alltime)
        allpositionsx_replicates.append(allpositionsx)
        allpositionsy_replicates.append(allpositionsy)
        all_rho_replicates.append(all_rho)

    # Save all data
    np.save(os.path.join(output_dir, 'alltime_replicates.npy'), alltime_replicates)
    np.save(os.path.join(output_dir, 'allpositionsx_replicates.npy'), allpositionsx_replicates)
    np.save(os.path.join(output_dir, 'allpositionsy_replicates.npy'), allpositionsy_replicates)
    np.save(os.path.join(output_dir, 'all_rho_replicates.npy'), all_rho_replicates)
    np.save(os.path.join(output_dir, 'simple_trajectories_times.npy'), simple_trajectories_times)
    np.save(os.path.join(output_dir, 'simple_trajectories_x.npy'), simple_trajectories_x)

    # Save parameters 
    params = {
        'T': T,
        'epsilon': epsilon,
        'h0': h0,
        'Nparticles': Nparticles
    }

    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f)

    print(f"Simulation completed: T={T}, epsilon={epsilon}, h0={h0}, Nparticles={Nparticles}")
    print(f"Data saved to {output_dir}/")
    
    return {
        'output_dir': output_dir,
        'params': params,
        'alltime_replicates': alltime_replicates,
        'allpositionsx_replicates': allpositionsx_replicates,
        'allpositionsy_replicates': allpositionsy_replicates,
        'all_rho_replicates': all_rho_replicates,
        'simple_trajectories_times': simple_trajectories_times,
        'simple_trajectories_x': simple_trajectories_x
    }


if __name__ == "__main__":
    # Default parameters for standalone execution
    T = 10
    epsilon = 0.05
    h0 = 0.05
    Nparticles = 20
    
    result = periodic_cos(T, epsilon, h0, Nparticles)
    print("Standalone simulation completed!")