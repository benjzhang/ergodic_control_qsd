"""
Main simulation file for periodic potential Fleming-Viot system.

This file contains functions to run individual simulations:
- Vanilla pure jump diffusion simulation (single particles)
- Fleming-Viot infinite swapping simulation (coupled particle system)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
from utils import pure_jump_approx_diffusion, fleming_viot
import json


def _define_periodic_potential():
    """
    Define the periodic cosine potential and its derivatives.
    
    Returns:
    - V, DV, D2V: potential and its first/second derivatives
    - a: diffusion coefficient function
    - c: additional coefficient (unused, set to 0)
    """
    def V(x): 
        return np.cos(2 * np.pi * x) / (2 * np.pi)

    def DV(x):
        return -np.sin(2 * np.pi * x) 

    def D2V(x):
        return -2 * np.pi * np.cos(2 * np.pi * x)

    def a(x): 
        return 2 * np.ones_like(x)  # Will be multiplied by epsilon in simulation

    def c(x):
        return 0
    
    return V, DV, D2V, a, c


def run_vanilla_simulation(T, epsilon, h0, Nparticles):
    """
    Run vanilla pure jump diffusion simulation for N independent particles.
    
    Parameters:
    - T: time horizon
    - epsilon: temperature parameter
    - h0: mean step size
    - Nparticles: number of independent particles to simulate
    
    Returns:
    - simple_trajectories_times: list of time arrays for each particle
    - simple_trajectories_x: list of position arrays for each particle
    """
    V, DV, D2V, a, c = _define_periodic_potential()
    
    simple_trajectories_times = []
    simple_trajectories_x = []
    
    for i in range(Nparticles):
        # Random initial position between -1 and 1
        initial_position = np.array([np.random.uniform(-1, 1)])
        
        # Run pure jump approximation with scaled diffusion
        X_reference, t_reference = pure_jump_approx_diffusion(
            T, 
            lambda x: -DV(x) / epsilon, 
            lambda x: a(x) * epsilon, 
            h0, 
            initial_position
        )
        
        simple_trajectories_times.append(t_reference)
        simple_trajectories_x.append(X_reference)
    
    return simple_trajectories_times, simple_trajectories_x


def run_fleming_viot_simulation(T, epsilon, h0, Nparticles):
    """
    Run Fleming-Viot infinite swapping simulation.
    
    Parameters:
    - T: time horizon
    - epsilon: temperature parameter
    - h0: mean step size
    - Nparticles: number of particles in the system
    
    Returns:
    - allpositionsx: forward process positions
    - allpositionsy: backward process positions  
    - alltime: time points
    - all_rho: infinite swapping weights
    """
    V, DV, D2V, a, c = _define_periodic_potential()
    dim = 1
    
    # Initial positions (centered at 0.5)
    initial_positionsx = np.zeros((Nparticles, dim)) + 0.5 
    initial_positionsy = np.zeros((Nparticles, dim)) + 0.5
    
    # Run Fleming-Viot simulation with scaled diffusion
    allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(
        T, V, DV, D2V, 
        lambda x: a(x) * epsilon, 
        epsilon, c, h0, 
        initial_positionsx, initial_positionsy
    )
    
    return allpositionsx, allpositionsy, alltime, all_rho


def periodic_simulation(T, epsilon, h0, Nparticles):
    """
    Run complete periodic potential simulation (both vanilla and Fleming-Viot).
    
    This function orchestrates both simulation types and saves results to disk.
    
    Parameters:
    - T: time horizon
    - epsilon: temperature parameter
    - h0: mean step size
    - Nparticles: number of particles
    
    Returns:
    - Dictionary containing simulation results and metadata
    """
    # Create output directory
    output_dir = f'periodic/T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running simulations: T={T}, epsilon={epsilon}, h0={h0}, Nparticles={Nparticles}")
    
    # Run vanilla simulation
    print("  Running vanilla simulation...")
    simple_trajectories_times, simple_trajectories_x = run_vanilla_simulation(T, epsilon, h0, Nparticles)
    
    # Run Fleming-Viot simulation  
    print("  Running Fleming-Viot simulation...")
    allpositionsx, allpositionsy, alltime, all_rho = run_fleming_viot_simulation(T, epsilon, h0, Nparticles)
    
    # Save vanilla simulation data
    np.save(os.path.join(output_dir, 'simple_trajectories_times.npy'), simple_trajectories_times)
    np.save(os.path.join(output_dir, 'simple_trajectories_x.npy'), simple_trajectories_x)
    
    # Save Fleming-Viot simulation data
    np.save(os.path.join(output_dir, 'alltime.npy'), alltime)
    np.save(os.path.join(output_dir, 'allpositionsx.npy'), allpositionsx)
    np.save(os.path.join(output_dir, 'allpositionsy.npy'), allpositionsy)
    np.save(os.path.join(output_dir, 'all_rho.npy'), all_rho)
    
    # Save parameters
    params = {
        'T': T,
        'epsilon': epsilon,
        'h0': h0,
        'Nparticles': Nparticles
    }
    
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f)
    
    print(f"  Simulation completed! Data saved to {output_dir}/")
    
    return {
        'output_dir': output_dir,
        'params': params,
        'simple_trajectories_times': simple_trajectories_times,
        'simple_trajectories_x': simple_trajectories_x,
        'alltime': alltime,
        'allpositionsx': allpositionsx,
        'allpositionsy': allpositionsy,
        'all_rho': all_rho
    }


