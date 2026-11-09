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


def to_object_array(seq):
    """Pack a (possibly ragged) sequence of arrays into a 1-D object array.

    numpy>=2 raises on np.save/np.array of ragged nested sequences unless an
    object dtype is used. Building the object array element-by-element works
    regardless of whether the inner arrays happen to share a shape.
    """
    arr = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        arr[i] = v
    return arr


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


def run_vanilla_simulation(T, epsilon, h0, Nparticles, periodic_bc=True):
    """
    Run vanilla pure jump diffusion simulation for N independent particles.

    Parameters:
    - T: time horizon
    - epsilon: temperature parameter
    - h0: mean step size
    - Nparticles: number of independent particles to simulate
    - periodic_bc: if True, use periodic boundary conditions with period 2
                   (positions wrapped to [-1, 1])

    Returns:
    - simple_trajectories_times: list of time arrays for each particle
    - simple_trajectories_x: list of position arrays for each particle
    """
    V, DV, D2V, a, c = _define_periodic_potential()

    # Set wrap_period based on periodic_bc flag (period 2 -> domain [-1, 1], two periods of V)
    wrap_period = 2.0 if periodic_bc else None

    simple_trajectories_times = []
    simple_trajectories_x = []

    for i in range(Nparticles):
        # Random initial position in [-0.5, 0.5] (one period)
        initial_position = np.array([np.random.uniform(-0.5, 0.5)])

        # Run pure jump approximation. Drift -DV with diffusion a*epsilon (=2*epsilon)
        # gives invariant measure exp(-V/epsilon), matching the Fleming-Viot target.
        # (Previously -DV/epsilon, which yielded exp(-V/epsilon^2): too cold, poor mixing.)
        X_reference, t_reference = pure_jump_approx_diffusion(
            T,
            lambda x: -DV(x),
            lambda x: a(x) * epsilon,
            h0,
            initial_position,
            wrap_period=wrap_period
        )

        simple_trajectories_times.append(t_reference)
        simple_trajectories_x.append(X_reference)

    return simple_trajectories_times, simple_trajectories_x


def run_fleming_viot_simulation(T, epsilon, h0, Nparticles, periodic_bc=True):
    """
    Run Fleming-Viot infinite swapping simulation.

    Parameters:
    - T: time horizon
    - epsilon: temperature parameter
    - h0: mean step size
    - Nparticles: number of particles in the system
    - periodic_bc: if True, use periodic boundary conditions with period 2
                   (positions wrapped to [-1, 1])

    Returns:
    - allpositionsx: forward process positions
    - allpositionsy: backward process positions
    - alltime: time points
    - all_rho: infinite swapping weights
    """
    V, DV, D2V, a, c = _define_periodic_potential()
    dim = 1

    # Set wrap_period based on periodic_bc flag (period 2 -> domain [-1, 1], two periods of V)
    wrap_period = 2.0 if periodic_bc else None

    # Initial positions (centered at 0, the center of [-0.5, 0.5])
    initial_positionsx = np.zeros((Nparticles, dim))
    initial_positionsy = np.zeros((Nparticles, dim))

    # Run Fleming-Viot simulation with scaled diffusion and periodic BCs
    allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(
        T, V, DV, D2V,
        lambda x: a(x) * epsilon,
        epsilon, c, h0,
        initial_positionsx, initial_positionsy,
        wrap_period=wrap_period
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
    
    # Save vanilla simulation data (ragged per-particle arrays -> object array)
    np.save(os.path.join(output_dir, 'simple_trajectories_times.npy'), to_object_array(simple_trajectories_times))
    np.save(os.path.join(output_dir, 'simple_trajectories_x.npy'), to_object_array(simple_trajectories_x))
    
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


