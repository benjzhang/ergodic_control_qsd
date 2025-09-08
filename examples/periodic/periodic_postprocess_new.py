"""
Postprocessing analysis for periodic potential simulation results.

This file processes the simulation data and performs empirical measure
analysis, resampling, and statistical computations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
from utils import weighted_empirical_measure_functional, empirical_measure_functional, resample_simple_weighted_empirical_measure, resample_weighted_empirical_measure, fleming_viot_vanilla, cumulative_empirical_measure_functional, cumulative_mean_weighted_empirical_measure


def process_simulation_data(T=100, epsilon=0.1, h0=0.1, Nparticles=10):
    """
    Process simulation data for given parameters.
    
    Parameters:
    - T, epsilon, h0, Nparticles: simulation parameters
    
    Returns:
    - Dictionary containing all analysis results
    """
    
    output_dir = f'periodic/T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}'
    
    # Check if data exists
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        raise FileNotFoundError(f"Simulation data not found in {output_dir}")
    
    print(f"Processing data for T={T}, epsilon={epsilon}, h0={h0}, Nparticles={Nparticles}")
    
    # Load simulation data
    alltime_replicates = np.load(output_dir_path / 'alltime_replicates.npy', allow_pickle=True)
    allpositionsx_replicates = np.load(output_dir_path / 'allpositionsx_replicates.npy', allow_pickle=True)
    allpositionsy_replicates = np.load(output_dir_path / 'allpositionsy_replicates.npy', allow_pickle=True)
    all_rho_replicates = np.load(output_dir_path / 'all_rho_replicates.npy', allow_pickle=True)
    simple_trajectories_times = np.load(output_dir_path / 'simple_trajectories_times.npy', allow_pickle=True)
    simple_trajectories_x = np.load(output_dir_path / 'simple_trajectories_x.npy', allow_pickle=True)

    # Reference empirical measure analysis
    results_reference = []
    for j in range(3):  # Loop over replicates (reduced for test)
        replicate_results = []
        for i in range(Nparticles):
            # Apply burn-in by filtering out samples before T = 10
            valid_indices = simple_trajectories_times[j][i] >= 10
            result = empirical_measure_functional(
                simple_trajectories_x[j][i][valid_indices], 
                lambda x: np.exp(-x/epsilon), 
                simple_trajectories_times[j][i][valid_indices]
            )
            replicate_results.append(result)
        results_reference.append(np.mean(replicate_results))
    results_reference = np.array(results_reference)

    # Functional for weighted analysis
    def functional(x,y):
        return np.exp(-x/epsilon)

    # Weighted empirical measure analysis
    results_reference_weighted_allpositions = []
    for j in range(3):  # Loop over replicates (reduced for test)
        replicate_results = []
        for i in range(Nparticles):
            # Apply burn-in by filtering out samples before T = 10
            valid_indices = alltime_replicates[j] >= 10
            result = weighted_empirical_measure_functional(
                allpositionsx_replicates[j][valid_indices, i, 0], 
                allpositionsy_replicates[j][valid_indices, i, 0], 
                all_rho_replicates[j][valid_indices, i, 0], 
                functional, 
                alltime_replicates[j][valid_indices]
            )
            replicate_results.append(result)
        results_reference_weighted_allpositions.append(np.mean(replicate_results))
    results_reference_weighted_allpositions = np.array(results_reference_weighted_allpositions)

    # Resampling from simple FV empirical measure
    results_reference_weighted_allpositions_resampled = []
    for j in range(1):  # Loop over first replicate
        replicate_results = []
        for i in range(10):
            # Apply burn-in by filtering out samples before T = 10
            valid_indices = simple_trajectories_times[j][i] >= 10
            result = resample_simple_weighted_empirical_measure(
                simple_trajectories_x[j][i][valid_indices],  
                simple_trajectories_times[j][i][valid_indices]
            )
            replicate_results.extend(result)  # Flatten before appending
        results_reference_weighted_allpositions_resampled.append(replicate_results)

    # Flatten results for histogram plotting
    flattened_simple_trajectories_x = np.array(results_reference_weighted_allpositions_resampled)[0]

    # Resampling from weighted empirical measure
    results_INS_weighted_allpositions_resampled = []
    for j in range(2):  # Loop over first 2 replicates
        replicate_results = []
        for i in range(Nparticles):
            # Apply burn-in by filtering out samples before T = 10
            valid_indices = alltime_replicates[j] >= 10
            result = resample_weighted_empirical_measure(
                allpositionsx_replicates[j][valid_indices, i, 0], 
                all_rho_replicates[j][valid_indices, i, 0],  
                alltime_replicates[j][valid_indices]
            )
            replicate_results.extend(result)  # Flatten before appending
        results_INS_weighted_allpositions_resampled.append(replicate_results)

    # Flatten the resampled results
    flattened_INS_resampled_results = [item for sublist in results_INS_weighted_allpositions_resampled for item in sublist]

    # Compute final statistics
    mean_exp_simple = np.mean(np.exp(-np.array(flattened_simple_trajectories_x)/epsilon))
    mean_exp_INS = np.mean(np.exp(-np.array(flattened_INS_resampled_results)/epsilon))

    print(f"✅ Postprocessing completed!")
    print(f"   Reference empirical mean: {np.mean(results_reference):.6f} ± {np.std(results_reference):.6f}")
    print(f"   Weighted empirical mean: {np.mean(results_reference_weighted_allpositions):.6f} ± {np.std(results_reference_weighted_allpositions):.6f}")
    print(f"   Simple resampled exp mean: {mean_exp_simple:.6f}")
    print(f"   INS resampled exp mean: {mean_exp_INS:.6f}")

    return {
        'output_dir': output_dir,
        'results_reference': results_reference,
        'results_reference_weighted_allpositions': results_reference_weighted_allpositions,
        'flattened_simple_trajectories_x': flattened_simple_trajectories_x,
        'flattened_INS_resampled_results': flattened_INS_resampled_results,
        'mean_exp_simple': mean_exp_simple,
        'mean_exp_INS': mean_exp_INS,
        'epsilon': epsilon
    }


def run_postprocessing_batch():
    """
    Run postprocessing for all available simulation data.
    
    Automatically finds all simulation directories and processes them.
    """
    
    # Look for all simulation directories
    periodic_dirs = []
    if os.path.exists('periodic'):
        for item in os.listdir('periodic'):
            if item.startswith('T_') and os.path.isdir(f'periodic/{item}'):
                periodic_dirs.append(item)
    
    if not periodic_dirs:
        print("No simulation data found to process!")
        return []
    
    print(f"Found {len(periodic_dirs)} simulation datasets to process:")
    
    results = []
    for dir_name in periodic_dirs:
        try:
            # Parse parameters from directory name
            parts = dir_name.split('_')
            T = float(parts[1])
            epsilon = float(parts[3])
            h0 = float(parts[5])
            Nparticles = int(parts[7])
            
            print(f"\n--- Processing {dir_name} ---")
            result = process_simulation_data(T, epsilon, h0, Nparticles)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to process {dir_name}: {e}")
    
    print(f"\n✅ Postprocessing batch completed! Processed {len(results)} datasets")
    return results


if __name__ == "__main__":
    # Run postprocessing for all available data
    results = run_postprocessing_batch()
    
    if results:
        print("\nPostprocessing Summary:")
        for result in results:
            print(f"  {result['output_dir']}: Reference={np.mean(result['results_reference']):.4f}, Weighted={np.mean(result['results_reference_weighted_allpositions']):.4f}")
    else:
        print("No data processed. Run simulations first!")