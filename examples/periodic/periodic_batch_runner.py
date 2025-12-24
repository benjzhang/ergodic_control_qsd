"""
Batch runner for periodic potential simulations.

This file runs multiple simulations with different parameter combinations
and replicates, using array-based storage with incremental saves.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from periodic_simulation import periodic_simulation
import time
import json


def save_checkpoint(output_dir, data_arrays, params, replicate_count):
    """Save current progress to disk."""
    
    # Save replicate arrays
    np.save(os.path.join(output_dir, 'alltime_replicates.npy'), data_arrays['alltime_replicates'])
    np.save(os.path.join(output_dir, 'allpositionsx_replicates.npy'), data_arrays['allpositionsx_replicates'])
    np.save(os.path.join(output_dir, 'allpositionsy_replicates.npy'), data_arrays['allpositionsy_replicates'])
    np.save(os.path.join(output_dir, 'all_rho_replicates.npy'), data_arrays['all_rho_replicates'])
    np.save(os.path.join(output_dir, 'simple_trajectories_times.npy'), data_arrays['simple_trajectories_times'])
    np.save(os.path.join(output_dir, 'simple_trajectories_x.npy'), data_arrays['simple_trajectories_x'])
    
    # Save parameters and progress info
    progress_info = params.copy()
    progress_info['completed_replicates'] = replicate_count
    progress_info['last_save_time'] = time.time()
    
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(progress_info, f)
    
    print(f"  💾 Checkpoint saved ({replicate_count} replicates)")


def run_batch_simulations(n_replicates=3, save_every=10):
    """
    Run simulations for multiple parameter combinations and replicates.
    
    Uses array-based storage with incremental saves every N replicates to
    protect against data loss in long-running simulations.
    
    Parameters:
    - n_replicates: number of replicate runs per parameter combination
    - save_every: save progress every N replicates (default: 10)
    """
    
    # Parameter combinations to test
    T_values = [2]
    epsilon_values = [0.1]
    h0_values = [0.1]
    Nparticles_values = [5]
    
    total_combinations = len(T_values) * len(epsilon_values) * len(h0_values) * len(Nparticles_values)
    
    print("Starting batch simulations...")
    print(f"Parameter combinations: {total_combinations}")
    print(f"Replicates per combination: {n_replicates}")
    print(f"Saving checkpoints every {save_every} replicates")
    print(f"Total runs: {total_combinations * n_replicates}")
    
    all_results = []
    total_start_time = time.time()
    
    for T in T_values:
        for epsilon in epsilon_values:
            for h0 in h0_values:
                for Nparticles in Nparticles_values:
                    
                    print(f"\n{'='*60}")
                    print(f"Processing parameter set: T={T}, ε={epsilon}, h0={h0}, N={Nparticles}")
                    print(f"{'='*60}")
                    
                    # Create output directory
                    output_dir = f'periodic/T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    params = {'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles}
                    
                    # Initialize data arrays
                    data_arrays = {
                        'alltime_replicates': [],
                        'allpositionsx_replicates': [], 
                        'allpositionsy_replicates': [],
                        'all_rho_replicates': [],
                        'simple_trajectories_times': [],
                        'simple_trajectories_x': []
                    }
                    
                    completed_replicates = 0
                    param_start_time = time.time()
                    
                    # Run replicates
                    for replicate in range(n_replicates):
                        print(f"\n--- Replicate {replicate + 1}/{n_replicates} ---")
                        
                        start_time = time.time()
                        
                        try:
                            # Run single simulation
                            result = periodic_simulation(T, epsilon, h0, Nparticles)
                            
                            # Collect data from this replicate
                            data_arrays['alltime_replicates'].append(result['alltime'])
                            data_arrays['allpositionsx_replicates'].append(result['allpositionsx'])
                            data_arrays['allpositionsy_replicates'].append(result['allpositionsy'])
                            data_arrays['all_rho_replicates'].append(result['all_rho'])
                            data_arrays['simple_trajectories_times'].append(result['simple_trajectories_times'])
                            data_arrays['simple_trajectories_x'].append(result['simple_trajectories_x'])
                            
                            # Clean up temporary single-replicate directory
                            import shutil
                            if os.path.exists(result['output_dir']) and result['output_dir'] != output_dir:
                                shutil.rmtree(result['output_dir'])
                            
                            completed_replicates += 1
                            elapsed = time.time() - start_time
                            print(f"  ✅ Completed in {elapsed:.1f}s")
                            
                            # Save checkpoint if needed
                            if completed_replicates % save_every == 0 or completed_replicates == n_replicates:
                                save_checkpoint(output_dir, data_arrays, params, completed_replicates)
                            
                        except Exception as e:
                            elapsed = time.time() - start_time
                            print(f"  ❌ Failed after {elapsed:.1f}s: {e}")
                            # Continue with other replicates even if one fails
                    
                    param_elapsed = time.time() - param_start_time
                    success_rate = completed_replicates / n_replicates * 100
                    
                    print(f"\n🎯 Parameter set completed!")
                    print(f"   Successful replicates: {completed_replicates}/{n_replicates} ({success_rate:.1f}%)")
                    print(f"   Total time: {param_elapsed/60:.1f} minutes")
                    print(f"   Data saved to: {output_dir}/")
                    
                    all_results.append({
                        'params': params,
                        'output_dir': output_dir,
                        'completed_replicates': completed_replicates,
                        'total_replicates': n_replicates,
                        'success_rate': success_rate,
                        'runtime_minutes': param_elapsed/60
                    })
    
    total_elapsed = time.time() - total_start_time
    successful_combinations = sum(1 for r in all_results if r['completed_replicates'] > 0)
    
    print(f"\n{'='*60}")
    print(f"🏁 BATCH SIMULATIONS COMPLETED!")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Successful parameter combinations: {successful_combinations}/{len(all_results)}")
    print(f"Total successful replicates: {sum(r['completed_replicates'] for r in all_results)}")
    
    return all_results


if __name__ == "__main__":
    # Run with more frequent saves for testing
    results = run_batch_simulations(n_replicates=3, save_every=2)
    
    # Print detailed summary
    print("\n📊 DETAILED SUMMARY:")
    for i, result in enumerate(results):
        params = result['params']
        status = "✅" if result['success_rate'] == 100 else "⚠️" if result['completed_replicates'] > 0 else "❌"
        print(f"{i+1:2d}. {status} T={params['T']}, ε={params['epsilon']}, h0={params['h0']}, N={params['Nparticles']}")
        print(f"     Replicates: {result['completed_replicates']}/{result['total_replicates']} ({result['success_rate']:.0f}%)")
        print(f"     Runtime: {result['runtime_minutes']:.1f}min")