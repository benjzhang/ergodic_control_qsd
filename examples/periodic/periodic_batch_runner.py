"""
Batch runner for periodic potential simulations.

This file runs multiple simulations with different parameter combinations,
calling the main simulation function for each set of parameters.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from periodic_simulation import periodic_cos
import time


def run_batch_simulations():
    """
    Run simulations for multiple parameter combinations.
    
    This function contains the parameter sweep logic and calls
    periodic_cos() for each parameter combination.
    """
    
    # Parameter combinations to test (quick test with T=2)
    T_values = [2]
    epsilon_values = [0.1]  # Just one value for quick test
    h0_values = [0.1]
    Nparticles_values = [5]  # Small number for speed
    
    print("Starting batch simulations...")
    print(f"Total combinations: {len(T_values) * len(epsilon_values) * len(h0_values) * len(Nparticles_values)}")
    
    completed_runs = []
    total_start_time = time.time()
    
    for T in T_values:
        for epsilon in epsilon_values:
            for h0 in h0_values:
                for Nparticles in Nparticles_values:
                    
                    print(f"\n--- Running simulation {len(completed_runs) + 1} ---")
                    print(f"Parameters: T={T}, epsilon={epsilon}, h0={h0}, Nparticles={Nparticles}")
                    
                    start_time = time.time()
                    
                    try:
                        result = periodic_cos(T, epsilon, h0, Nparticles)
                        
                        elapsed = time.time() - start_time
                        print(f"✅ Completed in {elapsed:.1f}s")
                        
                        completed_runs.append({
                            'params': result['params'],
                            'output_dir': result['output_dir'],
                            'runtime': elapsed,
                            'status': 'success'
                        })
                        
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"❌ Failed after {elapsed:.1f}s: {e}")
                        
                        completed_runs.append({
                            'params': {'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles},
                            'output_dir': None,
                            'runtime': elapsed,
                            'status': 'failed',
                            'error': str(e)
                        })
    
    total_elapsed = time.time() - total_start_time
    successful_runs = sum(1 for run in completed_runs if run['status'] == 'success')
    
    print(f"\n{'='*50}")
    print(f"Batch simulations completed!")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Successful runs: {successful_runs}/{len(completed_runs)}")
    print(f"{'='*50}")
    
    return completed_runs


if __name__ == "__main__":
    results = run_batch_simulations()
    
    # Print summary
    print("\nSummary of completed runs:")
    for i, run in enumerate(results):
        status_symbol = "✅" if run['status'] == 'success' else "❌"
        params = run['params']
        print(f"{i+1:2d}. {status_symbol} T={params['T']}, ε={params['epsilon']}, h0={params['h0']}, N={params['Nparticles']} ({run['runtime']:.1f}s)")