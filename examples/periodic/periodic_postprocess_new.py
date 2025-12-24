"""
Postprocessing analysis for periodic potential simulation results.

This file provides modular postprocessing functions for empirical measure
analysis, resampling, and statistical computations.
"""

import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
from utils import (weighted_empirical_measure_functional, empirical_measure_functional, 
                   resample_simple_weighted_empirical_measure, resample_weighted_empirical_measure,
                   cumulative_empirical_measure_functional, cumulative_mean_weighted_empirical_measure)


def load_simulation_data(output_dir):
    """
    Load simulation data from directory.
    
    Parameters:
    - output_dir: path to simulation data directory
    
    Returns:
    - Dictionary containing loaded arrays and metadata
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        raise FileNotFoundError(f"Simulation data not found in {output_dir}")
    
    # Load simulation arrays
    data = {}
    
    try:
        data['alltime_replicates'] = np.load(output_dir_path / 'alltime_replicates.npy', allow_pickle=True)
        data['allpositionsx_replicates'] = np.load(output_dir_path / 'allpositionsx_replicates.npy', allow_pickle=True)
        data['allpositionsy_replicates'] = np.load(output_dir_path / 'allpositionsy_replicates.npy', allow_pickle=True)
        data['all_rho_replicates'] = np.load(output_dir_path / 'all_rho_replicates.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"⚠️  Fleming-Viot data files missing: {e}")
        data.update({'alltime_replicates': None, 'allpositionsx_replicates': None, 
                    'allpositionsy_replicates': None, 'all_rho_replicates': None})
    
    try:
        data['simple_trajectories_times'] = np.load(output_dir_path / 'simple_trajectories_times.npy', allow_pickle=True)
        data['simple_trajectories_x'] = np.load(output_dir_path / 'simple_trajectories_x.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"⚠️  Vanilla simulation data files missing: {e}")
        data.update({'simple_trajectories_times': None, 'simple_trajectories_x': None})
    
    # Determine dimensions
    n_replicates = 0
    n_particles = 0
    
    if data['alltime_replicates'] is not None:
        n_replicates = len(data['alltime_replicates'])
        if len(data['allpositionsx_replicates']) > 0:
            n_particles = data['allpositionsx_replicates'][0].shape[1]
    elif data['simple_trajectories_times'] is not None:
        n_replicates = len(data['simple_trajectories_times'])
        n_particles = len(data['simple_trajectories_times'][0])
    
    data['n_replicates'] = n_replicates
    data['n_particles'] = n_particles
    
    print(f"📊 Loaded data: {n_replicates} replicates, {n_particles} particles")
    return data


def analyze_vanilla_empirical_measures(data, epsilon, burn_in_time=10):
    """
    Analyze empirical measures from vanilla (reference) simulations.
    
    Parameters:
    - data: loaded simulation data
    - epsilon: temperature parameter
    - burn_in_time: time threshold for burn-in
    
    Returns:
    - Dictionary with vanilla analysis results
    """
    if data['simple_trajectories_times'] is None:
        print("⚠️  No vanilla simulation data available")
        return {'results_reference': np.array([]), 'available': False}
    
    print(f"🔄 Analyzing vanilla empirical measures (burn-in: T≥{burn_in_time})")
    
    results_reference = []
    
    for j in range(data['n_replicates']):
        replicate_results = []
        
        for i in range(data['n_particles']):
            try:
                # Apply burn-in filter
                times = data['simple_trajectories_times'][j][i]
                positions = data['simple_trajectories_x'][j][i]
                
                valid_indices = times >= burn_in_time
                
                if np.sum(valid_indices) == 0:
                    print(f"⚠️  No data after burn-in for replicate {j}, particle {i}")
                    continue
                
                # Compute empirical measure functional
                result = empirical_measure_functional(
                    positions[valid_indices], 
                    lambda x: np.exp(-x/epsilon), 
                    times[valid_indices]
                )
                
                if not np.isnan(result):
                    replicate_results.append(result)
                    
            except Exception as e:
                print(f"⚠️  Error processing replicate {j}, particle {i}: {e}")
                continue
        
        if replicate_results:
            results_reference.append(np.mean(replicate_results))
    
    results_reference = np.array(results_reference)
    
    print(f"✅ Vanilla analysis: {len(results_reference)} valid replicate means")
    
    return {
        'results_reference': results_reference,
        'available': len(results_reference) > 0
    }


def analyze_weighted_empirical_measures(data, epsilon, burn_in_time=10):
    """
    Analyze weighted empirical measures from Fleming-Viot simulations.
    
    Parameters:
    - data: loaded simulation data
    - epsilon: temperature parameter  
    - burn_in_time: time threshold for burn-in
    
    Returns:
    - Dictionary with weighted analysis results
    """
    if data['alltime_replicates'] is None:
        print("⚠️  No Fleming-Viot simulation data available")
        return {'results_weighted': np.array([]), 'available': False}
    
    print(f"🔄 Analyzing weighted empirical measures (burn-in: T≥{burn_in_time})")
    
    # Define functional for weighted analysis
    def functional(x, y):
        return np.exp(-x/epsilon)
    
    results_weighted = []
    
    for j in range(data['n_replicates']):
        replicate_results = []
        
        try:
            # Apply burn-in filter to time array
            times = data['alltime_replicates'][j]
            valid_indices = times >= burn_in_time
            
            if np.sum(valid_indices) == 0:
                print(f"⚠️  No data after burn-in for replicate {j}")
                continue
            
            for i in range(data['n_particles']):
                try:
                    # Extract particle data with burn-in
                    x_pos = data['allpositionsx_replicates'][j][valid_indices, i, 0]
                    y_pos = data['allpositionsy_replicates'][j][valid_indices, i, 0] 
                    rho_vals = data['all_rho_replicates'][j][valid_indices, i, 0]
                    times_filtered = times[valid_indices]
                    
                    # Compute weighted empirical measure functional
                    result = weighted_empirical_measure_functional(
                        x_pos, y_pos, rho_vals, functional, times_filtered
                    )
                    
                    if not np.isnan(result):
                        replicate_results.append(result)
                        
                except Exception as e:
                    print(f"⚠️  Error processing replicate {j}, particle {i}: {e}")
                    continue
            
            if replicate_results:
                results_weighted.append(np.mean(replicate_results))
                
        except Exception as e:
            print(f"⚠️  Error processing replicate {j}: {e}")
            continue
    
    results_weighted = np.array(results_weighted)
    
    print(f"✅ Weighted analysis: {len(results_weighted)} valid replicate means")
    
    return {
        'results_weighted': results_weighted,
        'available': len(results_weighted) > 0
    }


def analyze_resampling(data, epsilon, burn_in_time=10, max_replicates=None):
    """
    Perform resampling analysis on both vanilla and weighted empirical measures.
    
    Parameters:
    - data: loaded simulation data
    - epsilon: temperature parameter
    - burn_in_time: time threshold for burn-in  
    - max_replicates: limit number of replicates to process (None for all)
    
    Returns:
    - Dictionary with resampling results
    """
    results = {
        'vanilla_resampled': [],
        'weighted_resampled': [],
        'vanilla_available': False,
        'weighted_available': False
    }
    
    # Determine how many replicates to process
    n_process = data['n_replicates']
    if max_replicates is not None:
        n_process = min(n_process, max_replicates)
    
    print(f"🔄 Performing resampling analysis (processing {n_process} replicates)")
    
    # Vanilla resampling
    if data['simple_trajectories_times'] is not None:
        print("  📍 Vanilla resampling...")
        
        for j in range(min(n_process, len(data['simple_trajectories_times']))):
            replicate_samples = []
            
            for i in range(min(data['n_particles'], len(data['simple_trajectories_times'][j]))):
                try:
                    times = data['simple_trajectories_times'][j][i]
                    positions = data['simple_trajectories_x'][j][i]
                    
                    valid_indices = times >= burn_in_time
                    
                    if np.sum(valid_indices) < 2:  # Need at least 2 points for resampling
                        continue
                    
                    samples = resample_simple_weighted_empirical_measure(
                        positions[valid_indices], 
                        times[valid_indices]
                    )
                    
                    replicate_samples.extend(samples.flatten())
                    
                except Exception as e:
                    print(f"⚠️  Vanilla resampling error (rep {j}, particle {i}): {e}")
                    continue
            
            if replicate_samples:
                results['vanilla_resampled'].extend(replicate_samples)
        
        results['vanilla_available'] = len(results['vanilla_resampled']) > 0
    
    # Weighted resampling
    if data['alltime_replicates'] is not None:
        print("  📍 Weighted resampling...")
        
        for j in range(min(n_process, len(data['alltime_replicates']))):
            replicate_samples = []
            
            try:
                times = data['alltime_replicates'][j]
                valid_indices = times >= burn_in_time
                
                if np.sum(valid_indices) < 2:
                    continue
                
                for i in range(data['n_particles']):
                    try:
                        x_pos = data['allpositionsx_replicates'][j][valid_indices, i, 0]
                        rho_vals = data['all_rho_replicates'][j][valid_indices, i, 0]
                        times_filtered = times[valid_indices]
                        
                        samples = resample_weighted_empirical_measure(
                            x_pos, rho_vals, times_filtered
                        )
                        
                        replicate_samples.extend(samples.flatten())
                        
                    except Exception as e:
                        print(f"⚠️  Weighted resampling error (rep {j}, particle {i}): {e}")
                        continue
                
                if replicate_samples:
                    results['weighted_resampled'].extend(replicate_samples)
                    
            except Exception as e:
                print(f"⚠️  Error processing weighted replicate {j}: {e}")
                continue
        
        results['weighted_available'] = len(results['weighted_resampled']) > 0
    
    print(f"✅ Resampling: {len(results['vanilla_resampled'])} vanilla, {len(results['weighted_resampled'])} weighted samples")
    
    return results


def compute_summary_statistics(vanilla_results, weighted_results, resampling_results, epsilon):
    """
    Compute summary statistics from all analysis results.
    
    Parameters:
    - vanilla_results: results from vanilla analysis
    - weighted_results: results from weighted analysis  
    - resampling_results: results from resampling analysis
    - epsilon: temperature parameter
    
    Returns:
    - Dictionary with summary statistics
    """
    stats = {}
    
    # Empirical measure statistics
    if vanilla_results['available']:
        stats['vanilla_mean'] = np.mean(vanilla_results['results_reference'])
        stats['vanilla_std'] = np.std(vanilla_results['results_reference'])
    else:
        stats['vanilla_mean'] = stats['vanilla_std'] = np.nan
    
    if weighted_results['available']:
        stats['weighted_mean'] = np.mean(weighted_results['results_weighted'])
        stats['weighted_std'] = np.std(weighted_results['results_weighted'])
    else:
        stats['weighted_mean'] = stats['weighted_std'] = np.nan
    
    # Resampling statistics
    if resampling_results['vanilla_available']:
        vanilla_samples = np.array(resampling_results['vanilla_resampled'])
        stats['vanilla_resampled_exp_mean'] = np.mean(np.exp(-vanilla_samples/epsilon))
    else:
        stats['vanilla_resampled_exp_mean'] = np.nan
    
    if resampling_results['weighted_available']:
        weighted_samples = np.array(resampling_results['weighted_resampled'])
        stats['weighted_resampled_exp_mean'] = np.mean(np.exp(-weighted_samples/epsilon))
    else:
        stats['weighted_resampled_exp_mean'] = np.nan
    
    return stats


def process_simulation_data(T, epsilon, h0, Nparticles, burn_in_time=10, max_replicates=None):
    """
    Complete postprocessing pipeline for simulation data.
    
    Parameters:
    - T, epsilon, h0, Nparticles: simulation parameters
    - burn_in_time: time threshold for burn-in (default: 10)
    - max_replicates: limit number of replicates to process (None for all)
    
    Returns:
    - Dictionary containing all analysis results and statistics
    """
    # Format directory name consistently (avoid .0 for integers)
    T_str = str(int(T)) if T == int(T) else str(T)
    epsilon_str = str(epsilon) 
    h0_str = str(h0)
    Nparticles_str = str(int(Nparticles))
    
    output_dir = f'periodic/T_{T_str}_epsilon_{epsilon_str}_h0_{h0_str}_Nparticles_{Nparticles_str}'
    
    print(f"🔍 Processing data for T={T}, ε={epsilon}, h0={h0}, N={Nparticles}")
    print(f"📁 Data directory: {output_dir}")
    
    # Load simulation data
    data = load_simulation_data(output_dir)
    
    # Perform analyses
    vanilla_results = analyze_vanilla_empirical_measures(data, epsilon, burn_in_time)
    weighted_results = analyze_weighted_empirical_measures(data, epsilon, burn_in_time)
    resampling_results = analyze_resampling(data, epsilon, burn_in_time, max_replicates)
    
    # Compute summary statistics
    stats = compute_summary_statistics(vanilla_results, weighted_results, resampling_results, epsilon)
    
    # Print results
    print(f"\n📊 ANALYSIS RESULTS:")
    if not np.isnan(stats['vanilla_mean']):
        print(f"   Vanilla empirical mean: {stats['vanilla_mean']:.6f} ± {stats['vanilla_std']:.6f}")
    if not np.isnan(stats['weighted_mean']):
        print(f"   Weighted empirical mean: {stats['weighted_mean']:.6f} ± {stats['weighted_std']:.6f}")
    if not np.isnan(stats['vanilla_resampled_exp_mean']):
        print(f"   Vanilla resampled exp mean: {stats['vanilla_resampled_exp_mean']:.6f}")
    if not np.isnan(stats['weighted_resampled_exp_mean']):
        print(f"   Weighted resampled exp mean: {stats['weighted_resampled_exp_mean']:.6f}")
    
    return {
        'output_dir': output_dir,
        'parameters': {'T': T, 'epsilon': epsilon, 'h0': h0, 'Nparticles': Nparticles},
        'vanilla_results': vanilla_results,
        'weighted_results': weighted_results,
        'resampling_results': resampling_results,
        'summary_stats': stats,
        'data_info': {'n_replicates': data['n_replicates'], 'n_particles': data['n_particles']}
    }


def run_postprocessing_batch(burn_in_time=10, max_replicates=None):
    """
    Run postprocessing for all available simulation data directories.
    
    Parameters:
    - burn_in_time: time threshold for burn-in (default: 10)
    - max_replicates: limit number of replicates to process per dataset (None for all)
    
    Returns:
    - List of processing results for each dataset
    """
    print("🚀 Starting batch postprocessing...")
    
    # Find all simulation directories
    periodic_dirs = []
    if os.path.exists('periodic'):
        for item in os.listdir('periodic'):
            if item.startswith('T_') and os.path.isdir(f'periodic/{item}'):
                periodic_dirs.append(item)
    
    if not periodic_dirs:
        print("❌ No simulation data found to process!")
        return []
    
    print(f"📂 Found {len(periodic_dirs)} simulation datasets")
    
    results = []
    for i, dir_name in enumerate(periodic_dirs, 1):
        try:
            print(f"\n{'='*60}")
            print(f"Processing dataset {i}/{len(periodic_dirs)}: {dir_name}")
            print(f"{'='*60}")
            
            # Parse parameters from directory name
            parts = dir_name.split('_')
            T = float(parts[1])
            epsilon = float(parts[3])
            h0 = float(parts[5])
            Nparticles = int(parts[7])
            
            # Process the dataset
            result = process_simulation_data(T, epsilon, h0, Nparticles, burn_in_time, max_replicates)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to process {dir_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Batch postprocessing completed! Processed {len(results)}/{len(periodic_dirs)} datasets")
    
    return results


if __name__ == "__main__":
    # Run postprocessing for all available data
    results = run_postprocessing_batch(burn_in_time=1, max_replicates=3)  # Reduced for testing
    
    if results:
        print(f"\n📋 SUMMARY ({len(results)} datasets processed):")
        print("-" * 80)
        for result in results:
            stats = result['summary_stats']
            info = result['data_info']
            params = result['parameters']
            
            print(f"T={params['T']}, ε={params['epsilon']}, h0={params['h0']}, N={params['Nparticles']} ({info['n_replicates']} reps)")
            if not np.isnan(stats['vanilla_mean']):
                print(f"  Vanilla: {stats['vanilla_mean']:.4f} ± {stats['vanilla_std']:.4f}")
            if not np.isnan(stats['weighted_mean']):
                print(f"  Weighted: {stats['weighted_mean']:.4f} ± {stats['weighted_std']:.4f}")
            print()
    else:
        print("❌ No data processed. Run simulations first!")