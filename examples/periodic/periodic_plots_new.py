"""
Comprehensive plotting for periodic potential simulations.

This file combines all plotting functionality: trajectory plots, 
effective potential plots, histograms, and movie generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.plotting import plot_periodic_trajectories, plot_periodic_trajectories_list
from pathlib import Path


def plot_trajectory_analysis(output_dir):
    """
    Create trajectory plots for a simulation dataset.
    
    This function creates the trajectory plots that were originally
    in main_periodic.py.
    """
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        print(f"❌ Data directory {output_dir} not found")
        return False
    
    # Load data
    alltime_replicates = np.load(output_dir_path / 'alltime_replicates.npy', allow_pickle=True)
    allpositionsx_replicates = np.load(output_dir_path / 'allpositionsx_replicates.npy', allow_pickle=True)
    allpositionsy_replicates = np.load(output_dir_path / 'allpositionsy_replicates.npy', allow_pickle=True)
    simple_trajectories_times = np.load(output_dir_path / 'simple_trajectories_times.npy', allow_pickle=True)
    simple_trajectories_x = np.load(output_dir_path / 'simple_trajectories_x.npy', allow_pickle=True)
    
    # Load parameters
    with open(output_dir_path / 'parameters.json', 'r') as f:
        import json
        params = json.load(f)
    
    T = params['T']
    epsilon = params['epsilon']
    h0 = params['h0']
    Nparticles = params['Nparticles']
    
    print(f"Creating trajectory plots for T={T}, epsilon={epsilon}, h0={h0}, Nparticles={Nparticles}")
    
    # Create figures directory
    figures_dir = output_dir_path / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Plot 1: All Positions X[0] (First Replicate)
    fig1 = plot_periodic_trajectories(alltime_replicates[0], allpositionsx_replicates[0][:, 0, 0])
    fig1.suptitle(f'All Positions X[0] (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig1.savefig(figures_dir / 'allpositionsx_0_first_replicate_plot.png')
    
    # Plot 2: All Positions Y[0] (First Replicate)
    fig2 = plot_periodic_trajectories(alltime_replicates[0], allpositionsy_replicates[0][:, 0, 0])
    fig2.suptitle(f'All Positions Y[0] (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig2.savefig(figures_dir / 'allpositionsy_0_first_replicate_plot.png')
    
    # Plot 3: All Positions X (First Replicate) 
    fig3 = plot_periodic_trajectories(alltime_replicates[0], allpositionsx_replicates[0])
    fig3.suptitle(f'All Positions X (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig3.savefig(figures_dir / 'allpositionsx_first_replicate_plot.png')
    
    # Plot 4: All Positions Y (First Replicate)
    fig4 = plot_periodic_trajectories(alltime_replicates[0], allpositionsy_replicates[0])
    fig4.suptitle(f'All Positions Y (First Replicate): T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig4.savefig(figures_dir / 'allpositionsy_first_replicate_plot.png')
    
    # Plot simple trajectories
    simple_times_flat = [time for replicate in simple_trajectories_times[0] for time in replicate]
    simple_x_flat = [x for replicate in simple_trajectories_x[0] for x in replicate]
    
    fig5 = plot_periodic_trajectories_list(simple_times_flat[:5], simple_x_flat[:5])  # Plot first 5
    fig5.suptitle(f'Simple Trajectories: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')
    fig5.savefig(figures_dir / 'simple_trajectories_plot.png')
    
    # Close all figures
    for fig in [fig1, fig2, fig3, fig4, fig5]:
        plt.close(fig)
    
    print(f"✅ Trajectory plots saved to {figures_dir}/")
    return True


def plot_effective_potential(output_dir):
    """
    Create effective potential plots with particle overlays.
    
    This function creates the effective potential plots that were
    originally in periodic_movies_plots.py.
    """
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        print(f"❌ Data directory {output_dir} not found")
        return False
    
    # Load data
    alltime_replicates = np.load(output_dir_path / 'alltime_replicates.npy', allow_pickle=True)
    allpositionsx_replicates = np.load(output_dir_path / 'allpositionsx_replicates.npy', allow_pickle=True)
    allpositionsy_replicates = np.load(output_dir_path / 'allpositionsy_replicates.npy', allow_pickle=True)
    
    # Load parameters
    with open(output_dir_path / 'parameters.json', 'r') as f:
        import json
        params = json.load(f)
    
    epsilon = params['epsilon']
    Nparticles = params['Nparticles']
    
    print(f"Creating effective potential plot for epsilon={epsilon}, Nparticles={Nparticles}")
    
    # Get first replicate data
    allpositionsx = allpositionsx_replicates[0]
    allpositionsy = allpositionsy_replicates[0] 
    
    def V(x): 
        return np.cos(2 * np.pi * x) / (2 * np.pi)

    def effective_potential(x, y):
        return - epsilon * np.log(np.exp(-2 * V(x) / epsilon) + np.exp(-2 * V(y) / epsilon))

    # Generate a grid of points in the range [-1, 1] x [-1, 1]
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Compute the effective potential on the grid
    Z = effective_potential(X, Y)

    # Plot the effective potential
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Effective Potential')

    # Overlay particle locations on the effective potential
    for idx in range(min(5, Nparticles)):  # Plot up to 5 particles
        plt.scatter(allpositionsx[:, idx, :], allpositionsy[:, idx, :], 
                   alpha=0.7, color='red', s=1, label=f'Particle {idx}' if idx == 0 else "")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Coupled particle locations on Effective Potential (epsilon={epsilon}, N={Nparticles})')
    plt.grid(True)
    if Nparticles >= 1:
        plt.legend()

    # Save the plot
    figures_dir = output_dir_path / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plot_filename = figures_dir / f'effective_potential_epsilon_{epsilon}_Nparticles_{Nparticles}.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"✅ Effective potential plot saved to {plot_filename}")
    return True


def plot_postprocessing_analysis(postprocessing_results):
    """
    Create histograms and analysis plots from postprocessing results.
    
    This function creates the histogram plots that were originally
    in periodic_postprocess.py.
    """
    
    if not postprocessing_results:
        print("❌ No postprocessing results provided")
        return False
    
    output_dir = postprocessing_results['output_dir']
    flattened_simple_trajectories_x = postprocessing_results['flattened_simple_trajectories_x']
    flattened_INS_resampled_results = postprocessing_results['flattened_INS_resampled_results']
    epsilon = postprocessing_results['epsilon']
    
    output_dir_path = Path(output_dir)
    figures_dir = output_dir_path / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print(f"Creating postprocessing plots for epsilon={epsilon}")
    
    # Plot 1: Histogram of Simple Trajectories
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_simple_trajectories_x, bins=60, color='green', alpha=0.7, edgecolor='black', density=True)
    plt.title('Histogram of Simple Trajectories X (First Entry)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(figures_dir / 'histogram_simple_trajectories.png', dpi=300)
    plt.close()

    # Plot 2: Histogram and density comparison
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(flattened_INS_resampled_results, bins=60, color='blue', alpha=0.7, 
            edgecolor='black', density=True, label='Histogram')
    
    # Define the potential function V(x)
    def V(x):
        return np.cos(2 * np.pi * x) / (2 * np.pi)
    
    # Define the range for x
    x_values = np.linspace(-1, 1, 500)
    
    # Compute exp(-V(x)/epsilon) for the given range of x
    y_values = np.exp(-V(x_values) / epsilon) / 2.3294  # Normalize by 2.3294
    
    # Plot the density
    plt.plot(x_values, y_values, color='red', label=r'$e^{-V(x)/\epsilon}$')
    
    # Add labels, title, and legend
    plt.title('Histogram and Density of Resampled Weighted Empirical Measure Results')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.ylim(0, None)  # Set the minimum y-limit to 0
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_dir / 'histogram_density_comparison.png', dpi=300)
    plt.close()
    
    print(f"✅ Postprocessing plots saved to {figures_dir}/")
    return True


def create_all_plots(output_dir, postprocessing_results=None):
    """
    Create all plots for a simulation dataset.
    
    Parameters:
    - output_dir: path to simulation data
    - postprocessing_results: results from postprocessing (optional)
    """
    
    print(f"Creating all plots for {output_dir}")
    
    success_count = 0
    
    # Create trajectory plots
    if plot_trajectory_analysis(output_dir):
        success_count += 1
    
    # Create effective potential plots
    if plot_effective_potential(output_dir):
        success_count += 1
    
    # Create postprocessing plots if results provided
    if postprocessing_results and plot_postprocessing_analysis(postprocessing_results):
        success_count += 1
    
    print(f"✅ Plot creation completed! {success_count} plot types created")
    return success_count > 0


def run_plotting_batch():
    """
    Run plotting for all available simulation data.
    """
    
    # Look for all simulation directories
    periodic_dirs = []
    if os.path.exists('periodic'):
        for item in os.listdir('periodic'):
            if item.startswith('T_') and os.path.isdir(f'periodic/{item}'):
                periodic_dirs.append(f'periodic/{item}')
    
    if not periodic_dirs:
        print("No simulation data found to plot!")
        return False
    
    print(f"Found {len(periodic_dirs)} simulation datasets to plot")
    
    success_count = 0
    for output_dir in periodic_dirs:
        try:
            print(f"\n--- Creating plots for {output_dir} ---")
            if create_all_plots(output_dir):
                success_count += 1
        except Exception as e:
            print(f"❌ Failed to create plots for {output_dir}: {e}")
    
    print(f"\n✅ Plotting batch completed! Created plots for {success_count}/{len(periodic_dirs)} datasets")
    return success_count > 0


if __name__ == "__main__":
    # Run plotting for all available data
    run_plotting_batch()