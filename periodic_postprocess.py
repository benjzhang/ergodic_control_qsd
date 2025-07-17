import numpy as np
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate, weighted_empirical_measure_functional, empirical_measure_functional, resample_simple_weighted_empirical_measure,resample_weighted_empirical_measure, fleming_viot_vanilla, cumulative_empirical_measure_functional, cumulative_mean_weighted_empirical_measure



T_values = [100]
epsilon_values = [ 0.05, 0.1,0.15, 0.25, 0.5, 1.0]
h0_values = [0.1]
Nparticles_values = [5, 10, 20, 25]

T = 100
epsilon = 0.1
h0 = 0.1
Nparticles = 10


output_dir = f'periodic/T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}'

output_dir_path = Path(output_dir)
alltime_replicates = np.load(output_dir_path / 'alltime_replicates.npy', allow_pickle=True)
allpositionsx_replicates = np.load(output_dir_path / 'allpositionsx_replicates.npy', allow_pickle=True)
allpositionsy_replicates = np.load(output_dir_path / 'allpositionsy_replicates.npy', allow_pickle=True)
all_rho_replicates = np.load(output_dir_path / 'all_rho_replicates.npy', allow_pickle=True)
simple_trajectories_times = np.load(output_dir_path / 'simple_trajectories_times.npy', allow_pickle=True)
simple_trajectories_x = np.load(output_dir_path / 'simple_trajectories_x.npy', allow_pickle=True)



results_reference = []
for j in range(100):  # Loop over j from 0 to 9
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


def functional(x,y):
    return np.exp(-x/epsilon)

results_reference_weighted_allpositions = []
for j in range(100):  # Loop over j from 0 to 9
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







# plot histogram of simple_trajectories_x
# Flatten the first entry of simple_trajectories_x for histogram plotting
flattened_simple_trajectories_x = np.concatenate(simple_trajectories_x[0])

# resample from simple FV empirical measure
results_reference_weighted_allpositions_resampled = []
for j in range(1):  # Loop over j from 0 to 9
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


# Flatten results_reference_weighted_allpositions for histogram plotting
# Ensure results_reference_weighted_allpositions contains arrays with at least one dimension
flattened_simple_trajectories_x = np.array(results_reference_weighted_allpositions_resampled)[0]


# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(flattened_simple_trajectories_x, bins=60, color='green', alpha=0.7, edgecolor='black', density=True)
plt.title('Histogram of Simple Trajectories X (First Entry)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# resample_weighted_empirical_measure
results_INS_weighted_allpositions_resampled = []
for j in range(2):  # Loop over j from 0 to 9
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



# Flatten the resampled results for histogram plotting
flattened_INS_resampled_results = [item for sublist in results_INS_weighted_allpositions_resampled for item in sublist]

# Plot histogram and density on the same graph
plt.figure(figsize=(10, 6))

# Plot histogram
plt.hist(flattened_INS_resampled_results, bins=60, color='blue', alpha=0.7, edgecolor='black', density=True, label='Histogram')

# Define the potential function V(x)
def V(x):
    return np.cos(2 * np.pi * x) / (2 * np.pi)

# Define the range for x
x_values = np.linspace(-1, 1, 500)

# Compute exp(-V(x)/epsilon) for the given range of x
y_values = np.exp(-V(x_values) / epsilon) / 2.3294  # Normalize by 2.2078

# Plot the density
plt.plot(x_values, y_values, color='red', label=r'$e^{-V(x)/\epsilon}$')

# Add labels, title, and legend
plt.title('Histogram and Density of Resampled Weighted Empirical Measure Results')
plt.xlabel('Value')
plt.ylabel('Density')
plt.ylim(0, None)  # Set the minimum y-limit to 0
plt.legend()
plt.grid(True)
plt.show()


np.mean(np.exp(-np.array(flattened_INS_resampled_results)/epsilon))

np.mean(np.exp(-np.array(flattened_simple_trajectories_x)/epsilon))