import numpy as np
import importlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate, weighted_empirical_measure_functional, empirical_measure_functional, resample_weighted_empirical_measure, fleming_viot_vanilla, cumulative_empirical_measure_functional, cumulative_mean_weighted_empirical_measure
from utils.plotting import plot_periodic_trajectories, plot_periodic_trajectories_list
import os
import json


T = 25
epsilon = 0.125 # 0.25
h0 = 0.1
Nparticles = 50



# def periodic_cos(T, epsilon, h0, Nparticles):
# periodic potential problem
def V(x): 
    return np.cos(2 * np.pi * x) / (2 * np.pi)

def DV(x):
    return -  np.sin(2 * np.pi * x) 

def D2V(x):
    return - 2 * np.pi * np.cos(2 * np.pi * x)

def a(x): 
    return  2 * epsilon * np.ones_like(x)

def c(x):
    return  1/( np.pi) * x ** 2

dim = 1


initial_positionsx = np.zeros((Nparticles,dim)) +1
initial_positionsy = np.zeros((Nparticles,dim)) +1



# Vanilla Fleming-Viot simulation

all_positions_vanilla, all_time_vanilla = fleming_viot_vanilla(T, lambda x: -DV(x), a, c, h0, initial_positionsx)



# infinite swap Fleming-Viot simulation
allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T,V,DV,D2V,a,2 * epsilon,c,h0,initial_positionsx,initial_positionsy)






# Plot all positions of the vanilla trajectories
plt.figure(figsize=(10, 6))
for i in range(Nparticles):
    plt.plot(all_time_vanilla, all_positions_vanilla[:, i, 0], alpha=0.7, label=f'Particle {i+1}' if i < 5 else None)
plt.ylim(-5, 5)  # Set the same y-axis range
plt.title('Ensemble trajectories of F-V system')
plt.xlabel('Time')
plt.ylabel('Position')
# plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
plt.show()

# Plot one representative trajectory for vanilla
plt.figure(figsize=(10, 6))
plt.plot(all_time_vanilla, all_positions_vanilla[:, 0, 0], color='red', label='Representative Trajectory (Particle 1)')
plt.ylim(-5, 5)  # Set the same y-axis range
plt.title('Representative Trajectory of F-V system')
plt.xlabel('Time')
plt.ylabel('Position')
# plt.legend()
plt.show()


# # plotting

# Plot all positions of the FV system
plt.figure(figsize=(10, 6))
for i in range(Nparticles):
    plt.plot(alltime, allpositionsx[:, i, 0], alpha=0.7, label=f'Particle {i+1}' if i < 5 else None)
plt.ylim(-5, 5)  # Set the same y-axis range
plt.title('Ensemble trajectories of INS system')
plt.xlabel('Time')
plt.ylabel('Position')
# plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
plt.show()

# Plot one representative trajectory
plt.figure(figsize=(10, 6))
plt.plot(alltime, allpositionsx[:, 0, 0], color='red', label='Representative Trajectory (Particle 1)')
plt.ylim(-5, 5)  # Set the same y-axis range
plt.title('Representative Trajectory of INS system')
plt.xlabel('Time')
plt.ylabel('Position')
# plt.legend()
plt.show()







#Eigenvalue computation


# Apply burn-in: only keep data where time > burn-in
burn_in = 10

# For vanilla FV
burnin_mask_vanilla = all_time_vanilla > burn_in
all_positions_vanilla = all_positions_vanilla[burnin_mask_vanilla]
all_time_vanilla = all_time_vanilla[burnin_mask_vanilla]

# For infinite swap FV
burnin_mask_ins = alltime > burn_in
allpositionsx = allpositionsx[burnin_mask_ins]
allpositionsy = allpositionsy[burnin_mask_ins]
alltime = alltime[burnin_mask_ins]
all_rho = all_rho[burnin_mask_ins]


def functional(x,y):
    return c(x)



# compute expectations vanilla
results_vanilla = []
for i in range(Nparticles):
    result = empirical_measure_functional(all_positions_vanilla[:, i, 0], c, all_time_vanilla)
    results_vanilla.append(result)



#compute expectations


results_IS_FV = []
for i in range(Nparticles):
    result = weighted_empirical_measure_functional(
        allpositionsx[:, i, 0], allpositionsy[:, i, 0], all_rho[:, i, 0], functional, alltime
    )
    results_IS_FV.append(result)
results_IS_FV = np.array(results_IS_FV)




# compute cumulative expectations
cumulative_results_vanilla = []
for i in range(Nparticles):
    result = cumulative_empirical_measure_functional(all_positions_vanilla[:, i, 0], c, all_time_vanilla)
    cumulative_results_vanilla.append(result)
cumulative_results_vanilla = np.array(cumulative_results_vanilla)

cumulative_results_IS_FV = []
for i in range(Nparticles):
    result = cumulative_mean_weighted_empirical_measure(allpositionsx[:, i, 0],allpositionsy[:,i,0], all_rho[:, i, 0], functional,alltime)
    cumulative_results_IS_FV.append(result)
cumulative_results_IS_FV = np.array(cumulative_results_IS_FV)


# Plot cumulative results 
# Plot cumulative results for vanilla
plt.figure(figsize=(10, 6))
for i in range(Nparticles):
    plt.plot(all_time_vanilla, cumulative_results_vanilla[i], alpha=0.7, label=f'Particle {i+1}' if i < 5 else None)
plt.ylim(0, 0.3)  # Set y-axis range from 0 to 0.6
plt.title('Cumulative eigenvalue estimate (simple FV)')
plt.xlabel('Time')
plt.ylabel('Cumulative Result')
# plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
plt.show()

# Plot cumulative results for infinite swap

plt.figure(figsize=(10, 6))
for i in range(Nparticles):
    plt.plot(alltime[alltime >= 10**0], cumulative_results_IS_FV[i][alltime >= 10**0], alpha=0.7, label=f'Particle {i+1}' if i < 5 else None)
plt.ylim(0, 0.3)  # Set y-axis range from 0 to 0.6

plt.title('Cumulative eigenvalue estimate (INS)')
plt.xlabel('Time')
plt.ylabel('Cumulative Result')

# plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
plt.show()



# Compute variance of cumulative results for vanilla
variance_cumulative_vanilla = np.var(cumulative_results_vanilla[:, all_time_vanilla >= 10**0], axis=0)

# Compute variance of cumulative results for infinite swap
variance_cumulative_IS_FV = np.var(cumulative_results_IS_FV[:, alltime >= 10**0], axis=0)

# Plot variance of cumulative results for vanilla and infinite swap on the same plot
plt.figure(figsize=(10, 6))
plt.plot(all_time_vanilla[all_time_vanilla >= 10**0], variance_cumulative_vanilla, label='Simple FV', color='blue')
plt.plot(alltime[alltime >= 10**0], variance_cumulative_IS_FV, label='Infinite Swap', color='orange')
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('log')  # Set y-axis to log scale
plt.title('Variance of Cumulative eigenvalue estimate')
plt.xlabel('Time (Log Scale)')
plt.ylabel('Variance (Log Scale)')
plt.legend()
plt.show()


# Plot cumulative mean-squared error (MSE) for vanilla and infinite swap
truth = 0.14259209

# Compute MSE for vanilla
mse_cumulative_vanilla = np.mean((cumulative_results_vanilla - truth) ** 2, axis=0)

# Compute MSE for infinite swap
mse_cumulative_IS_FV = np.mean((cumulative_results_IS_FV - truth) ** 2, axis=0)
# Plot MSE for both methods with more y-axis ticks and grid
plt.figure(figsize=(10, 6))
plt.plot(all_time_vanilla, mse_cumulative_vanilla, label='Simple FV', color='blue')
plt.plot(alltime, mse_cumulative_IS_FV, label='Infinite Swap', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.001, 0.1)  # Set y-axis range from 0.001 to 0.1
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title('Cumulative Mean-Squared Error')
plt.xlabel('Time (Log Scale)')
plt.ylabel('MSE (Log Scale)')
plt.legend()
plt.show()


# Plot cumulative squared bias for vanilla and infinite swap
squared_bias_vanilla = (np.mean(cumulative_results_vanilla, axis=0) - truth) ** 2
squared_bias_IS_FV = (np.mean(cumulative_results_IS_FV, axis=0) - truth) ** 2

plt.figure(figsize=(10, 6))
plt.plot(all_time_vanilla, squared_bias_vanilla, label='Simple FV', color='blue')
plt.plot(alltime, squared_bias_IS_FV, label='Infinite Swap', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.title('Cumulative Squared Bias vs Truth')
plt.xlabel('Time (Log Scale)')
plt.ylabel('Squared Bias (Log Scale)')
plt.legend()
plt.show()




# resampling from empirical measure for infinite swap
resampled_x = []
for i in range(Nparticles):
    resampled = resample_weighted_empirical_measure(allpositionsx[:, i, 0], all_rho[:, i, 0], alltime)
    resampled_x.append(resampled)
resampled_x = np.concatenate(resampled_x)

# Plot histogram of resampled_x
plt.hist(resampled_x, bins=100, density=True, alpha=0.7, color='blue', label='Resampled INS positions')
plt.title('Histogram of Resampled INS positions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.plot(x, f0, label="True eigenfunction", color='red', linewidth=2)
plt.legend(loc='upper left', fontsize='small')
plt.show()


# Plot histogram of all_positions_vanilla
flattened_positions_vanilla = all_positions_vanilla[:, :, 0].flatten()
plt.hist(flattened_positions_vanilla, bins=100, density=True, alpha=0.7, color='green', label='Vanilla FV positions')
plt.title('Histogram of Standard FV system')
plt.xlabel('Value')
plt.ylabel('Density')
plt.plot(x, f0, label="True eigenfunction", color='red', linewidth=2)
plt.legend(loc='upper left', fontsize='small')
plt.show()








# Save all relevant data to a file
output_data = {
    "all_positions_vanilla": all_positions_vanilla,
    "all_time_vanilla": all_time_vanilla,
    "allpositionsx": allpositionsx,
    "allpositionsy": allpositionsy,
    "alltime": alltime,
    "all_rho": all_rho,
    "results_vanilla": results_vanilla,
    "results_IS_FV": results_IS_FV,
    "resampled_x": resampled_x,
    "flattened_positions_vanilla": flattened_positions_vanilla
}

# Save to a .npy file
output_file = "eigenfunc_example.npy"
np.save(output_file, output_data)
print(f"Data saved to {output_file}")