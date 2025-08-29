import numpy as np
import importlib
import matplotlib.pyplot as plt
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate, weighted_empirical_measure_functional, empirical_measure_functional
import plot_utils
importlib.reload(plot_utils)
from plot_utils import plot_periodic_trajectories, plot_periodic_trajectories_list
import os
import json
from scipy.special import logsumexp


T = 100
epsilon = 0.4
h0 = 0.1
Nparticles = 1



# def periodic_cos(T, epsilon, h0, Nparticles):
# periodic potential problem

sigma = 0.1
dim = 2
left = 0
right = 4




def V(x):
    centers = np.array([(i,j) for i in range(5) for j in range(5)])
    
    # Wrap around a torus
    x_wrapped = np.mod(x - right, right - left) + left  # Wrap x[0] and x[1] to be between -2 and 6
    
    distances = -np.sum((x_wrapped[..., None, :] - centers) ** 2, axis=-1) / (2 * sigma ** 2)
    return -logsumexp(distances, axis=-1)  # Use logsumexp for numerical stability






def DV(x):
    centers = np.array([(i, j) for i in range(5) for j in range(5)])
    
    # Wrap around a torus
    x_wrapped = np.mod(x - right, right - left) + left  # Wrap x[0] and x[1] to be between -2 and 6
    
    distances = -np.sum((x_wrapped[..., None, :] - centers) ** 2, axis=-1) / (2 * sigma ** 2)
    weights = np.exp(distances - logsumexp(distances, axis=-1, keepdims=True))
    
    # Compute the gradient of V (DV)
    gradient = - np.sum(weights[..., None] * (centers - x_wrapped[..., None, :]) / (sigma ** 2), axis=-2)
    return gradient


def D2V(x):
    centers = np.array([(i, j) for i in range(5) for j in range(5)])
    
    # Wrap around a torus
    x_wrapped = np.mod(x - right, right - left) + left  # Wrap x[0] and x[1] to be between -2 and 6
    
    distances = -np.sum((x_wrapped[..., None, :] - centers) ** 2, axis=-1) / (2 * sigma ** 2)
    weights = np.exp(distances - logsumexp(distances, axis=-1, keepdims=True))
    
    # Compute the Laplacian of V (D2V)
    # gradient_terms = (centers - x_wrapped[..., None, :]) / (sigma ** 2)
    laplacian = -np.sum(weights * (
        -2 * distances / (sigma ** 2) -
        dim / (sigma ** 2)
    ), axis=-1) + np.sum(DV(x) ** 2 , axis = -1)
    return laplacian

def a(x): 
    return  2 * epsilon * np.ones_like(x)

def c(x):
    return 0





# Generate a 2D grid for plotting
x = np.linspace(left, right, 100)
y = np.linspace(left, right, 100)
X, Y = np.meshgrid(x, y)
grid_points = np.stack([X, Y], axis=-1)

# Compute the potential on the grid
Z = V(grid_points)

# Plot the 2D contour
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')
plt.title('2D Contour Plot of Potential V(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

dim = 2


# Compute the Laplacian of the potential on the grid
D2V_values = D2V(grid_points)

# Plot the 2D contour for D2V
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, -D2V_values, levels=50, cmap='coolwarm', vmin=-D2V_values.max(), vmax=D2V_values.max())
plt.colorbar(contour, label='Neg Laplacian Value (D2V)')
plt.title('Backward kill rate')
plt.xlabel('x')
plt.ylabel('y')
plt.show()




initial_positionsx = np.zeros(( Nparticles, dim)) +2
initial_positionsy = np.zeros((Nparticles,dim)) +2

# Reference simulations
simple_trajectories_times = []
simple_trajectories_x = []

for i in range(2 * Nparticles):
    traj_x, times = pure_jump_approx_diffusion(T, lambda x: -DV(x), a, h0, initial_positionsx[0])
    simple_trajectories_times.append(times)
    simple_trajectories_x.append(traj_x)



simple_trajectories_x = np.mod(np.array(simple_trajectories_x) - left, right - left) + left

# Plot the trajectories on the same plot as the contour
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

for traj in simple_trajectories_x:
    traj = np.array(traj)
    plt.scatter(traj[:, 0], traj[:, 1], s=1, label='Trajectory')

plt.title('Vanilla Trajectories')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

plt.scatter(simple_trajectories_x[0][:, 0], simple_trajectories_x[0][:, 1], s=1, label='Trajectory',color='red')

plt.title('Vanilla Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# fig0 = plot_periodic_trajectories_list(simple_trajectories_times, simple_trajectories_x)
# fig0.suptitle(f'Simple Trajectories: T={T}, epsilon={epsilon}, Nparticles={Nparticles}, h0={h0}')





# Fleming-Viot simulation
allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T,V,DV,D2V,a,epsilon,c,h0,initial_positionsx,initial_positionsy)

allpositionsx = np.mod(allpositionsx - left, right - left) + left
allpositionsy = np.mod(allpositionsy - left, right - left) + left

# Ensure the output directory exists
output_dir = "/Users/bjzhang/gitrepos/ergodic_control_qsd/modular/aog"
os.makedirs(output_dir, exist_ok=True)

# Save outputs, parameters, and potential plot values to a .npy file
output_data = {
    "allpositionsx": allpositionsx,
    "allpositionsy": allpositionsy,
    "alltime": alltime,
    "all_rho": all_rho,
    "simple_trajectories_times": simple_trajectories_times,
    "simple_trajectories_x": simple_trajectories_x,
    "parameters": {
        "T": T,
        "epsilon": epsilon,
        "h0": h0,
        "Nparticles": Nparticles,
        "sigma": sigma,
        "dim": dim,
        "left": left,
        "right": right
    },
    "potential_plot_values": {
        "X": X,
        "Y": Y,
        "Z": Z,
        "D2V_values": D2V_values
    }
}

output_file = os.path.join(output_dir, "aog_single_pair.npy")
np.save(output_file, output_data)
print(f"Outputs, parameters, and potential plot values saved to {output_file}")