import numpy as np
import os
import matplotlib.pyplot as plt

epsilon = 0.2
T = 100
h0 = 0.1
Nparticles = 5
output_dir = f'periodic/T_{T}_epsilon_{epsilon}_h0_{h0}_Nparticles_{Nparticles}'
os.makedirs(output_dir, exist_ok=True)

alltime_replicates = np.load(f'{output_dir}/alltime_replicates.npy', allow_pickle=True)
allpositionsx_replicates = np.load(f'{output_dir}/allpositionsx_replicates.npy', allow_pickle=True)
allpositionsy_replicates = np.load(f'{output_dir}/allpositionsy_replicates.npy', allow_pickle=True)
all_rho_replicates = np.load(f'{output_dir}/all_rho_replicates.npy', allow_pickle=True)
simple_trajectories_times = np.load(f'{output_dir}/simple_trajectories_times.npy', allow_pickle=True)
simple_trajectories_x = np.load(f'{output_dir}/simple_trajectories_x.npy', allow_pickle=True)


allpositionsx = allpositionsx_replicates[0]
allpositionsy = allpositionsy_replicates[0] 
alltime = alltime_replicates[0]
all_rho = all_rho_replicates[0]
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
for idx in range(5):  # Loop over particle indices 0 to 4
    plt.scatter(allpositionsx[:, idx, :], allpositionsy[:, idx, :], alpha=0.7, color='red', s=1, label=f'Particle {idx}')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Coupled particle locations on Effective Potential (epsilon={epsilon}, N={Nparticles})')
plt.grid(True)

# Save the plot
plot_filename = f'{output_dir}/effective_potential_epsilon_{epsilon}_Nparticles_{Nparticles}.png'
plt.savefig(plot_filename, dpi=300)
plt.close()


# import matplotlib.animation as animation

# # Generate a sequence of figures for particle locations at time points with increments of 0.05
# T = 100
# time_points = np.arange(0, T+0.1, 0.1)
# time_indices = [np.max(np.where(alltime <= t)[0]) for t in time_points]

# fig, ax = plt.subplots(figsize=(8, 6))
# def update(frame):
#     ax.clear()
#     ax.contourf(X, Y, Z, levels=50, cmap='viridis')
#     ax.scatter(allpositionsx[time_indices[frame], :, :].flatten(), 
#                allpositionsy[time_indices[frame], :, :].flatten(), 
#                alpha=0.7, color='red', s=20, label='Particles')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title(f'Particle Locations on Effective Potential at t={time_points[frame]:.3f}')
#     ax.grid(True)

# ani = animation.FuncAnimation(fig, update, frames=len(time_indices), interval=50)

# # Save the animation as a movie file
# movie_filename = f'particle_locations_epsilon_{epsilon}_Nparticles_{Nparticles}.mp4'
# ani.save(movie_filename, writer='ffmpeg', fps=15)

# plt.close()
