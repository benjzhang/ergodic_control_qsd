import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from utils import resample_weighted_empirical_measure



# Load the aog_test.npy file
data_path = '/Users/bjzhang/gitrepos/ergodic_control_qsd/modular/aog/aog_test_example.npy'
data = np.load(data_path, allow_pickle=True)

# Extract relevant data from the loaded file
 
allpositionsx = data.item().get('allpositionsx')  # Particle positions over time (x-coordinates)
allpositionsy = data.item().get('allpositionsy')  # Particle positions over time (y-coordinates)
alltime = data.item().get('alltime')  # Time points
all_rho = data.item().get('all_rho')  # Density values over time
simple_trajectories_times = data.item().get('simple_trajectories_times')  # Times for simple trajectories
simple_trajectories_x = data.item().get('simple_trajectories_x')  # Simple trajectories (x-coordinates)
T = data.item().get('T')  # Parameter T
epsilon = data.item().get('epsilon')  # Parameter epsilon
h0 = data.item().get('h0')  # Parameter h0
Nparticles = data.item().get('Nparticles')  # Number of particles
sigma = data.item().get('sigma')  # Parameter sigma
dim = data.item().get('dim')  # Dimensionality
left = data.item().get('left')  # Left boundary
right = data.item().get('right')  # Right boundary
D2V_values = data.item().get('D2V_values')  # Second derivative of potential values







T = 100
epsilon = 0.4
h0 = 0.1
Nparticles = 5



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




# Plot the trajectories on the same plot as the contour
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

for i in range(Nparticles):
    plt.scatter(allpositionsx[:, i, 0], allpositionsx[:, i, 1], s=1, label=f'Particle {i+1}')

plt.title('Infinite swapping trajectories')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()

# Plot one representative trajectory on the potential
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')
# Plot the trajectory of the first particle
particle_idx = 0  # Index of the particle to plot
plt.scatter(allpositionsx[:, particle_idx, 0], 
            allpositionsx[:, particle_idx, 1], 
            color='red', s=1, label=f'Trajectory of Particle {particle_idx + 1}')

plt.title(f'Trajectory of Particle {particle_idx + 1}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



# Plot simple trajectories
# Plot simple trajectories up to time T
plt.figure(figsize=(8, 6))

# Find the index corresponding to T
T = 25  # You can change this value to any desired time
time_idx = (np.abs(simple_trajectories_times[0] - T)).argmin()

# Plot the potential as background
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

# Plot the simple trajectories up to time T
for i in range(2 * Nparticles):
    plt.scatter(simple_trajectories_x[i][:time_idx, 0], 
                simple_trajectories_x[i][:time_idx, 1], 
                s=1, label=f'Particle {i+1}',color = 'red')

plt.title(f'Ensemble locations of simple dynamics up to T={T}')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()

# Plot one trajectory
plt.figure(figsize=(8, 6))

# Find the index corresponding to T
T = 25  # You can change this value to any desired time
time_idx = (np.abs(simple_trajectories_times[0] - T)).argmin()

# Plot the potential as background
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

# Plot the simple trajectories up to time T
for i in range(1):
    plt.scatter(simple_trajectories_x[i][:time_idx, 0], 
                simple_trajectories_x[i][:time_idx, 1], 
                s=1, label=f'Particle {i+1}',color = 'red')

plt.title(f'One sample of simple dynamics up to T={T}')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()










# Plot allpositionsx and allpositionsy up to time T=10 with potential as background
plt.figure(figsize=(8, 6))

# Find the index corresponding to T=10
time_idx = (np.abs(alltime - 25)).argmin()

# Plot the potential as background
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

# Plot the trajectories as scatter plots
for i in range(5):
    plt.scatter(allpositionsx[:time_idx, i, 0], allpositionsx[:time_idx, i, 1], label=f'Particle {i+1} (x)', alpha=0.7, s=1,color = 'red')
    # plt.scatter(allpositionsy[:time_idx, i, 0], allpositionsy[:time_idx, i, 1], label=f'Particle {i+1} (y)', alpha=0.7, s=1,color = 'red')

plt.title('Ensemble locations up to T=25')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()




## One trajectory
# Plot the trajectory of one particle for the entire T=100 simulation
plt.figure(figsize=(8, 6))

# Find the index corresponding to the flexible time T
T = 25  # You can change this value to any desired time
time_idx = (np.abs(alltime - T)).argmin()

# Plot the potential as background
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

# Plot the trajectory of the first particle up to time T
particle_idx = 0  # Index of the particle to plot
plt.scatter(allpositionsx[:time_idx, particle_idx, 0], 
            allpositionsx[:time_idx, particle_idx, 1], 
            color='red', s=1, label=f'Trajectory of Particle {particle_idx + 1} up to T={T}')

plt.title(f'Single trajectory of Fleming-Viot system up to T={T}')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()







## Plot one trajectory with opacity based on all_rho
plt.figure(figsize=(8, 6))

# Find the index corresponding to the flexible time T
T = 0.1 # You can change this value to any desired time
time_idx = (np.abs(alltime - T)).argmin()

# Plot the potential as background
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')

# Plot the trajectory of the first particle up to time T with opacity based on all_rho
particle_idx = 0  # Index of the particle to plot
trajectory_x = allpositionsx[:time_idx, particle_idx, 0]
trajectory_y = allpositionsx[:time_idx, particle_idx, 1]
rho_values = all_rho[:time_idx, particle_idx]/100



for i in range(len(trajectory_x)):
    plt.scatter(trajectory_x[i], trajectory_y[i], color='red', s=1, alpha=rho_values[i])

plt.title(f'Trajectory of Particle {particle_idx + 1} with Opacity Based on all_rho up to T={T}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()





# resample weighted empirical measure

reweighted_INS_samples = resample_weighted_empirical_measure(allpositionsx[:,0,:],all_rho[:,0],alltime)

# plot reweighted samples on the potential
plt.figure(figsize=(8, 6))
# Plot the potential as background 
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')
# Plot the reweighted samples
plt.scatter(reweighted_INS_samples[0:100000, 0], reweighted_INS_samples[0:100000, 1], color='red', s=1, label='Reweighted Samples')
plt.title('Reweighted Samples on the Potential')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()



# sample from equivalent gaussian mixture

# Sample from an array of 16 Gaussians with periodic boundary conditions
def sample_from_gaussians(num_samples, centers, variance, left, right):
    samples = []
    for _ in range(num_samples):
        # Randomly choose a center
        center = centers[np.random.choice(len(centers))]
        # Sample from a Gaussian centered at the chosen center
        sample = np.random.multivariate_normal(center, variance * np.eye(len(center)))
        # Apply periodic boundary conditions
        sample = np.mod(sample - left, right - left) + left
        samples.append(sample)
    return np.array(samples)

# Define parameters
num_samples = 100000
centers = np.array([(i, j) for i in range(4) for j in range(4)])  # Centers of the Gaussians
variance = 0.01

# Generate samples
gaussian_samples = sample_from_gaussians(num_samples, centers, variance, left, right)

# Plot the samples on the potential
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray', vmin=0, vmax=10)
plt.colorbar(contour, label='Potential Value')
plt.scatter(gaussian_samples[:, 0], gaussian_samples[:, 1], color='red', s=1, label='Gaussian Samples')
plt.title('Samples from Gaussian Mixture with Periodic Boundary Conditions')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
plt.show()