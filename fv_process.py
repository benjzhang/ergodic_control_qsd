import numpy as np
import matplotlib.pyplot as plt

def simulate_fleming_viot_with_killing(T, b, a, h0, N, c, initial_position, min_samples):
    """
    Simulate N pure jump processes on a lattice with given jump rate functions, step size, and killing rate.
    
    Parameters:
    - T: Total time of the simulation.
    - b: Drift function b(x).
    - a: Diffusion function a(x).
    - h0: Mean step size of the lattice.
    - N: Number of trajectories to simulate.
    - c: Killing rate function c(x).
    - initial_position: Starting position of the processes.
    - min_samples: Minimum number of samples to collect after time T.
    
    Returns:
    - all_times: Array of time points for all trajectories (shared time axis).
    - all_positions: 2D array of positions at each time point for each trajectory.
    """
    all_times = [0]
    all_positions = np.full((N, 1), initial_position, dtype=float)
    
    current_time = 0
    sample_count_after_T = 0
    
    while sample_count_after_T < min_samples:
        h = np.random.uniform(h0 / 2, 3 * h0 / 2, size=N)
        
        # Calculate rates for all particles
        rate_right = (1 / (h ** 2)) * (h * np.maximum(b(all_positions[:, -1]), 0) + a(all_positions[:, -1]) / 2)
        rate_left = (1 / (h ** 2)) * (-h * np.minimum(b(all_positions[:, -1]), 0) + a(all_positions[:, -1]) / 2)
        killing_rate = c(all_positions[:, -1])
        total_rate = rate_left + rate_right + killing_rate
        
        if np.any(total_rate == 0):
            break
        
        # Sample the waiting time until the next jump for all particles
        waiting_times = np.random.exponential(1, size=N) / total_rate
        min_waiting_time = np.min(waiting_times)
        current_time += min_waiting_time
        
        if current_time > T:
            sample_count_after_T += 1
        
        all_times.append(current_time)
        
        new_positions = all_positions[:, -1].copy()
        
        # Determine actions based on the minimum waiting time
        for i in range(N):
            if waiting_times[i] == min_waiting_time:
                p_jump = [rate_left[i] / total_rate[i], rate_right[i] / total_rate[i]]
                p_kill = killing_rate[i] / total_rate[i]
                p = np.array([p_jump[0], p_jump[1], p_kill])
                p /= p.sum()  # Normalize to ensure probabilities sum to 1
                choice = np.random.choice([-h[i], h[i], np.nan], p=p)
                if not np.isnan(choice):
                    new_positions[i] += choice
                else:
                    # Resample position from surviving particles
                    survivors = new_positions[~np.isnan(new_positions)]
                    if len(survivors) > 0:
                        new_positions[i] = np.random.choice(survivors)
                    else:
                        new_positions[i] = initial_position  # Fallback if all particles are killed
        
        all_positions = np.column_stack((all_positions, new_positions))
    
    return np.array(all_times), all_positions

# Example drift, diffusion, and killing rate functions
def b(x):
    return -0.5* (2*x * (x**2 - 4)**2 + 4 * x** 3 * (x**2 - 4) )# Example drift function

# def b(x): 
#     return x

def a(x):
    return 1  # Example diffusion function

# def c(x):
#     return 0.5 * (x**2)

def c(x):
    return 0*(x**2 +21)+ 0*23+0* 0.5 * (2* (x**2 -4)**2 + 8* x**2 *(x**2 - 4) + 12*x**2 * (x**2-4) + 8 * x**4 )# Example killing rate function

# epsilon = 0.15

# def V(x): 
#     return np.cos(2 * np.pi * x) / (2 * np.pi)

# def b(x):
#     return -np.sin(2 * np.pi * x) 

# def D2V(x):
#     return -2 * np.pi * np.cos(2 * np.pi * x)

# def a(x): 
#     return  epsilon * np.ones_like(x)

# def c(x):
#     return np.sum( x**2 + 1)+2 * np.pi * np.cos(2 * np.pi * x) 

# Parameters
T = 1.5 # Time after which to start collecting samples
h0 = 0.1  # Mean step size of the lattice
N = 20 # Number of trajectories to simulate
initial_position = 0  # Starting position

# Run the simulation
all_times, all_positions = simulate_fleming_viot_with_killing(T, b, a, h0, N, c, initial_position, min_samples=1000)
# Plot the results with larger title, axis labels, and ticks
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.step(all_times, all_positions[i], where='post', label=f'Trajectory {i+1}')
plt.xlabel('Time', fontsize=14)  # Larger font size for x-axis label
plt.ylabel('Position', fontsize=14)  # Larger font size for y-axis label
# plt.title('Simulation of Fleming-Viot Process with Killing Rate', fontsize=16)  # Larger font size for the title
plt.ylim(-4, 4)  # Set y-axis limits to [-2, 2]
plt.tick_params(axis='both', which='major', labelsize=12)  # Larger ticks for both x and y axis
# plt.legend()
plt.grid(True)
plt.show()

# Extract samples after time T
after_T_samples = all_positions[:, np.array(all_times) > T]

# Flatten the array to combine samples from all trajectories
all_samples = after_T_samples.flatten()

# Plot the normalized histogram of all samples after time T with larger title, axis labels, and ticks
plt.figure(figsize=(10, 6))
plt.hist(all_samples, bins=20, alpha=0.75, density=True)  # Use density=True to normalize the histogram
plt.xlabel('Position', fontsize=14)  # Larger font size for x-axis label
plt.ylabel('Density', fontsize=14)  # Larger font size for y-axis label
# plt.title('Normalized Histogram of Samples for All Trajectories', fontsize=16)  # Larger font size for the title
plt.xlim(-4, 4)  # Set x-axis limits to [-2, 2] for the histogram
plt.tick_params(axis='both', which='major', labelsize=12)  # Larger ticks for both x and y axis
plt.grid(True)
plt.show()





