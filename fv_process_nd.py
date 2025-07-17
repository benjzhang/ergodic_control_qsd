import numpy as np
import matplotlib.pyplot as plt

def simulate_fleming_viot_with_killing(T, b, a, h0, N, c, initial_position, min_samples):
    dim = len(initial_position)
    all_times = [0]
    all_positions = np.tile(initial_position, (N, 1)).reshape(N, dim, 1)  # Shape (N, dim, 1)
    
    current_time = 0
    sample_count_after_T = 0
    
    while sample_count_after_T < min_samples:
        h = np.random.uniform(h0 / 2, 3 * h0 / 2, size=(N, dim))
        
        # Calculate rates for all particles in all dimensions
        b_val = b(all_positions[:, :, -1])
        a_val = a(all_positions[:, :, -1])
        
        rate_right = (1 / (h ** 2)) * (h * np.maximum(b_val, 0) + a_val / 2)
        rate_left = (1 / (h ** 2)) * (-h * np.minimum(b_val, 0) + a_val / 2)
        killing_rate = c(all_positions[:, :, -1])
        
        total_rate = np.sum(rate_left + rate_right, axis=1) + killing_rate
        
        if np.any(total_rate == 0):
            break
        
        waiting_times = np.random.exponential(1, size=N) / total_rate
        min_waiting_time = np.min(waiting_times)
        current_time += min_waiting_time
        
        if current_time > T:
            sample_count_after_T += 1
        
        all_times.append(current_time)
        
        new_positions = all_positions[:, :, -1].copy()
        
        for i in range(N):
            if waiting_times[i] == min_waiting_time:
                p_jump = [rate_left[i] / total_rate[i], rate_right[i] / total_rate[i]]
                p_kill = killing_rate[i] / total_rate[i]
                p = np.concatenate((p_jump[0], p_jump[1], [p_kill]), axis=0)
                p /= p.sum()
                choices = np.concatenate((-h[i], h[i], [np.nan]), axis=0)
                choice = np.random.choice(np.arange(len(choices)), p=p)
                
                if choice < dim:
                    new_positions[i, choice] -= h[i, choice]
                elif choice < 2 * dim:
                    new_positions[i, choice - dim] += h[i, choice - dim]
                else:
                    survivors = new_positions[~np.isnan(new_positions).any(axis=1)]
                    if len(survivors) > 0:
                        new_positions[i] = survivors[np.random.choice(len(survivors))]
                    else:
                        new_positions[i] = initial_position
        
        all_positions = np.concatenate((all_positions, new_positions[:, :, np.newaxis]), axis=2)
    
    return np.array(all_times), all_positions

# Example drift, diffusion, and killing rate functions for a 2D system
def b(x):
    return np.array([0.5 * (2*x[:, 0] * (x[:, 0]**2 - 4)**2 + 4 * x[:, 0]**3 * (x[:, 0]**2 - 4)), 
                     0.5 * (2*x[:, 1] * (x[:, 1]**2 - 4)**2 + 4 * x[:, 1]**3 * (x[:, 1]**2 - 4))]).T

def a(x):
    return np.ones_like(x)

def c(x):
    return np.sum(x**2, axis=1) + 7

# Parameters
T = 1
h0 = 0.05
N = 25
initial_position = np.array([0, 0])

# Run the simulation
all_times, all_positions = simulate_fleming_viot_with_killing(T, b, a, h0, N, c, initial_position, min_samples=10000)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(all_positions[i, 0, :], all_positions[i, 1, :], label=f'Trajectory {i+1}')
plt.xlabel('X Position', fontsize=14)
plt.ylabel('Y Position', fontsize=14)
plt.title('Simulation of Fleming-Viot Process in 2D with Killing Rate', fontsize=16)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True)
plt.show()

# Extract samples after time T
after_T_samples = all_positions[:, :, np.array(all_times) > T]

# Flatten the array to combine samples from all trajectories and all dimensions
all_samples = after_T_samples.reshape(-1, 2)

# Plot the normalized histogram of all samples after time T
plt.figure(figsize=(10, 6))
plt.hist2d(all_samples[:, 0], all_samples[:, 1], bins=50, density=True, cmap='viridis')
plt.xlabel('X Position', fontsize=14)
plt.ylabel('Y Position', fontsize=14)
plt.title('Normalized Histogram of Samples for All Trajectories in 2D', fontsize=16)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.colorbar(label='Density')
plt.grid(True)
plt.show()
