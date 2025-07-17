import numpy as np
import matplotlib.pyplot as plt

def simulate_pure_jump_processes_with_rates_and_step(T, b, a, h0, N, initial_position=0):
    """
    Simulate N pure jump processes on a lattice with given jump rate functions and step size.
    
    Parameters:
    - T: Total time of the simulation.
    - b: Drift function b(x).
    - a: Diffusion function a(x).
    - h0: Mean step size of the lattice.
    - N: Number of trajectories to simulate.
    - initial_position: Starting position of the processes.
    
    Returns:
    - all_times: Array of time points for all trajectories (shared time axis).
    - all_positions: 2D array of positions at each time point for each trajectory.
    """
    all_times = [0]
    all_positions = np.full((N, 1), initial_position, dtype=float)
    
    current_time = 0
    
    while current_time < T:
        h = np.random.uniform(h0 / 2, 3 * h0 / 2, size=N)
        
        # Calculate rates for all particles
        rate_right = (1 / (h ** 2)) * (h * np.maximum(b(all_positions[:, -1]), 0) + a(all_positions[:, -1]) / 2)
        rate_left = (1 / (h ** 2)) * (-h * np.minimum(b(all_positions[:, -1]), 0) + a(all_positions[:, -1]) / 2)
        total_rate = rate_left + rate_right
        
        if np.any(total_rate == 0):
            break
        
        # Sample the waiting time until the next jump for all particles
        waiting_times = np.random.exponential(1/total_rate)
        min_waiting_time = np.min(waiting_times)
        current_time += min_waiting_time
        
        if current_time > T:
            break
        
        all_times.append(current_time)
        
        # Update positions based on the direction of the jump
        jump_directions = np.zeros(N)
        for i in range(N):
            if waiting_times[i] == min_waiting_time:
                p = [rate_left[i] / total_rate[i], rate_right[i] / total_rate[i]]
                jump_directions[i] = np.random.choice([-h[i], h[i]], p=p)
        new_positions = all_positions[:, -1] + jump_directions
        all_positions = np.column_stack((all_positions, new_positions))
    
    return np.array(all_times), all_positions

# Example drift and diffusion functions
def b(x):
    return x  # Example drift function

def a(x):
    return 1 # Example diffusion function

# Parameters
T = 1  # Total time
h0 = 0.1  # Mean step size of the lattice
N = 25  # Number of trajectories to simulate
initial_position = 0  # Starting position

# Run the simulation
all_times, all_positions = simulate_pure_jump_processes_with_rates_and_step(T, b, a, h0, N, initial_position)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.step(all_times, all_positions[i], where='post', label=f'Trajectory {i+1}')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Pure jump process approximation of diffusion on a random lattice')
# plt.legend()
plt.grid(True)
plt.show()
