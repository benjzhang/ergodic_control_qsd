import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    return x  # Example drift function

def a(x):
    return 1  # Example diffusion function

def c(x):
    return (0.5 * x ** 2 )

# Parameters
T = 2  # Time after which to start collecting samples
h0 = 0.05  # Mean step size of the lattice
N = 10  # Number of trajectories to simulate
initial_position = 0  # Starting position

# Run the simulation
all_times, all_positions = simulate_fleming_viot_with_killing(T, b, a, h0, N, c, initial_position, min_samples=1000)

# Create a video of the simulation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, np.max(all_times))
ax.set_ylim(np.min(all_positions) - 1, np.max(all_positions) + 1)
lines = [ax.step([], [], where='post')[0] for _ in range(N)]
ax.set_xlabel('Time')
ax.set_ylabel('Position')
ax.set_title('Simulation of Fleming-Viot Process with Killing Rate')

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i, line in enumerate(lines):
        line.set_data(all_times[:frame], all_positions[i, :frame])
    return lines

ani = animation.FuncAnimation(fig, update, frames=range(0,len(all_times),10), init_func=init, blit=True, interval=5)

# Save the video
ani.save('fleming_viot_simulation.mp4', writer='ffmpeg')

plt.show()
