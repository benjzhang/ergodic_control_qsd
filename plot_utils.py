import numpy as np
import matplotlib.pyplot as plt

def plot_periodic_trajectories(t, X, period=2, threshold=1.5, color='b', alpha=0.7):
    """
    Plots multiple particle trajectories in a periodic domain without artificial
    connections at boundaries.

    Parameters:
    - t: array-like, shape (time_length,), time values
    - X: array-like, shape (time_length, N_particles, 1) or (time_length, N_particles),
         particle positions in a periodic domain
    - period: float, periodic domain length (default: 2 for [-1,1])
    - threshold: float, threshold for detecting boundary crossings (default: 1.5)
    - color: str, color for all trajectories (default: 'b' for blue)
    - alpha: float, transparency of the plotted lines (default: 0.7)
    """
    t = np.asarray(t)
    X = np.asarray(X).squeeze()  # Ensures shape is (time_length, N_particles)

    # Check dimensions after squeezing
    if X.ndim == 1:  # If only one particle, reshape to 2D
        X = X[:, np.newaxis]
    elif X.ndim != 2:
        raise ValueError(f"X must have shape (time_length, N_particles), but got {X.shape}")

    N_particles = X.shape[1]

    plt.figure(figsize=(8, 4))

    # Loop over all particles
    for i in range(N_particles):
        x = X[:, i]  # Extract trajectory of particle i
        
        # Detect boundary crossings
        diffs = np.abs(np.diff(x))
        jumps = np.where(diffs > threshold)[0]  # Find large jumps

        # Split the trajectory into continuous segments
        segments = np.split(np.arange(len(x)), jumps + 1)

        # Plot each segment separately to avoid artificial crossings
        for seg in segments:
            plt.plot(t[seg], x[seg], linewidth=1.5, color=color, alpha=alpha)  

    # Labeling
    plt.xlabel("Time")
    plt.ylabel("Position")
    # plt.title(f"Periodic Trajectories of {N_particles} Particles Without Artificial Crossings")
    fig = plt.gcf()
    return fig



def plot_periodic_trajectories_list(t_list, X_list, period=2, threshold=1.5, color='b', alpha=0.7):
    """
    Plots multiple particle trajectories in a periodic domain without artificial
    connections at boundaries, handling lists of time and position arrays.

    Parameters:
    - t_list: list of arrays, each array is a time series for one particle
    - X_list: list of arrays, each array is a position series for one particle
    - period: float, periodic domain length (default: 2 for [-1,1])
    - threshold: float, threshold for detecting boundary crossings (default: 1.5)
    - color: str, color for all trajectories (default: 'b' for blue)
    - alpha: float, transparency of the plotted lines (default: 0.7)
    """
    plt.figure(figsize=(8, 4))

    N_particles = len(t_list)
    
    if len(t_list) != len(X_list):
        raise ValueError("t_list and X_list must have the same length.")

    # Loop over all particles
    for i in range(N_particles):
        t = np.array(t_list[i])  # Time array for particle i
        x = np.array(X_list[i]).squeeze()  # Position array for particle i (ensure it's 1D)
        
        if x.ndim != 1:
            raise ValueError(f"Each X_list[i] must be 1D, but got shape {x.shape} for particle {i}.")

        # Detect boundary crossings
        diffs = np.abs(np.diff(x))
        jumps = np.where(diffs > threshold)[0]  # Find large jumps

        # Split the trajectory into continuous segments
        segments = np.split(np.arange(len(x)), jumps + 1)

        # Plot each segment separately to avoid artificial crossings
        for seg in segments:
            plt.plot(t[seg], x[seg], linewidth=1.5, color=color, alpha=alpha)

    # Labeling
    plt.xlabel("Time")
    plt.ylabel("Position")
    # plt.title(f"Periodic Trajectories of {N_particles} Particles Without Artificial Crossings")
    fig = plt.gcf()
    return fig
