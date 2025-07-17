import numpy as np
import importlib
import utils
importlib.reload(utils)
from utils import inf_swap_rate, lex_order, lex_to_pair, event_rate, ordinary_transition, killing_cloning, sym_dynamics, vanilla_estimator

# Parameters
V0 = 2  # potential barrier height
Nparticles = 25  # number of particles
epsilon = 0.1 # temperature
V = np.array([V0, 0, V0, 0])

# Precompute transition rates
r = np.array([[-2, 1, 0, 1],
              [np.exp(-(V[0] - V[1]) / epsilon), -np.exp(-(V[0] - V[1]) / epsilon) - np.exp(-(V[2] - V[1]) / epsilon), np.exp(-(V[2] - V[1]) / epsilon), 0],
              [0, 1, -2, 1],
              [np.exp(-(V[0] - V[3]) / epsilon), 0, np.exp(-(V[2] - V[3]) / epsilon), -np.exp(-(V[0] - V[3]) / epsilon) - np.exp(-(V[2] - V[3]) / epsilon)]])

h = np.array([[-np.exp(-(V[0] - V[1]) / epsilon) - np.exp(-(V[0] - V[3]) / epsilon), np.exp(-(V[0] - V[1]) / epsilon), 0, np.exp(-(V[0] - V[3]) / epsilon)],
              [1, -2, 1, 0],
              [0, np.exp(-(V[2] - V[1]) / epsilon), -np.exp(-(V[2] - V[1]) / epsilon) - np.exp(-(V[2] - V[3]) / epsilon), np.exp(-(V[2] - V[3]) / epsilon)],
              [1, 0, 1, -2]])

# Initialize particles
states = 0 * np.random.randint(4, size=(Nparticles, 1))

# Initialize effective event rates
rates = -r[states,states]

# Initialize initial waiting times
dt = np.random.exponential(1 / rates)

# Initialize time
time = 0
Ntransitions = 100
time_vec = np.zeros(Ntransitions)

x_states = np.zeros((Nparticles, Ntransitions))
x_states[:, 0] = states[:,0]

# Run the simulation
for ii in range(Ntransitions - 1):
    dtmin = np.min(dt)
    index = np.argmin(dt)
    time += dtmin
    time_vec[ii + 1] = time
    z = int(x_states[index,ii])

    probabilities = -r[z, :] / r[z, z]
    probabilities[z] = 0 
    new_state = np.random.choice(4, 1, p=probabilities)    

    
    dt = dt - dtmin
    rates[index] = -r[new_state, new_state]
    dt[index] = np.random.exponential(1 / rates[index])

    x_states[:,ii+1] = x_states[:,ii]
    x_states[index, ii+1] = new_state

est = vanilla_estimator(x_states, time_vec, lambda x: x**2)

print(est)