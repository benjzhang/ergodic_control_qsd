import numpy as np
import importlib
import utils
importlib.reload(utils)
from utils import inf_swap_rate, lex_order, lex_to_pair, event_rate, ordinary_transition, killing_cloning, sym_dynamics, inf_swap_estimator

# Parameters
V0 = 2  # potential barrier height
Nparticles = 25  # number of particles
epsilon = 10 # temperature
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

# Precompute killing/cloning rates
c = np.zeros(4); offset = 0
c[0] = offset+ 2 - (np.exp(-(V[0] - V[1]) / epsilon) + np.exp(-(V[0] - V[3]) / epsilon))
c[1] = offset-2 + (np.exp(-(V[0] - V[1]) / epsilon) + np.exp(-(V[2] - V[1]) / epsilon))
c[2] = offset+2 - (np.exp(-(V[2] - V[1]) / epsilon) + np.exp(-(V[2] - V[3]) / epsilon))
c[3] = offset-2 + (np.exp(-(V[0] - V[3]) / epsilon) + np.exp(-(V[2] - V[3]) / epsilon))

# Construct rate matrix
sym_rate_matrix = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        z = lex_to_pair(i, 4)
        z1 = lex_to_pair(j, 4)
        if i != j:
            sym_rate_matrix[i, j] = sym_dynamics(z, z1, r, h, V, epsilon)

for i in range(16):
    sym_rate_matrix[i, i] = -np.sum(sym_rate_matrix[i, :])


time_limit = 100
num_reps = 1
est_vec = np.zeros(num_reps)

for i in range(num_reps):

    # Initialize particles
    states =  0*np.random.randint(4, size=(Nparticles, 2))

    # Initialize effective event rates
    rates = event_rate(states, sym_rate_matrix, c, V, epsilon)

    # Initialize initial waiting times
    dt = np.random.exponential(1 / rates)

    # Initialize time
    time = 0
    # Ntransitions = 20000
    time_vec = time

    # x_states = np.zeros((Nparticles, Ntransitions))
    # x_states[:, 0] = states[:, 0]

    x_states = states[:,0].reshape(Nparticles,1)
    # y_states = np.zeros((Nparticles, Ntransitions))
    # y_states[:, 0] = states[:, 1]
    y_states = states[:,1].reshape(Nparticles,1)

    # Run the simulation
    while time < time_limit:
        dtmin = np.min(dt)
        index = np.argmin(dt)
        time += dtmin
        time_vec = np.hstack((time_vec, time))
        z = states[index, :]

        ordinary_prob = -sym_rate_matrix[lex_order(np.array([z])), lex_order(np.array([z]))] / rates[index]
        # states_now = np.delete(states, index, axis=0)
        if np.random.rand() < ordinary_prob:
            z = ordinary_transition(np.array([z]), sym_rate_matrix)
            states[index, :] = z
            indices = index
        else:
            z, states, indices = killing_cloning(index, states, c, V, epsilon)

        # states[:index, :] = states_now[:index, :]
        # states[index + 1:, :] = states_now[index:, :]

        dt = dt - dtmin
        rates[indices] = event_rate(states[indices,:], sym_rate_matrix, c, V, epsilon)
        dt[indices] = np.random.exponential(1 / rates[indices])

        x_states = np.hstack((x_states, states[:, 0].reshape(Nparticles,1)))
        y_states = np.hstack((y_states, states[:, 1].reshape(Nparticles,1)))
            # y_states[:, ii+1] = states[:, 1]

    def g_function(x):
        return x**2

    est = inf_swap_estimator(x_states, y_states, time_vec, g_function, V, epsilon)




    est_vec[i] = est

partition_function = np.sum(np.exp(-V/epsilon))

true_expectation = np.sum(g_function(np.array([0,1,2,3])) * np.exp(-V/epsilon)/partition_function)

print(est_vec, true_expectation)



