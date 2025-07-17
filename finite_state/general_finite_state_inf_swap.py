import numpy as np
import importlib
import utils
importlib.reload(utils)
from utils import inf_swap_rate, forward_rate, backward_rate, lex_order, lex_to_pair, event_rate, ordinary_transition, killing_cloning, sym_dynamics, construct_sym_rate_matrix, inf_swap_estimator

# Parameters
epsilon = 0.2 # temperature
Nparticles = 100 # number of particles

# V = np.array([2,1,0,1,2,1,0,1]) 

num_states = 32
x_pos = np.linspace(0, 4, num_states+1)
V =  (np.cos(3/2 * x_pos* np.pi)+1)

V = V[0:-1]
x_pos = x_pos[0:-1]
# V = np.array([2,0,2,0])


# Precompute transition rates
r = forward_rate(V, epsilon)
h = backward_rate(V,epsilon)

# Precompute killing/cloning rates
offset = 0
c = np.diag(h -r.T) + offset
c_forward = -np.full(c.shape[0], offset) 
# Symmetrized rate matrix
sym_rate_matrix = construct_sym_rate_matrix(r, h, V, epsilon)


time_limit = 100
num_reps = 1
est_vec = np.zeros(num_reps)

for i in range(num_reps):

    # Initialize particles
    states =  np.random.randint(num_states, size=(Nparticles, 2)) # + num_states//2
    # Initialize effective event rates
    rates = event_rate(states, sym_rate_matrix, c_forward,c, V, epsilon)

    # Initialize initial waiting times
    dt = np.random.exponential(1 / rates)

    # Initialize time
    time = 0
    time_vec = time


    x_states = states[:,0].reshape(Nparticles,1)
    y_states = states[:,1].reshape(Nparticles,1)

    # Run the simulation
    while time < time_limit:
        dtmin = np.min(dt)
        index = np.argmin(dt)
        time += dtmin
        time_vec = np.hstack((time_vec, time))
        z = states[index, :]

        ordinary_prob = -sym_rate_matrix[lex_order(np.array([z]),num_states), lex_order(np.array([z]),num_states)] / rates[index]
        if np.random.rand() < ordinary_prob:
            z = ordinary_transition(np.array([z]), sym_rate_matrix)
            states[index, :] = z
            indices = index
        else:
            z, states, indices = killing_cloning(index, states, c_forward,c, V, epsilon)

        dt = dt - dtmin
        rates[indices] = event_rate(states[indices,:], sym_rate_matrix, c_forward,c, V, epsilon)
        dt[indices] = np.random.exponential(1 / rates[indices])

        x_states = np.hstack((x_states, states[:, 0].reshape(Nparticles,1)))
        y_states = np.hstack((y_states, states[:, 1].reshape(Nparticles,1)))

    def g_function(x):
        return x**2
    trajectories_x= x_pos[x_states]
    trajectories_y = x_pos[y_states]
    est = inf_swap_estimator(x_states, y_states,x_pos, time_vec, g_function, V, epsilon)




    est_vec[i] = est

partition_function = np.sum(np.exp(-V/epsilon))


true_expectation = np.sum(g_function(x_pos) * np.exp(-V/epsilon)/partition_function)

print(est_vec, true_expectation)


import matplotlib.pyplot as plt

for i in range(Nparticles):
    plt.scatter(time_vec, trajectories_x[i, :], s=0.1, color='blue')
    plt.ylim(0, 4)


plt.xlabel('Time')
plt.ylabel('X States')
plt.title('X States of Particles Over Time')
plt.show()


for i in range(Nparticles):
    plt.scatter(time_vec, trajectories_y[i, :], s=0.1, color='blue')
    plt.ylim(0, 4)

plt.xlabel('Time')
plt.ylabel('Y States')
plt.title('Y States of Particles Over Time')
plt.show()


print(np.mean(est_vec), true_expectation)