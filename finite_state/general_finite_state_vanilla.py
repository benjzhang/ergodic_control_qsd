import numpy as np
import importlib
import utils
importlib.reload(utils)
from utils import inf_swap_rate, forward_rate, backward_rate, lex_order, lex_to_pair, event_rate, ordinary_transition, killing_cloning, sym_dynamics, construct_sym_rate_matrix, inf_swap_estimator, vanilla_estimator

# Parameters
epsilon = 0.2 # temperature
Nparticles = 50 # number of particles

# V = np.array([2,1,0,1,2,1,0,1]) 

num_states =25
x_pos = np.linspace(0, 4, num_states+1)
V =  (np.cos(3/2 * x_pos* np.pi)+1)

V = V[0:-1]
x_pos = x_pos[0:-1]
# V = np.array([2,0,2,0])


# Precompute transition rates
r = forward_rate(V,epsilon)



# Initialize particles
states = 0 * np.random.randint(num_states, size=(Nparticles, 1))+ num_states//2

# Initialize effective event rates
rates = -r[states,states]

# Initialize initial waiting times
dt = np.random.exponential(1 / rates)


time_limit = 100
num_reps = 1
est_vec = np.zeros(num_reps)

for i in range(num_reps):

    # Initialize particles
    states =  0*np.random.randint(num_states, size=(Nparticles, 1))+num_states//2

    # Initialize effective event rates
    rates = -r[states,states]

    # Initialize initial waiting times
    dt = np.random.exponential(1 / rates)

    # Initialize time
    time = 0
    time_vec = time

    x_states_now = states[:,0].reshape(Nparticles,1)
    x_states = states[:,0].reshape(Nparticles,1)

    # Run the simulation
    while time < time_limit:
        dtmin = np.min(dt)
        index = np.argmin(dt)
        time += dtmin
        time_vec = np.hstack((time_vec, time))
        z = x_states_now[index][0]

        probabilities = -r[z, :] / r[z, z]
        probabilities[z] = 0
        new_state = np.random.choice(num_states,1,p=probabilities)


        dt = dt - dtmin
        rates[index] = -r[new_state,new_state]
        dt[index] = np.random.exponential(1 / rates[index])

        x_states_now[index] = new_state
        x_states = np.hstack((x_states, x_states_now.reshape(Nparticles,1)))

    def g_function(x):
        return x**2
    trajectories_x= x_pos[x_states]
    est = vanilla_estimator(trajectories_x, time_vec, g_function)



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
