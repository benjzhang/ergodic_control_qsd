import numpy as np
import importlib
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils
importlib.reload(utils)
from utils import pure_jump_approx_diffusion


# This file contains the main function to simulate a diffusion process via pure jump process on a lattice with given drift and diffusion functions


def b(x): 
    return -x * (x**2 - 4)
def a(x): 
    return 0.25 * np.ones_like(x)

# parameters
T = 10
h0 = 0.1
initial_position = np.array([0.])+2
# simulate
positions,times = pure_jump_approx_diffusion(T,b,a,h0,initial_position)
# # plot
# plt.plot(positions[:,0],positions[:,1])     
# plt.title('Diffusion Process')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()

# plt.plot(times,positions[:,0])
# plt.plot(times,positions[:,1])
# plt.title('Diffusion Process')
# plt.xlabel('time')
# plt.ylabel('x')
# plt.legend(['x1','x2'])
# plt.show()