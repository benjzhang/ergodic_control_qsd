import numpy as np
import importlib
import matplotlib.pyplot as plt
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_dynamics,symmetrized_kill_clone_rate

# Fleming-Viot system simulation
epsilon = 1


def V(x): 
    return -0.5 * np.sum(x**2)

def DV(x): 
    return -x

def D2V(x): 
    return -np.sum(np.ones_like(x))

def a(x): 
    return  epsilon * np.ones_like(x)

def c(x):
    return np.sum(x**2) 

# parameters
T = 20
h0 = 0.1
Nparticles = 20
dim = 1

initial_positionsx = np.zeros((Nparticles,dim)) 
initial_positionsy = np.zeros((Nparticles,dim)) 
 

allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T,V,DV,D2V,a,epsilon,c,h0,initial_positionsx,initial_positionsy)

plt.plot(alltime, allpositionsx[:,:,0], label='x')
plt.plot(alltime, allpositionsy[:,:,0], label='y')


# plt.hist(allpositionsx[10000:20000,1,0], bins=10, density=True, alpha=0.5, label='x')