import numpy as np
import importlib
import matplotlib.pyplot as plt
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate, fleming_viot_vanilla, resample_weighted_empirical_measure, resample_simple_weighted_empirical_measure
from plot_utils import plot_periodic_trajectories, plot_periodic_trajectories_list



# Fleming-Viot system simulation
x0 = 1.25
x1 = -1.25
epsilon = 0.1
lamda = 1

A = 1



def V(x): 
    return 0.25 * (x**2 - x0 ** 2) ** 2

def DV(x): 
    return x ** 3 - x * x0 **2 

def D2V(x): 
    return 3 * x ** 2 - x0 ** 2

def a(x): 
    return  epsilon  * np.ones_like(x)


# def c(x):
#     c0 = lamda + epsilon * x1 ** 4 / 2
#     c2 = x0**2 * x1 ** 2 + epsilon * x1 ** 4 /2 - 3 * epsilon / 2
#     c4 = -(x0**2 + x1 ** 2) - epsilon * x1 ** 2
#     c6 = 1 + epsilon / 2
#     return c0 + c2 * x ** 2 + c4 * x ** 4 + c6 * x ** 6


def c(x): 
    t1 = lamda 
    t2 = (x**3 - x0 ** 2 * x) * 4 * A * (x - x1) ** 3
    t3 = - 12 * A * epsilon / 2 * (x - x1) ** 2
    t4 = epsilon / 2 * 16 * A** 2 *  (x - x1) ** 6
    return t1 + t2 + t3 + t4

# parameters
T = 25
h0 = 0.1
Nparticles = 10
dim = 1

# initial_positionsx = 4* np.random.rand(Nparticles,dim) - 2
initial_positionsx = np.ones((Nparticles,dim)) * x0

# initial_positionsy = 4* np.random.rand(Nparticles,dim) - 2
initial_positionsy = np.ones((Nparticles,dim)) * x0

allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T,V,DV,D2V,a,epsilon,c,h0,initial_positionsx,initial_positionsy)

# vanilla fleming viot
allpositionsx_vanilla, alltime_vanilla = fleming_viot_vanilla(T,lambda x: -DV(x),a,c,h0,initial_positionsx)



# Plotting
# resample from trajectories according to rho
# Burn-in: exclude first T=5 seconds
burnin_time = 0
burnin_mask = alltime >= burnin_time

allpositions_resample = resample_weighted_empirical_measure(
    allpositionsx[burnin_mask,0,0], 
    all_rho[burnin_mask,0,0], 
    alltime[burnin_mask]
)



plt.figure(figsize=(8, 5))
plt.hist(allpositions_resample, bins=25, density=True, alpha=0.7, color='blue', label='Resampled Positions')
# Plot exp(-1/4 * (x^2-x1^2)^2) for comparison
x_vals = np.linspace(-2, 2, 400)
y_vals = np.exp(-0.25 * (x_vals**2 - x1**2)**2)

plt.plot(x_vals, y_vals, 'r-', label=r'$e^{-1/4 (x^2-x_1^2)^2}$')
plt.xlabel('Position')
plt.ylabel('Density')
plt.title('Histogram of FV-INS Resampled Positions (after burn-in)')
plt.legend()
plt.grid(True)
plt.show()



burnin_mask = alltime_vanilla >= burnin_time

allpositions_resample_vanilla = resample_simple_weighted_empirical_measure(
    allpositionsx_vanilla[burnin_mask,0,0], 
    alltime_vanilla[burnin_mask]
)
plt.figure(figsize=(8, 5))
plt.hist(allpositions_resample_vanilla, bins=25, density=True, alpha=0.7, color='blue', label='Resampled Positions')

# Plot exp(-1/4 * (x^2-x1^2)^2) for comparison
x_vals = np.linspace(-2, 2, 400)
y_vals = np.exp(-A* (x_vals - x1)**4)
y_vals /= np.trapz(y_vals, x_vals)  # Normalize for comparison

plt.plot(x_vals, y_vals, 'r-', label=r'$e^{-1/4 (x^2-x_1^2)^2}$')
plt.xlabel('Position')
plt.ylabel('Density')
plt.title('Histogram of Vanilla Positions (after burn-in)')
plt.legend()
plt.grid(True)
plt.show()

# Plot c(x) over a range of x values
x_vals = np.linspace(-2, 2, 400)
c_vals = c(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, c_vals, label='c(x)')
plt.xlabel('x')
plt.ylabel('c(x)')
plt.title('Plot of c(x)')
plt.legend()
plt.grid(True)
plt.show()



