import numpy as np
import importlib
import matplotlib.pyplot as plt
import utils
importlib.reload(utils)
from utils import sde_transition_rates, one_step_sde, pure_jump_approx_diffusion, fleming_viot, event_rates, inf_swap_rate, killing_cloning, symmetrized_kill_clone_rate, fleming_viot_vanilla
from plot_utils import plot_periodic_trajectories, plot_periodic_trajectories_list


# Example drift, diffusion, and killing rate functions
# def b(x):
#     return 0.5* (2*x * (x**2 - 4)**2 + 4 * x** 3 * (x**2 - 4) )# Example drift function

# def b(x): 
#     return x

# def a(x):
#     return 1  # Example diffusion function

# def c(x):
#     return 0.5 * (x**2)+1

# def c(x):
#     return x**2 +7 + 0.5 * (2* (x**2 -4)**2 + 8* x**2 *(x**2 - 4) + 12*x**3 * (x**2-4) + 8 * x**4 )# Example killing rate function


# Fleming-Viot system simulation
epsilon = 1

dl = 4

def V(x): 
    x = -dl + np.mod(x+dl,2*dl)
    return -0.5 * x**2 * (x**2 - 4) ** 2

def DV(x): 
    x = -dl + np.mod(x+dl,2*dl)
    return -0.5* (2*x * (x**2 - 4)**2 + 4 * x** 3 * (x**2 - 4) )

def D2V(x): 
    x = -dl + np.mod(x+dl,2*dl)
    return -((x**2 - 4) **2 + 4 * x**2 * (x**2 - 4) + 6 * x**2 * (x**2 - 4) + 4 * x**4)

def a(x): 
    return  epsilon * np.ones_like(x)

def c(x):
    x = -dl + np.mod(x+dl,2*dl)
    return (x**2-25 ) #- 0.5 * (2* (x**2 -4)**2 + 8* x**2 *(x**2 - 4) + 12*x**2 * (x**2-4) + 8 * x**4 )

# parameters
T = 2
h0 = 0.1
Nparticles = 20
dim = 1

# initial_positionsx = 4* np.random.rand(Nparticles,dim) - 2
initial_positionsx = -np.ones((Nparticles,dim)) * 0

# initial_positionsy = 4* np.random.rand(Nparticles,dim) - 2
initial_positionsy = np.ones((Nparticles,dim)) * 0

allpositionsx, allpositionsy, alltime, all_rho = fleming_viot(T,V,DV,D2V,a,epsilon,c,h0,initial_positionsx,initial_positionsy)

# allpositions, alltime = fleming_viot_vanilla(T,lambda x: -DV(x), a, lambda x: -D2V(x), h0, initial_positionsx)

# Nparticles = initial_positionsx.shape[0]
# x_allparticles = initial_positionsx
# y_allparticles = initial_positionsy
# current_time = 0
# all_time = np.array([current_time])
# all_positionsx = np.array([initial_positionsx])
# all_positionsy = np.array([initial_positionsy])


# rho_particles = np.array([inf_swap_rate(x_allparticles[i,:],y_allparticles[i,:],V,epsilon) for i in range(Nparticles)])
# all_rho = np.array([rho_particles])

# # compute rates
# ratesx = np.array([event_rates(x_allparticles[i,:],rho_particles[i],DV,D2V,a,c,h0) for i in range(Nparticles)])
# ratesy = np.array([event_rates(y_allparticles[i,:],1-rho_particles[i],DV,D2V,a,c,h0) for i in range(Nparticles)])
# clocksx = np.array([np.random.exponential(1/ratesx[i,0]) for i in range(Nparticles)])
# clocksy = np.array([np.random.exponential(1/ratesy[i,0]) for i in range(Nparticles)])

# net_event_ratex = ratesx[:,0]
# net_event_ratey = ratesy[:,0]

# dynamics_ratex = ratesx[:,1]
# dynamics_ratey = ratesy[:,1]

# killclone_ratex = ratesx[:,2]
# killclone_ratey = ratesy[:,2]


# while current_time < T: 
#     # find next event
#     index = np.argmin(np.append(clocksx,clocksy))
#     elapsed_time = np.min(np.append(clocksx,clocksy))
#     current_time += elapsed_time

    
#     #subtract time from all other clocks
#     clocksx = clocksx - elapsed_time
#     clocksy = clocksy - elapsed_time


#     # figure out x or y event
#     if index < Nparticles:
#         #figure out regular dynamics rate or killing-cloning rate
#         if np.random.rand() < np.sum(dynamics_ratex[index])/ net_event_ratex[index]: # regular dynamics event
#             x_allparticles[index,:] = one_step_sde(x_allparticles[index,:],dynamics_ratex[index],h0)
#             index_to_recompute = [index]
#         else: #else killing/cloning event
#             x_allparticles, y_allparticles, index_to_recompute = killing_cloning(x_allparticles,y_allparticles,killclone_ratex[index], rho_particles, index)

#         # update rho and rates only for changed particles
#         for i in index_to_recompute:
#             i = int(i)
#             if i > Nparticles:
#                 i = i - Nparticles
#             rho_particles[i] = inf_swap_rate(x_allparticles[i,:],y_allparticles[i,:],V,epsilon)
#             ratesx[i,:] = event_rates(x_allparticles[i,:],rho_particles[i],DV,D2V,a,c,h0)
#             ratesy[i,:] = event_rates(y_allparticles[i,:],1-rho_particles[i],DV,D2V,a,c,h0)

#             # update clock for next event
#             clocksx[i] = np.random.exponential(1/ratesx[i,0])
#             clocksy[i] = np.random.exponential(1/ratesy[i,0])
        

#     else:
#         index = index - Nparticles
#         #figure out regular dynamics rate or killing-cloning rate
#         if np.random.rand() < np.sum(dynamics_ratey[index]) / net_event_ratey[index]:
#             y_allparticles[index,:] = one_step_sde(y_allparticles[index,:],dynamics_ratey[index],h0)
#             index_to_recompute = [index]
#         else:
#             y_allparticles, x_allparticles, index_to_recompute = killing_cloning(y_allparticles,x_allparticles,killclone_ratey[index], 1-rho_particles, index)


#         # update rho and rates only for changed particles
#         for i in index_to_recompute:
#             i = int(i)

#             rho_particles[i] = inf_swap_rate(x_allparticles[i,:],y_allparticles[i,:],V,epsilon)
#             ratesx[i,:] = event_rates(x_allparticles[i,:],rho_particles[i],DV,D2V,a,c,h0)
#             ratesy[i,:] = event_rates(y_allparticles[i,:],1-rho_particles[i],DV,D2V,a,c,h0)

#             # update clock for next event
#             clocksx[i] = np.random.exponential(1/ratesx[i,0])
#             clocksy[i] = np.random.exponential(1/ratesy[i,0])
    

#     net_event_ratex = ratesx[:,0]
#     net_event_ratey = ratesy[:,0]

#     dynamics_ratex = ratesx[:,1]
#     dynamics_ratey = ratesy[:,1]

#     killclone_ratex = ratesx[:,2]
#     killclone_ratey = ratesy[:,2]
    
#     all_positionsx = np.vstack((all_positionsx,np.array([x_allparticles])))
#     all_positionsy = np.vstack((all_positionsy,np.array([y_allparticles])))
#     all_time = np.append(all_time,current_time)
#     all_rho = np.vstack((all_rho,np.array([rho_particles])))


all_positionsx = -dl + np.mod(allpositionsx+dl,2*dl)
all_positionsy = -dl + np.mod(allpositionsy+dl,2*dl)
plt.figure(figsize=(10, 6))
plt.plot(alltime, all_positionsy[:,:,0],  label='x')
plt.xlim(0,T)
plt.ylim(-4, 4) 
# plt.plot(all_time, all_positionsy[:,:,0], c='r', label='y')


# plot_periodic_trajectories(all_time,all_positionsx,2*dl-0.05)
# plot_periodic_trajectories(all_time,all_positionsy,2*dl-0.05)

# traj_x,times = pure_jump_approx_diffusion(T,lambda x: DV(x), a, h0, np.array([0.]))