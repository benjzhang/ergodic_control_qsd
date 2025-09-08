import numpy as np
from .sde_simulation import sde_transition_rates, one_step_sde


### infinite swapping functions

def inf_swap_rate(x,y,V,epsilon): 
    """
    Computes infinite swap fraction between pair of coupled particles

    x : particle 1, shape (d,)
    y: particle 2, shape (d,)
    V: potential function, callable
    epsilon: temperature, scalar
    """
    Vx = V(x)
    Vy = V(y) 

    rho_eps = 1/ (np.exp(2 *(Vx - Vy)/epsilon) + 1)
    return rho_eps


def symmetrized_dynamics(x,rho_eps,DV): 
    """
    Evaluates symmetrized dynamics of particle x with partner particle y. Evaluating this function in the other direction provides the symmetrized dynamics for particle y with partner particle x

    x : particle 1, shape (d,)
    rhoeps: infinite swap fraction recomputed with partner particle y, scalar
    DV: potential gradient, callable

    returns evaluations of symmetrized dynamics bsymm, asymm , shape (d,)
    """
    # rho_eps = inf_swap_rate(x,y,V,epsilon)
    bsymm = (1 - 2 * rho_eps) * DV(x)

    return bsymm


def symmetrized_kill_clone_rate(x,rhoeps,D2V,c): 
    """
    Computes symmetrized kill clone rate for pair of coupled particles

    x : particle 1, shape (d,)
    rhoeps: infinite swap fraction recomputed with partner particle y, scalar
    D2V: potential laplacian, callable
    c: kill rate, callable
    
    returns evaluation of kill/clone rate for particle x
    """
    
    # rho_eps_yx = inf_swap_rate(y,x,V,epsilon)
    rho_eps_yx = 1 - rhoeps
    c_symm = c(x) - rho_eps_yx * D2V(x)

    return c_symm


###

### Fleming-Viot event rates

def event_rates(x,rhoeps,DV,D2V,a,c,h):
    """ 
    Computes net event rate for particle x with partner pair y.
    
    x : particle 1, shape (d,)
    rhoeps: infinite swap fraction recomputed with partner particle y, scalar
    DV: potential gradient, callable
    D2V: potential laplacian, callable
    a: diffusion coefficient, diagonal components only , callable
    c: kill rate, callable
    h: mean step size of lattice, scalar

    returns net_event_rate for particle x, scalar
            kill_clone_rate for particle x, scalar
            dynamics transition rate for particle x, vector of shape (2*d,)
    """

    h_step = np.random.uniform(h/2,3*h/2)

    ## dynamics rate
    bsymm = symmetrized_dynamics(x,rhoeps,DV)
    aeval = a(x)
    dynamics_rates = sde_transition_rates(x,bsymm,aeval,h_step)

    ## kill/clone rate
    c_symm = symmetrized_kill_clone_rate(x,rhoeps,D2V,c)

    ## net event rate
    net_event_rate = np.sum(dynamics_rates) + abs(c_symm)

    return net_event_rate, dynamics_rates, c_symm, h_step



###

### Killing-cloning functions

def killing_cloning(x_allparticles,y_allparticles,c_symm, rho_all, index): 
    """
    Performs killing-cloning event for particle x with partner particle y. The index denotes the particle pair to be killed or cloned. 
    
    x_allparticles : all particle positions, shape (n,d)
    y_allparticles: all particle positions, shape (n,d)"""

    Nparticles = x_allparticles.shape[0]
    xnow = x_allparticles[index,:].copy()
    ynow = y_allparticles[index,:].copy()
    rho_eps_xy = rho_all[index]
    
    newindex = np.random.choice(Nparticles) # randomly sample from all indices

    if newindex == index: 
        xnew = xnow; index_to_recompute = np.array([index]) # if new index is same as old, assign back to original particle

    else: 
        if c_symm> 0:  # kill and respawn
            if np.random.rand() < rho_eps_xy : # figure out if fwd or bwd
                if np.random.rand() < rho_all[newindex] :# otherwise, find a forward particle to jump to 
                    xnew = x_allparticles[newindex,:].copy(); index_to_recompute = np.array([newindex])
                else: xnew = y_allparticles[newindex,:].copy(); index_to_recompute = np.array([newindex])
            else: 
                if np.random.rand() < 1-rho_all[newindex] :# otherwise, find a backward particle to jump to 
                    xnew = y_allparticles[newindex,:].copy(); index_to_recompute = np.array([newindex])
                else: xnew = x_allparticles[newindex,:].copy(); index_to_recompute = np.array([newindex])
            
            x_allparticles[index,:] = xnew

        else: # clone and cull
            if np.random.rand() < rho_eps_xy : # figure out if fwd or bwd
                if np.random.rand() < rho_all[newindex] : # otherwise, clone current particle by body-snatching the forward particle at newindex
                    x_allparticles[newindex,:] = xnow; index_to_recompute = np.array([index, newindex])
                else: y_allparticles[newindex,:] = xnow; index_to_recompute = np.array([index, newindex])
            else:
                if np.random.rand() < 1-rho_all[newindex] : # otherwise, clone current particle by body-snatching the backward particle at newindex
                    y_allparticles[newindex,:] = xnow; index_to_recompute = np.array([index, newindex])
                else: x_allparticles[newindex,:] = xnow; index_to_recompute = np.array([index, newindex])
                
    return x_allparticles, y_allparticles, index_to_recompute



## Fleming-Viot particle system

def fleming_viot(T, V, DV, D2V, a, epsilon, c,h0, initial_positionsx,initial_positionsy): 
    """
    Simulates n-dimensional Fleming-Viot particle system with given potential, drift, diffusion and kill/clone functions.
    inputs. This method implements infinite swapping. 
    - T: time horizon 
    - V: potential function
    - DV: potential gradient
    - D2V: potential laplacian
    - a: diffusion coefficient diagonal components only
    - epsilon: temperature
    - c: kill rate
    - h0: mean step size of lattice
    - Nparticles: number of particles
    - initial_positions: initial positions of the particles
    """
        
    # simulate
    Nparticles = initial_positionsx.shape[0]
    x_allparticles = initial_positionsx.copy()
    y_allparticles = initial_positionsy.copy()
    current_time = 0
    all_time = np.array([current_time])
    all_positionsx = np.array([initial_positionsx.copy()])
    all_positionsy = np.array([initial_positionsy.copy()])


    rho_particles = np.array([inf_swap_rate(x_allparticles[i,:],y_allparticles[i,:],V,epsilon) for i in range(Nparticles)])
    all_rho = np.array([rho_particles])

    # compute rates
    ratesx = np.array([event_rates(x_allparticles[i,:],rho_particles[i],DV,D2V,a,c,h0) for i in range(Nparticles)])
    ratesy = np.array([event_rates(y_allparticles[i,:],1-rho_particles[i],DV,D2V,a,c,h0) for i in range(Nparticles)])
    clocksx = np.array([np.random.exponential(1/ratesx[i,0]) for i in range(Nparticles)])
    clocksy = np.array([np.random.exponential(1/ratesy[i,0]) for i in range(Nparticles)])

    net_event_ratex = ratesx[:,0]
    net_event_ratey = ratesy[:,0]

    dynamics_ratex = ratesx[:,1]
    dynamics_ratey = ratesy[:,1]

    killclone_ratex = ratesx[:,2]
    killclone_ratey = ratesy[:,2]

    h_stepx = ratesx[:,3]
    h_stepy = ratesy[:,3]
    

    while current_time < T: 
        # find next event
        index = np.argmin(np.append(clocksx,clocksy))
        elapsed_time = np.min(np.append(clocksx,clocksy))
        current_time += elapsed_time

        
        #subtract time from all other clocks
        clocksx = clocksx - elapsed_time
        clocksy = clocksy - elapsed_time

        # figure out if x or y event
        if index < Nparticles:
            particles1 = x_allparticles.copy()
            particles2 = y_allparticles.copy()
            dynamics_rate = dynamics_ratex[index]
            killclone_rate = killclone_ratex[index]
            net_event_rate = net_event_ratex[index]
            h_step = h_stepx[index]
            rho_particles1 = rho_particles
        else: 
            particles1 = y_allparticles.copy()
            particles2 = x_allparticles.copy()
            dynamics_rate = dynamics_ratey[index - Nparticles]
            killclone_rate = killclone_ratey[index - Nparticles]
            net_event_rate = net_event_ratey[index - Nparticles]
            h_step = h_stepy[index - Nparticles]
            rho_particles1 = 1 - rho_particles

        transition_index = np.mod(index,Nparticles)
        
        # regular dynamics or killing-cloning
        if np.random.rand() < np.sum(dynamics_rate)/ net_event_rate : # regular dynamics event
            particles1[transition_index,:] = one_step_sde(particles1[transition_index,:],dynamics_rate,h_step)
            index_to_recompute = [transition_index]
        else: # else killing/cloning event
            particles1, particles2, index_to_recompute = killing_cloning(particles1,particles2,killclone_rate,rho_particles1, transition_index)

        # update particle positions
        if index < Nparticles:
            x_allparticles = particles1.copy()
            y_allparticles = particles2.copy()
        else: 
            y_allparticles = particles1.copy()
            x_allparticles = particles2.copy()

        # update rho and rates only for changed particles
        for i in index_to_recompute:
            i = int(i)

            rho_particles[i] = inf_swap_rate(x_allparticles[i,:],y_allparticles[i,:],V,epsilon)
            ratesx[i,:] = event_rates(x_allparticles[i,:],rho_particles[i],DV,D2V,a,c,h0)
            ratesy[i,:] = event_rates(y_allparticles[i,:],1-rho_particles[i],DV,D2V,a,c,h0)

            # update clock for next event
            clocksx[i] = np.random.exponential(1/ratesx[i,0])
            clocksy[i] = np.random.exponential(1/ratesy[i,0])
        

        net_event_ratex = ratesx[:,0]
        net_event_ratey = ratesy[:,0]

        dynamics_ratex = ratesx[:,1]
        dynamics_ratey = ratesy[:,1]

        killclone_ratex = ratesx[:,2]
        killclone_ratey = ratesy[:,2]

        h_stepx = ratesx[:,3]
        h_stepy = ratesy[:,3]
        
        all_positionsx = np.vstack((all_positionsx,np.array([x_allparticles])))
        all_positionsy = np.vstack((all_positionsy,np.array([y_allparticles])))
        all_time = np.append(all_time,current_time)
        all_rho = np.vstack((all_rho,np.array([rho_particles])))

    return all_positionsx, all_positionsy, all_time, all_rho



def estimator_inf_swap(x_allparticles,y_allparticles,all_rho,all_time,f):
    """
    Computes estimator for given function f on the particle system. """
    fx = f(x_allparticles)
    fy = f(y_allparticles)
    Nparticles = x_allparticles.shape[0]

    estimator = np.sum(fx * all_rho + fy * (1 - all_rho), axis = 0)
    estimator_sum = np.sum(estimator * np.diff(all_time)) / Nparticles / all_time[-1]

    return estimator_sum




## Vanilla Fleming-Viot system


def killing_cloning_vanilla(allparticles,ceval, index): 
    """
    Performs killing-cloning event for particle with index i.  
    
    allparticles : all particle positions, shape (n,d)
    ceval: kill/clone rate for particle i, scalar
    """

    Nparticles = allparticles.shape[0]
    xnow = allparticles[index,:]


    if ceval> 0:  # kill and respawn
        newindex = np.random.choice(Nparticles) # randomly sample from all indices
        xnew = allparticles[newindex,:]; index_to_recompute = np.array([newindex]) # new location is the random particle
        allparticles[index,:] = xnew # assign new location to current particle

    else: # clone and cull
        newindex = np.random.choice(Nparticles) # randomly sample from all indices
        if newindex == index: 
            index_to_recompute = np.array([index])
        else: 
            allparticles[newindex,:] = xnow; index_to_recompute = np.array([index, newindex])

    return allparticles, index_to_recompute

        




def event_rates_vanilla(x,b,a,c,h):
    """ 
    Computes net event rate for particle x with partner pair y.
    
    x : particle 1, shape (d,)
    DV: potential gradient, callable
    a: diffusion coefficient, diagonal components only , callable
    c: kill rate, callable
    h: mean step size of lattice, scalar

    returns net_event_rate for particle x, scalar
            kill_clone_rate for particle x, scalar
            dynamics transition rate for particle x, vector of shape (2*d,)
    """

    h_step = np.random.uniform(h/2,3*h/2)

    ## dynamics rate
    beval = b(x)
    aeval = a(x)
    dynamics_rates = sde_transition_rates(x,beval,aeval,h_step)

    ## kill/clone rate
    ceval = c(x)

    ## net event rate
    net_event_rate = np.sum(dynamics_rates) + abs(ceval)

    return net_event_rate, dynamics_rates, ceval, h_step



def fleming_viot_vanilla(T, b, a, c, h0, initial_positions):
    """
    Simulates n-dimensional vanilla Fleming-Viot particle system with given potential, drift, diffusion and kill/clone functions.
    inputs: 
    - T: time horizon 
    - b: drift
    - a: diffusion coefficient diagonal components only
    - epsilon: temperature
    - c: kill/clone rate
    - h0: mean step size of lattice
    - Nparticles: number of particles
    - initial_positions: initial positions of the particles
    """

    # simulate
    Nparticles = initial_positions.shape[0]
    allparticles = initial_positions
    current_time = 0
    all_time = np.array([current_time])
    all_positions = np.array([initial_positions])



    # compute rates
    rates = np.array([event_rates_vanilla(allparticles[i,:],b,a,c,h0) for i in range(Nparticles)] )
    clocks = np.array([np.random.exponential(1/rates[i,0]) for i in range(Nparticles)])

    net_event_rate = rates[:,0]
    dynamics_rate = rates[:,1]
    killclone_rate = rates[:,2]
    h_step = rates[:,3]
    

    while current_time < T: 
        # find next event
        index = np.argmin(clocks)
        elapsed_time = np.min(clocks)
        current_time += elapsed_time

        
        #subtract time from all other clocks
        clocks = clocks - elapsed_time


        # figure out x or y event
            #figure out regular dynamics rate or killing-cloning rate
        if np.random.rand() < np.sum(dynamics_rate[index])/ net_event_rate[index]: # regular dynamics event
            allparticles[index,:] = one_step_sde(allparticles[index,:],dynamics_rate[index],h_step[index])
            index_to_recompute = [index]
        else: #else killing/cloning event
            allparticles, index_to_recompute = killing_cloning_vanilla(allparticles,killclone_rate[index], index)            

        

        # update rho and rates only for changed particles
        for i in index_to_recompute:
            i = int(i)

            rates[i,:] = event_rates_vanilla(allparticles[i,:],b,a,c,h0)

            # update clock for next event
            clocks[i] = np.random.exponential(1/rates[i,0])
        

        net_event_rate = rates[:,0]
        dynamics_rate = rates[:,1]
        killclone_rate = rates[:,2]
        h_step = rates[:,3]
        
        all_positions = np.vstack((all_positions,np.array([allparticles])))
        all_time = np.append(all_time,current_time)

    return all_positions, all_time