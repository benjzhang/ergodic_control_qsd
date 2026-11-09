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
    Computes the forward and backward kill/clone rates for the active replica x.

    The symmetrized kill/clone rate c(x) - (1-rhoeps)*D2V(x) is split into two
    signed channels:
        r_kf = rhoeps * c(x)                     (forward channel)
        r_kb = (1 - rhoeps) * (c(x) - D2V(x))    (backward channel)
    so that r_kf + r_kb = c(x) - (1-rhoeps)*D2V(x). Each channel fires
    independently at rate |r_k.|; its sign selects kill (>0) vs clone (<0).

    x : particle 1, shape (d,)
    rhoeps: infinite swap fraction recomputed with partner particle y, scalar
    D2V: potential laplacian, callable
    c: kill rate, callable

    returns forward and backward kill/clone rates r_kf, r_kb for particle x
    """
    cx = c(x)
    r_kf = rhoeps * cx
    r_kb = (1 - rhoeps) * (cx - D2V(x))

    return r_kf, r_kb


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
            dynamics transition rate for particle x, vector of shape (2*d,)
            forward kill/clone rate r_kf for particle x, scalar
            backward kill/clone rate r_kb for particle x, scalar
            h_step, scalar
    """

    h_step = np.random.uniform(h/2,3*h/2)

    ## dynamics rate
    bsymm = symmetrized_dynamics(x,rhoeps,DV)
    aeval = a(x)
    dynamics_rates = sde_transition_rates(x,bsymm,aeval,h_step)

    ## forward/backward kill/clone rates (two independent signed channels)
    r_kf, r_kb = symmetrized_kill_clone_rate(x,rhoeps,D2V,c)

    ## net event rate
    net_event_rate = np.sum(dynamics_rates) + abs(r_kf) + abs(r_kb)

    return net_event_rate, dynamics_rates, r_kf, r_kb, h_step



###

### Killing-cloning functions

def killing_cloning(x_allparticles,y_allparticles,rho_all,index,rate):
    """
    Performs a single-channel killing/cloning event for the active replica x^index.

    This is called once per fired kill/clone channel (forward or backward). The
    sign of `rate` selects kill (>0) vs clone (<0); `rho_all` is the swap-weight
    array of the channel's own marginal (rho for the forward channel, 1-rho for
    the backward channel), used to pick which replica at the donor index `i'`.

    x_allparticles : all particle positions, shape (n,d)
    y_allparticles : all particle positions, shape (n,d)
    rho_all : per-particle swap weights for this channel's marginal, shape (n,)
    index : active particle index i
    rate : signed kill/clone rate of this channel (scalar)
    """

    Nparticles = x_allparticles.shape[0]
    xnow = x_allparticles[index,:].copy()
    newindex = np.random.choice(Nparticles) # randomly sample from all indices

    if newindex == index:
        index_to_recompute = np.array([index]) # self-selection: do not move

    elif rate > 0:  # kill and respawn from this channel's marginal
        if np.random.rand() < rho_all[newindex]:
            x_allparticles[index,:] = x_allparticles[newindex,:].copy()
        else:
            x_allparticles[index,:] = y_allparticles[newindex,:].copy()
        index_to_recompute = np.array([index])

    else:  # clone into this channel's marginal at the donor index
        if np.random.rand() < rho_all[newindex]:
            x_allparticles[newindex,:] = xnow
        else:
            y_allparticles[newindex,:] = xnow
        index_to_recompute = np.array([index, newindex])

    return x_allparticles, y_allparticles, index_to_recompute



## Fleming-Viot particle system

def fleming_viot(T, V, DV, D2V, a, epsilon, c, h0, initial_positionsx, initial_positionsy, wrap_period=None):
    """
    Simulates n-dimensional Fleming-Viot particle system with given potential, drift, diffusion and kill/clone functions.
    This method implements infinite swapping.

    Parameters:
    - T: time horizon
    - V: potential function
    - DV: potential gradient
    - D2V: potential laplacian
    - a: diffusion coefficient diagonal components only
    - epsilon: temperature
    - c: kill rate (use c(x)=0 for pure sampling, non-zero for eigenfunction problems)
    - h0: mean step size of lattice
    - initial_positionsx: initial x positions of the particles
    - initial_positionsy: initial y positions of the particles
    - wrap_period: if not None, wrap positions to [-wrap_period/2, wrap_period/2]
                   Use for periodic BCs when c(x)=0 (e.g., wrap_period=1.0)
                   Do NOT use when c(x) depends on position (e.g., quadratic killing)
    """
        
    # simulate
    Nparticles = initial_positionsx.shape[0]
    x_allparticles = initial_positionsx.copy()
    y_allparticles = initial_positionsy.copy()
    current_time = 0
    # Accumulate per-event snapshots in lists and stack once at the end.
    # (vstack-per-event was O(n^2) in the number of events.)
    all_time = [current_time]
    all_positionsx = [initial_positionsx.copy()]
    all_positionsy = [initial_positionsy.copy()]


    rho_particles = np.array([inf_swap_rate(x_allparticles[i,:],y_allparticles[i,:],V,epsilon) for i in range(Nparticles)])
    all_rho = [rho_particles.copy()]

    # compute rates
    # Use dtype=object because event_rates returns mixed types (scalars and arrays)
    ratesx = np.array([event_rates(x_allparticles[i,:],rho_particles[i],DV,D2V,a,c,h0) for i in range(Nparticles)], dtype=object)
    ratesy = np.array([event_rates(y_allparticles[i,:],1-rho_particles[i],DV,D2V,a,c,h0) for i in range(Nparticles)], dtype=object)
    clocksx = np.array([np.random.exponential(1/ratesx[i,0]) for i in range(Nparticles)])
    clocksy = np.array([np.random.exponential(1/ratesy[i,0]) for i in range(Nparticles)])

    net_event_ratex = ratesx[:,0]
    net_event_ratey = ratesy[:,0]

    dynamics_ratex = ratesx[:,1]
    dynamics_ratey = ratesy[:,1]

    rkf_x = ratesx[:,2]
    rkf_y = ratesy[:,2]

    rkb_x = ratesx[:,3]
    rkb_y = ratesy[:,3]

    h_stepx = ratesx[:,4]
    h_stepy = ratesy[:,4]


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
            rkf = rkf_x[index]
            rkb = rkb_x[index]
            net_event_rate = net_event_ratex[index]
            h_step = h_stepx[index]
            rho_particles1 = rho_particles
        else:
            particles1 = y_allparticles.copy()
            particles2 = x_allparticles.copy()
            dynamics_rate = dynamics_ratey[index - Nparticles]
            rkf = rkf_y[index - Nparticles]
            rkb = rkb_y[index - Nparticles]
            net_event_rate = net_event_ratey[index - Nparticles]
            h_step = h_stepy[index - Nparticles]
            rho_particles1 = 1 - rho_particles

        transition_index = np.mod(index,Nparticles)
        
        # regular dynamics, forward kill/clone, or backward kill/clone
        a_draw = np.random.rand()
        dyn_frac = np.sum(dynamics_rate) / net_event_rate
        if a_draw < dyn_frac: # regular dynamics event
            particles1[transition_index,:] = one_step_sde(particles1[transition_index,:], dynamics_rate, h_step, wrap_period)
            index_to_recompute = [transition_index]
        elif a_draw < dyn_frac + abs(rkf)/net_event_rate: # forward kill/clone channel
            particles1, particles2, index_to_recompute = killing_cloning(particles1,particles2,rho_particles1, transition_index, rkf)
        else: # backward kill/clone channel
            particles1, particles2, index_to_recompute = killing_cloning(particles1,particles2,1-rho_particles1, transition_index, rkb)

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

        rkf_x = ratesx[:,2]
        rkf_y = ratesy[:,2]

        rkb_x = ratesx[:,3]
        rkb_y = ratesy[:,3]

        h_stepx = ratesx[:,4]
        h_stepy = ratesy[:,4]

        all_positionsx.append(x_allparticles.copy())
        all_positionsy.append(y_allparticles.copy())
        all_time.append(current_time)
        all_rho.append(rho_particles.copy())

    return np.array(all_positionsx), np.array(all_positionsy), np.array(all_time), np.array(all_rho)



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



def fleming_viot_vanilla(T, b, a, c, h0, initial_positions, wrap_period=None):
    """
    Simulates n-dimensional vanilla Fleming-Viot particle system.

    Parameters:
    - T: time horizon
    - b: drift function
    - a: diffusion coefficient diagonal components only
    - c: kill/clone rate function
    - h0: mean step size of lattice
    - initial_positions: initial positions of the particles
    - wrap_period: if not None, wrap positions to [-wrap_period/2, wrap_period/2]
                   Use for periodic BCs when c(x)=0. Do NOT use with position-dependent killing.
    """

    # simulate
    Nparticles = initial_positions.shape[0]
    allparticles = initial_positions
    current_time = 0
    all_time_list = [current_time]
    all_positions_list = [initial_positions.copy()]   # O(n) accumulation (avoid O(n^2) vstack)



    # compute rates
    # Use dtype=object because event_rates_vanilla returns mixed types (scalars and arrays)
    rates = np.array([event_rates_vanilla(allparticles[i,:],b,a,c,h0) for i in range(Nparticles)], dtype=object)
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
            allparticles[index,:] = one_step_sde(allparticles[index,:], dynamics_rate[index], h_step[index], wrap_period)
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
        
        all_positions_list.append(allparticles.copy())
        all_time_list.append(current_time)

    return np.array(all_positions_list), np.array(all_time_list)