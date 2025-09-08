"""
Postprocessing and Measure Functional Utilities

This module contains functions for computing various measure functionals and 
postprocessing operations on particle systems, including empirical measure 
computations, weighted empirical measures, and resampling procedures.

Functions:
---------
- empirical_measure_functional: Compute expectation of f with respect to empirical measure
- cumulative_empirical_measure_functional: Compute cumulative mean of empirical measure functional
- weighted_empirical_measure_functional: Compute expectation with weighted empirical measure
- cumulative_mean_weighted_empirical_measure: Compute cumulative mean of weighted empirical measure
- resample_simple_weighted_empirical_measure: Resample particle path by time weights
- resample_weighted_empirical_measure: Resample particle path by weighted empirical measure
"""

import numpy as np


def empirical_measure_functional(particle_path, f, all_time): 
    """
    Computes expectation of f with respect to empirical measure of one particle path.
    
    This function calculates the time-averaged value of a function f evaluated 
    along a particle trajectory, weighted by the time spent in each state.
    
    Parameters
    ----------
    particle_path : array_like, shape (n_steps, d)
        Particle trajectory over time, where n_steps is the number of time steps
        and d is the spatial dimension
    f : callable
        Function to evaluate along the particle path. Should accept particle_path
        as input and return array of shape (n_steps,) or (n_steps, k) for 
        vector-valued functions
    all_time : array_like, shape (n_steps,)
        Time points corresponding to each step in the particle path
        
    Returns
    -------
    float or array_like
        Time-averaged expectation of f with respect to the empirical measure.
        Returns a scalar for scalar-valued f, or array for vector-valued f
        
    Examples
    --------
    >>> import numpy as np
    >>> # Simple 1D particle path
    >>> path = np.array([[0], [1], [2], [1], [0]])
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> f = lambda x: x**2  # quadratic function
    >>> result = empirical_measure_functional(path, f, times)
    """
    dt = np.append(np.diff(all_time), 0)
    feval = f(particle_path).squeeze()
    estimator = np.sum(dt * feval, axis=0) / np.sum(dt)
    return estimator


def cumulative_empirical_measure_functional(particle_path, f, all_time):
    """
    Computes the cumulative mean of f with respect to the empirical measure of one particle path.
    
    This function computes the running time average of a function f evaluated along
    a particle trajectory. At each time point, it returns the cumulative average
    up to that point in time.
    
    Parameters
    ----------
    particle_path : array_like, shape (n_steps, d)
        Particle trajectory over time, where n_steps is the number of time steps
        and d is the spatial dimension
    f : callable
        Function to evaluate along the particle path. Should accept particle_path
        as input and return array of shape (n_steps,) or (n_steps, k) for 
        vector-valued functions
    all_time : array_like, shape (n_steps,)
        Time points corresponding to each step in the particle path
        
    Returns
    -------
    array_like, shape (n_steps,) or (n_steps, k)
        Cumulative time-averaged expectation of f with respect to the empirical 
        measure at each time point. For vector-valued f, returns array of shape 
        (n_steps, k) where k is the output dimension of f
        
    Examples
    --------
    >>> import numpy as np
    >>> # Simple 1D particle path  
    >>> path = np.array([[0], [1], [2], [1], [0]])
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> f = lambda x: x**2
    >>> cumulative_mean = cumulative_empirical_measure_functional(path, f, times)
    >>> # cumulative_mean[i] gives the average of f up to time times[i]
    """
    dt = np.append(np.diff(all_time), 0)
    feval = f(particle_path).squeeze()
    cumulative_sum = np.cumsum(dt * feval)
    cumulative_time = np.cumsum(dt)
    cumulative_mean = cumulative_sum / cumulative_time
    return cumulative_mean


def weighted_empirical_measure_functional(x_particle, y_particle, rho, f, all_time): 
    """
    Computes expectation of f with respect to empirical measure of coupled particles.
    
    This function evaluates a function f on coupled particle trajectories and computes
    the time-averaged expectation using infinite swapping weights rho. The expectation
    is computed as a weighted average of f evaluated on forward and backward trajectories.
    
    Parameters
    ----------
    x_particle : array_like, shape (n_steps, d)
        Forward particle trajectory over time
    y_particle : array_like, shape (n_steps, d)  
        Backward particle trajectory over time
    rho : array_like, shape (n_steps,)
        Infinite swapping weights at each time step, typically in [0,1]
    f : callable
        Function to evaluate on particle pairs. Should accept two arguments
        (forward_particle, backward_particle) and return array of shape (n_steps,)
        or (n_steps, k) for vector-valued functions
    all_time : array_like, shape (n_steps,)
        Time points corresponding to each step in the particle trajectories
        
    Returns
    -------
    float or array_like
        Weighted time-averaged expectation of f with respect to the coupled 
        empirical measure. Returns scalar for scalar-valued f, or array for 
        vector-valued f
        
    Notes
    -----
    The weighted expectation is computed as:
    E[f] = ∫ [f(x,y) * ρ + f(y,x) * (1-ρ)] dt / ∫ dt
    
    where ρ represents the infinite swapping fraction between forward and 
    backward processes.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Coupled particle paths
    >>> x_path = np.array([[0, 0], [1, 1], [2, 0]])
    >>> y_path = np.array([[1, 1], [0, 0], [1, 1]]) 
    >>> rho = np.array([0.3, 0.7, 0.5])
    >>> times = np.array([0, 1, 2])
    >>> f = lambda x, y: np.sum((x - y)**2, axis=1)  # squared distance
    >>> result = weighted_empirical_measure_functional(x_path, y_path, rho, f, times)
    """
    dt = np.append(np.diff(all_time), 0)
    feval_fwd = f(x_particle, y_particle).squeeze()
    feval_bwd = f(y_particle, x_particle).squeeze()
    estimator = np.sum(dt * (feval_fwd * rho + feval_bwd * (1 - rho)), axis=0) / np.sum(dt)
    return estimator


def cumulative_mean_weighted_empirical_measure(x_particle, y_particle, rho, f, all_time):
    """
    Computes the cumulative mean of f with respect to the weighted empirical measure of coupled particles.
    
    This function computes the running weighted time average of a function f evaluated
    on coupled particle trajectories. At each time point, it returns the cumulative 
    weighted average up to that point in time.
    
    Parameters
    ----------
    x_particle : array_like, shape (n_steps, d)
        Forward particle trajectory over time
    y_particle : array_like, shape (n_steps, d)
        Backward particle trajectory over time  
    rho : array_like, shape (n_steps,)
        Infinite swapping weights at each time step, typically in [0,1]
    f : callable
        Function to evaluate on particle pairs. Should accept two arguments
        (forward_particle, backward_particle) and return array of shape (n_steps,)
        or (n_steps, k) for vector-valued functions
    all_time : array_like, shape (n_steps,)
        Time points corresponding to each step in the particle trajectories
        
    Returns
    -------
    array_like, shape (n_steps,) or (n_steps, k)
        Cumulative weighted time-averaged expectation of f with respect to the 
        coupled empirical measure at each time point. For vector-valued f, returns 
        array of shape (n_steps, k) where k is the output dimension of f
        
    Notes
    -----
    The cumulative weighted expectation at time t is computed as:
    E[f](t) = (∫₀ᵗ [f(x,y) * ρ + f(y,x) * (1-ρ)] dt) / (∫₀ᵗ dt)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Coupled particle paths
    >>> x_path = np.array([[0, 0], [1, 1], [2, 0]])
    >>> y_path = np.array([[1, 1], [0, 0], [1, 1]])
    >>> rho = np.array([0.3, 0.7, 0.5])
    >>> times = np.array([0, 1, 2])
    >>> f = lambda x, y: np.sum((x - y)**2, axis=1)  # squared distance
    >>> cumulative_mean = cumulative_mean_weighted_empirical_measure(x_path, y_path, rho, f, times)
    >>> # cumulative_mean[i] gives the weighted average of f up to time times[i]
    """
    dt = np.append(np.diff(all_time), 0)
    feval_fwd = f(x_particle, y_particle).squeeze()
    feval_bwd = f(y_particle, x_particle).squeeze()
    weighted_values = dt * (feval_fwd * rho + feval_bwd * (1 - rho))
    cumulative_sum = np.cumsum(weighted_values)
    cumulative_time = np.cumsum(dt)
    cumulative_mean = cumulative_sum / cumulative_time
    return cumulative_mean


def resample_simple_weighted_empirical_measure(x_particle, all_time):
    """
    Resamples empirical measure of particle path with respect to time-weighted empirical measure.
    
    This function resamples points from a particle trajectory using weights proportional
    to the time spent in each state. This is useful for converting continuous-time 
    trajectories to discrete samples that preserve the time-weighted distribution.
    
    Parameters
    ----------
    x_particle : array_like, shape (n_steps, d)
        Particle trajectory over time, where n_steps is the number of time steps
        and d is the spatial dimension
    all_time : array_like, shape (n_steps,)
        Time points corresponding to each step in the particle path
        
    Returns
    -------
    array_like, shape (n_steps, d)
        Resampled particle positions drawn according to time-weighted empirical measure.
        The number of samples equals the number of original time steps.
        
    Notes
    -----
    The resampling weights are proportional to dt[i] = time[i+1] - time[i], representing
    the time spent in state x_particle[i]. States with longer residence times have
    higher probability of being selected.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Simple 1D particle path with varying time steps
    >>> path = np.array([[0], [1], [2], [1], [0]])
    >>> times = np.array([0, 0.5, 1.0, 2.0, 3.0])  # longer time step at position [1]
    >>> resampled = resample_simple_weighted_empirical_measure(path, times)
    >>> # Position [1] more likely to appear in resampled due to longer residence time
    """
    dt = np.append(np.diff(all_time), 0)
    weights = dt
    weights = weights / np.sum(weights)
    samples_index = np.random.choice(np.arange(len(weights)), size=len(weights), p=weights)
    samples = x_particle[samples_index]
    return samples


def resample_weighted_empirical_measure(x_particle, rho, all_time):
    """
    Resamples empirical measure of particle path with respect to weighted empirical measure.
    
    This function resamples points from a particle trajectory using weights that combine
    both time weighting and infinite swapping weights rho. This is particularly useful
    for sampling from the forward component of a coupled particle system.
    
    Parameters
    ----------
    x_particle : array_like, shape (n_steps, d)
        Particle trajectory over time, where n_steps is the number of time steps
        and d is the spatial dimension
    rho : array_like, shape (n_steps,)
        Infinite swapping weights at each time step, typically in [0,1]
    all_time : array_like, shape (n_steps,)
        Time points corresponding to each step in the particle path
        
    Returns
    -------
    array_like, shape (n_steps, d)
        Resampled particle positions drawn according to the weighted empirical measure.
        The number of samples equals the number of original time steps.
        
    Notes
    -----
    The resampling weights are proportional to dt[i] * rho[i], where dt[i] represents
    the time spent in state i and rho[i] represents the infinite swapping weight.
    This gives higher probability to states with both longer residence times and
    higher swapping weights.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Particle path with swapping weights
    >>> path = np.array([[0], [1], [2], [1], [0]])
    >>> rho = np.array([0.1, 0.9, 0.5, 0.8, 0.2])  # high weight at position [1]
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> resampled = resample_weighted_empirical_measure(path, rho, times)
    >>> # Position [1] more likely to appear due to high rho value
    """
    dt = np.append(np.diff(all_time), 0)
    weights = dt * rho
    weights = weights / np.sum(weights)
    samples_index = np.random.choice(np.arange(len(weights)), size=len(weights), p=weights)
    samples = x_particle[samples_index]
    return samples