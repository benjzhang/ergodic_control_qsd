import numpy as np

### SDE discretizations through jump processes


def sde_transition_rates(x,beval,aeval,h): 
    """
    Computes rates for jump process given drift and diffusion functions

    x : particle positions, shape (d,)
    b: drift function evaluation, shape (d,)
    a: diagonal function evaluation, shape (d,)
    h: mean step size of lattice, scalar

    returns vector of rates for jump process, shape (2*d,) first d entries are rates for positive jumps, last d entries are rates for negative jumps in the canonical basis
    """

    rates_plus = (h * beval + aeval) / (2 * h**2)
    rates_minus = (-h * beval + aeval) / (2 * h**2)
    if np.any(rates_plus < 0) or np.any(rates_minus < 0):
        rates_plus = (h * np.maximum(beval,0) + aeval /2 ) / (h**2)
        rates_minus = (h * np.maximum(-beval,0) + aeval /2 ) / (h**2)

    rates = np.append(rates_plus,rates_minus)
    return rates

def one_step_sde(x, rates, h, wrap_period=None):
    """
    Performs one step of the jump process given current position and rates.

    Parameters:
    - x: particle positions, shape (d,)
    - rates: rates for jump process, shape (2*d,)
    - h: step size
    - wrap_period: if not None, wrap positions to [-wrap_period/2, wrap_period/2]

    Returns:
    - new_position: updated position, shape (d,)
    """
    d = len(x)
    total_rate = np.sum(rates)
    p = rates / total_rate
    jump_direction = np.random.choice(np.arange(np.size(p)), p=p)
    new_position = x.copy()
    if jump_direction < d:
        new_position[jump_direction] += h
    else:
        new_position[jump_direction - d] -= h

    # Apply periodic boundary conditions if specified
    if wrap_period is not None:
        new_position = np.mod(new_position + wrap_period/2, wrap_period) - wrap_period/2

    return new_position


def pure_jump_approx_diffusion(T, b, a, h0, initial_position, wrap_period=None):
    """Simulate n-dimension diffusion process via pure jump process on a lattice.

    Parameters:
    - T: time horizon
    - b: drift function
    - a: diffusion function
    - h0: mean step size of lattice
    - initial_position: initial position of the process
    - wrap_period: if not None, wrap positions to [-wrap_period/2, wrap_period/2]
                   Use this for periodic boundary conditions (e.g., wrap_period=1.0)

    Returns:
    - all_positions: array of positions at each time step
    - all_time: array of time points
    """

    d = len(initial_position)  # dimension of the process
    current_time = 0
    all_time = np.array([current_time])

    all_positions = np.array([initial_position])

    while current_time < T:
        h = np.random.uniform(h0/2, 3*h0/2)
        beval = b(all_positions[-1, :])

        aeval = a(all_positions[-1, :])

        rates = sde_transition_rates(all_positions[-1, :], beval, aeval, h)
        total_rate = np.sum(rates)

        waiting_time = np.random.exponential(1/total_rate)
        current_time += waiting_time
        if current_time >= T:
            break

        p = rates / total_rate
        jump_direction = np.random.choice(np.arange(np.size(p)), p=p)
        new_position = all_positions[-1, :].copy()
        if jump_direction < d:
            new_position[jump_direction] += h
        else:
            new_position[jump_direction - d] -= h

        # Apply periodic boundary conditions if specified
        if wrap_period is not None:
            new_position = np.mod(new_position + wrap_period/2, wrap_period) - wrap_period/2

        all_positions = np.vstack((all_positions, new_position))
        all_time = np.append(all_time, current_time)
    return all_positions, all_time