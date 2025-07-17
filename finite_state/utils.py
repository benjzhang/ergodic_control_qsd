import numpy as np

def inf_swap_rate(x, y, V, epsilon):
    return np.exp(2 * V[y]/epsilon) / (np.exp(2 * V[x]/epsilon) + np.exp(2 * V[y]/epsilon))


def forward_rate(V,epsilon):
   # Precompute transition rates
    num_states = V.shape[0]
    r = np.zeros((num_states,num_states))
    for i in range(num_states):
        for j in [i-1, i, i+1]:
            if 0 <= j <= num_states-1 and i != j:
                r[i, j] = min(np.exp(-(V[j] - V[i]) / epsilon),1)
            elif j == -1:
                r[i,num_states-1] = min(np.exp(-(V[num_states-1] - V[i]) / epsilon),1)
            elif j == num_states:
                r[i,0] = min(np.exp(-(V[0] - V[i]) / epsilon),1)
    r = r - np.diag(np.sum(r, axis=1))
    return r


def backward_rate(V,epsilon):
    # precompute transition rates
    num_states = V.shape[0]
    h = np.zeros((num_states,num_states))
    for i in range(num_states):
        for j in [i-1, i, i+1]:
            if 0 <= j <= num_states-1 and i != j:
                h[i, j] = min(np.exp((V[j] - V[i]) / epsilon),1)
            elif j == -1:
                h[i,num_states-1] = min(np.exp((V[num_states-1] - V[i]) / epsilon),1)
            elif j == num_states:
                h[i,0] = min(np.exp((V[0] - V[i]) / epsilon),1)
    h = h - np.diag(np.sum(h, axis=1))
    return h



def sym_dynamics(z, z1, r, h, V, epsilon):
    rho = inf_swap_rate(z[0], z[1], V, epsilon)
    if np.all(z == z1):
        return 0
    if z[1] == z1[1] and z[0] != z1[0]:
        return r[z[0], z1[0]] * rho + h[z[0], z1[0]] * (1-rho)
    if z[0] == z1[0] and z[1] != z1[1]:
        return r[z[1], z1[1]] * (1 - rho) + h[z[1], z1[1]] * rho
    return 0

def lex_order(pairs, n):
    return pairs[:,0] * n + pairs[:,1]

def lex_to_pair(orders, n):
    x = orders // n
    y = orders % n
    return x, y



def construct_sym_rate_matrix(r, h, V, epsilon):
    num_states = V.shape[0]
    sym_rate_matrix = np.zeros((num_states**2, num_states**2))
    for i in range(num_states**2):
        for j in range(num_states**2):
            z = lex_to_pair(i, num_states)
            z1 = lex_to_pair(j, num_states)
            if i != j:
                sym_rate_matrix[i, j] = sym_dynamics(z, z1, r, h, V, epsilon)
    sym_rate_matrix = sym_rate_matrix - np.diag(np.sum(sym_rate_matrix, axis=1))
    return sym_rate_matrix



def event_rate(z, sym_rate_matrix, c_forward,c, V, epsilon):
    z = np.array(z)
    num_states = V.shape[0]
    if z.ndim == 1:  # Single state
        x = z[0]
        y = z[1]
        order = lex_order(np.array([z]),num_states)
    else:  # Multiple states
        x = z[:, 0]
        y = z[:, 1]
        order = lex_order(z,num_states)
    
    rho = inf_swap_rate(x, y, V, epsilon)
    rate = -sym_rate_matrix[order, order] + (abs(c[x.astype(int)]) + abs(c_forward[y.astype(int)]) ) * (1 - rho) + (abs(c[y.astype(int)]) +c_forward[x.astype(int)] )* rho
    return rate

def ordinary_transition(z, sym_rate_matrix):
    num_pairs = sym_rate_matrix.shape[0]
    order = (lex_order(z,np.sqrt(num_pairs))[0]).astype(int)
    probabilities = -sym_rate_matrix[order, :] / sym_rate_matrix[order, order]
    probabilities[order] = 0
    order1 = np.random.choice(num_pairs, 1, p=probabilities)
    z1 = lex_to_pair(order1, np.sqrt(num_pairs))
    return z1




def killing_cloning(indx, states, c_forward,c, V, epsilon):
    Nparticles = states.shape[0]
    x_states = states[:,0].astype(int)
    y_states = states[:,1].astype(int)
    rho_all = inf_swap_rate(x_states, y_states, V, epsilon)

    z = states[indx,:]; x = z[0]; y = z[1]
    rho = rho_all[indx]
    kill_clone_rate = abs(c[x.astype(int)]) * (1 - rho) + abs(c[y.astype(int)]) * rho  + abs(c_forward[x.astype(int)]) * rho + abs(c_forward[y.astype(int)]) * (1 - rho)

    forward_kill_clone = abs(c_forward[x.astype(int)]) * rho + abs(c_forward[y.astype(int)]) * (1 - rho)

    backward_kill_clone = abs(c[x.astype(int)]) * (1 - rho) + abs(c[y.astype(int)]) * rho

    if np.random.rand() < backward_kill_clone / kill_clone_rate:
        if np.random.rand() < abs(c[x.astype(int)]) * (1-rho)/ backward_kill_clone:


            if c[x.astype(int)] > 0:  # Kill and respawn
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    x = z[0]; indices = [indx]
                elif np.random.rand()< (1-rho_all[indz]):
                    x = x_states[indz]; indices = [indx]
                else: x = y_states[indz]; indices = [indx]
                
            else:  # Clone particle
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    states[indx,0] = z[0]; indices = [indx]
                elif np.random.rand()< (1-rho_all[indz]):
                    states[indz,0] = x; indices = [indx,indz]
                else: states[indz,1] = x; indices = [indx,indz]

            
                
        else:
            if c[y.astype(int)] > 0:  # Kill and respawn
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    y = z[1]; indices = [indx]
                elif np.random.rand()< (1-rho_all[indz]):
                    y = x_states[indz]; indices = [indx]
                else: y = y_states[indz]; indices = [indx]
                
            else:  # Clone particle
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    states[indx,1] = y; indices = [indx]
                elif np.random.rand()< (1-rho_all[indz]):
                    states[indz,0] = y; indices = [indx,indz]
                else: states[indz,1] = y; indices = [indx,indz]
    
    else: 
        if np.random.rand() < abs(c_forward[x.astype(int)]) * rho / forward_kill_clone:
            if c_forward[x.astype(int)] > 0:  # Kill and respawn
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    x = z[0]; indices = [indx]
                elif np.random.rand()< rho_all[indz]:
                    x = x_states[indz]; indices = [indx]
                else: x = y_states[indz]; indices = [indx]

                
            else:  # Clone particle
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    states[indx,0] = z[0]; indices = [indx]
                elif np.random.rand()< rho_all[indz]:
                    states[indz,0] = x; indices = [indx,indz]
                else: states[indz,1] = x; indices = [indx,indz]
                
                
        else:
            if c_forward[y.astype(int)] > 0:  # Kill and respawn
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    y = z[1]; indices = [indx]
                elif np.random.rand()< rho_all[indz]:
                    y = x_states[indz]; indices = [indx]
                else: y = y_states[indz]; indices = [indx]

                
            else:  # Clone particle
                indz = np.random.choice(Nparticles)
                if indz == indx:
                    states[indx,1] = y; indices = [indx]
                elif np.random.rand()< rho_all[indz]:
                    states[indz,0] = y; indices = [indx,indz]
                else: states[indz,1] = y; indices = [indx,indz]
    
    z = np.array([x, y])
    states[indx,:] = z
  
    return z, states, indices


    


def vanilla_estimator(x_states, time_vec, g_function):
    Nparticles = x_states.shape[0]
    dt = np.diff(time_vec)
    gx = g_function(x_states)[:,0:-1]
    estimator = np.sum(gx, axis = 0)
    estimator_sum = np.sum(estimator * dt) / Nparticles / time_vec[-1]
    return estimator_sum


def inf_swap_estimator(x_states,y_states, x_pos,time_vec, g_function, V, epsilon):
    trajectories_x= x_pos[x_states]
    trajectories_y = x_pos[y_states]
    Nparticles = x_states.shape[0]
    dt = np.diff(time_vec) 
    inf_swap = inf_swap_rate(x_states.astype(int), y_states.astype(int), V, epsilon)[:,0:-1]
    gx = g_function(trajectories_x)[:,0:-1]
    gy = g_function(trajectories_y)[:,0:-1]

    estimator = np.sum(gx * inf_swap + gy * (1 - inf_swap),axis = 0)
    estimator_sum = np.sum(estimator * dt ) / Nparticles/ time_vec[-1]

    return estimator_sum
