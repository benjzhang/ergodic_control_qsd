import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Parameters
epsilon = 0.125
L = 4
N = 1000
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Potential and drift terms
V_prime = np.sin(2 * np.pi * x)
c_x = (1 / (np.pi)) * x**2

# First and second derivative matrices (central differences)
D1 = sp.diags([-0.5, 0, 0.5], [-1, 0, 1], shape=(N, N)) / dx
D2 = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2

# Apply Neumann boundary conditions (f' = 0 at boundaries)
D1 = D1.tolil()
D1[0, :] = D1[-1, :] = 0
D2 = D2.tolil()
D2[0, :] = D2[-1, :] = 0
D1 = D1.tocsc()
D2 = D2.tocsc()

# Operator L: -V'(x) f' - epsilon f'' + c(x) f
L_op = -sp.diags(V_prime) @ D1 - epsilon * D2 + sp.diags(c_x)

# Compute eigenvalues and eigenvectors
num_eigs = 5
eigs, vecs = spla.eigsh(L_op, k=num_eigs, sigma=0, which='LM')

# Sort eigenvalues and get corresponding eigenvector
idx = np.argsort(eigs)
eigs = eigs[idx]
vecs = vecs[:, idx]

# Normalize the eigenfunction corresponding to the smallest eigenvalue
f0 = vecs[:, 0]
f0 = np.abs(f0)
f0 /= np.trapz(f0, x)

eigs[0]  # eigenvalue closest to zero
# Plot the eigenfunction
plt.figure(figsize=(8, 5))
plt.plot(x, f0, label="Density eigenfunction")
plt.xlabel("x")
plt.ylabel(r"$\phi^{\varepsilon}(x)$")
plt.title("Eigenfunction corresponding to the smallest eigenvalue")
plt.grid(True)
# plt.legend()
plt.show()