import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import eigsh

# Parameters
epsilon = 0.1
L = 6
N = 1000
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Functions
V = lambda x: np.cos(2 * np.pi * x) / (2 * np.pi)
DV = lambda x: -np.sin(2 * np.pi * x)
c = lambda x: x**2 / 50

# Discrete derivatives (sparse)
D = (eye(N, k=1) - eye(N, k=-1)) / (2 * dx)       # 1st derivative
D2 = (eye(N, k=1) - 2 * eye(N) + eye(N, k=-1)) / dx**2  # 2nd derivative

# Impose Neumann BCs (zero derivative at boundaries)
D = D.tolil()
D2 = D2.tolil()

# First derivative (one-sided at boundaries)
D[0, 0] = -1 / dx
D[0, 1] = 1 / dx
D[-1, -2] = -1 / dx
D[-1, -1] = 1 / dx

# Second derivative (symmetric at boundaries)
D2[0, 0] = 1 / dx**2
D2[0, 1] = -2 / dx**2
D2[0, 2] = 1 / dx**2
D2[-1, -3] = 1 / dx**2
D2[-1, -2] = -2 / dx**2
D2[-1, -1] = 1 / dx**2

D = D.tocsc()
D2 = D2.tocsc()

# Potential and advection
V_p = DV(x)
V_diag = diags(V_p, 0)
c_diag = diags(c(x), 0)

# Construct operator: A = -V'(x) * d/dx - epsilon * d^2/dx^2 + c(x)
A = -V_diag @ D - epsilon * D2 + c_diag
A = csr_matrix(A)  # ensure sparse format

# Solve for eigenvalues near 0
evals, evecs = eigsh(A, k=5, sigma=0, which='LM')

# Sort eigenvalues and eigenvectors
idx = np.argsort(evals)
evals = evals[idx]
evecs = evecs[:, idx]

# Normalize eigenfunction
f0 = np.abs(evecs[:, 0])
f0 /= np.trapz(f0, x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, f0, label="Density eigenfunction (Neumann BCs)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Smallest eigenfunction of the operator with Neumann BCs")
plt.grid(True)
plt.legend()
plt.show()

# Print smallest eigenvalue
print("Smallest eigenvalue near 0:", evals[0])
