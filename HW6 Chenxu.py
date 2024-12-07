import numpy as np
from scipy.integrate import solve_ivp
from numpy.fft import fft2, ifft2
from numpy import tanh
import matplotlib.pyplot as plt

####
# A1
####

# Define parameters
L = 20  # Spatial domain [-L/2, L/2]
n = 64  # Number of grid points
beta = 1
D1, D2 = 0.1, 0.1  # Diffusion coefficients
t_span = np.arange(0, 4.5, 0.5)  # Time range

# Create spatial grid
x = np.linspace(-L / 2, L / 2, n, endpoint=False)
y = np.linspace(-L / 2, L / 2, n, endpoint=False)
dx = L / n
dy = L / n
X, Y = np.meshgrid(x, y)

# Initial conditions
m = 1  # Number of spirals
angle = np.angle(X + 1j * Y)  # Compute the angle for each grid point
radius = np.sqrt(X**2 + Y**2)  # Compute the radius for each grid point
U0 = tanh(radius) * np.cos(m * angle - radius)  # Initial condition for U
V0 = tanh(radius) * np.sin(m * angle - radius)  # Initial condition for V

# Reaction term lambda(A)
def lambda_A(U, V):
    A2 = np.abs(U)**2 + np.abs(V)**2  # Compute A^2
    return 1 - A2

# Reaction term omega(A)
def omega_A(U, V):
    A2 = np.abs(U)**2 + np.abs(V)**2  # Compute A^2
    return -beta * A2

# Define the right-hand side of the equations
def rhs(t, Z):
    # Extract U and V in the frequency domain from Z
    U_hat = Z[:n**2].reshape((n, n))
    V_hat = Z[n**2:].reshape((n, n))

    # Transform back to the spatial domain
    U = ifft2(U_hat)
    V = ifft2(V_hat)

    # Compute reaction terms
    dUdt_reaction = lambda_A(U, V) * U - omega_A(U, V) * V
    dVdt_reaction = omega_A(U, V) * U + lambda_A(U, V) * V

    # Transform reaction terms back to the frequency domain
    dUdt_reaction_hat = fft2(dUdt_reaction)
    dVdt_reaction_hat = fft2(dVdt_reaction)

    # Compute the Laplacian operator in the frequency domain
    kx = 2 * np.pi * np.fft.fftfreq(U.shape[1], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(U.shape[0], d=dy)
    KX, KY = np.meshgrid(kx, ky)
    laplacian_operator = -(KX**2 + KY**2)

    # Compute diffusion terms in the frequency domain
    dUdt_diffusion_hat = D1 * laplacian_operator * U_hat
    dVdt_diffusion_hat = D2 * laplacian_operator * V_hat

    # Combine reaction and diffusion terms
    dUdt_hat = dUdt_reaction_hat + dUdt_diffusion_hat
    dVdt_hat = dVdt_reaction_hat + dVdt_diffusion_hat

    # Concatenate results and return
    rhs = np.concatenate([dUdt_hat.flatten(), dVdt_hat.flatten()])
    return rhs

# Initialize the system (keep in the frequency domain)
Z0 = np.concatenate([fft2(U0).flatten(), fft2(V0).flatten()])

# Solve the system numerically
sol = solve_ivp(
    rhs,  # Right-hand side of the equations
    (t_span[0], t_span[-1]),  # Time range
    Z0,  # Initial conditions
    t_eval=t_span,  # Time points to evaluate
    method='RK45'  # Integration method
)

A1 = sol.y  # Save the solution as A1

# Plotting A1 over time with a different colormap
fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # Create a 3x3 grid for subplots

# Create grid for spatial domain
X, Y = np.meshgrid(x, y)

for j, t in enumerate(t_span):
    # Extract U_hat and V_hat from the solution
    U_hat = A1[:n**2, j].reshape((n, n))
    V_hat = A1[n**2:, j].reshape((n, n))
    
    # Transform back to spatial domain
    U = np.real(ifft2(U_hat))  # Use real part as ifft2 can return small imaginary parts
    V = np.real(ifft2(V_hat))
    
    ax = axes[j // 3, j % 3]  # Determine subplot position
    c = ax.pcolor(X, Y, U, cmap="plasma", shading='auto')  # Change colormap here
    ax.set_title(f"Time: {t:.2f}")
    fig.colorbar(c, ax=ax)

plt.tight_layout()
plt.show()

####
# A2
####

def Chebyshev(N):
    if N == 0:
        D = 0.
        x = 1.
    else:
        n = np.arange(0, N + 1)
        x = np.cos(np.pi * n / N).reshape(N + 1, 1)
        c = (np.hstack(([2.], np.ones(N - 1), [2.])) * (-1) ** n).reshape(N + 1, 1)
        X = np.tile(x, (1, N + 1))
        dX = X - X.T
        D = np.dot(c, 1. / c.T) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N + 1)

# Generate Chebyshev grid and differentiation matrix
n = 30  # Number of Chebyshev points
n2 = (n + 1) ** 2
D, x = Chebyshev(n)
D[n, :] = 0  # Set boundary conditions for Chebyshev differentiation matrix
D[0, :] = 0
Dxx = np.dot(D, D) / ((L / 2) ** 2)  # 2nd derivative approximation
y = x  # Use the same grid for both x and y directions

I = np.eye(len(Dxx))
L = np.kron(I, Dxx) + np.kron(Dxx, I)  # 2D Laplacian using Kronecker product

X, Y = np.meshgrid(x, y)
X = X * 10
Y = Y * 10

# Define initial condition for the grid
angle = np.angle(X + 1j * Y)
radius = np.sqrt(X ** 2 + Y ** 2)
U0 = np.tanh(radius) * np.cos(m * angle - radius)
V0 = np.tanh(radius) * np.sin(m * angle - radius)


# Define rhs for the Chebyshev problem
def rhs(t, uv_t):
    n_rhs = n + 1

    ut, vt = uv_t[:n_rhs ** 2], uv_t[n_rhs ** 2:]

    # Reaction terms and diffusion
    dUdt = (lambda_A(ut, vt) * ut - omega_A(ut, vt) * vt) + D1 * (L @ ut)
    dVdt = (omega_A(ut, vt) * ut + lambda_A(ut, vt) * vt) + D2 * (L @ vt)

    return np.concatenate([dUdt, dVdt])

# Initial conditions for the Chebyshev problem
init = np.concatenate([U0.reshape(n2), V0.reshape(n2)])

# Solve the system using solve_ivp
sol = solve_ivp(rhs, (t_span[0], t_span[-1]), init, t_eval=t_span, method='RK45')

# Print solution information
A2 = sol.y
print(A2)

# Reshape `x` into a 2D grid for plotting
n = int(np.sqrt(A2.shape[0] // 2) - 1)  # Assuming Chebyshev points match A2 dimensions
x = np.cos(np.pi * np.arange(0, n + 1) / n)
y = x
X, Y = np.meshgrid(x, y)

# Plot the results for each time step
fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # Create a grid for subplots

for j, t in enumerate(t_span):
    U_hat = A2[: (n + 1) ** 2, j]  # Extract U_hat for time step j
    U = U_hat.reshape((n + 1, n + 1))  # Reshape U to 2D grid
    
    ax = axes[j // 3, j % 3]  # Determine subplot position
    c = ax.pcolor(X, Y, U, cmap="RdBu_r", shading='auto')  # Plot U
    ax.set_title(f"Time: {t:.2f}")
    fig.colorbar(c, ax=ax)

plt.tight_layout()
plt.show()
