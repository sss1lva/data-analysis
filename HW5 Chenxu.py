import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.linalg import solve
from scipy.linalg import lu, solve_triangular
import time 
from matplotlib.animation import FuncAnimation

####### PROBLEM (a): 
####### Solve these equations where for the streamline (∇2ψ = ω) use a Fast Fourier Transform (NOTE: set
####### kx(0) = ky(0) = 10−6

###
# Step 1: Define parameters
###

L = 20  # Domain length [-10, 10]
m = 64  # Grid resolution
n = m * m  # Total number of points
nu = 0.001  # Diffusion coefficient
tspan = (0, 4)  # Time span
t_eval = np.arange(0, 4.5, 0.5)  # Evaluation times

###
# Step 2: Define spatial domain and initial conditions
###

start_time = time.time()

x2 = np.linspace (-L/2, L/2, m+1)
x = x2[:m]
y2 = np.linspace (-L/2, L/2, m+1)
y = y2[:m]
dx = x2[1] - x2[0]
dy = dx  # Assuming dx = dy
X, Y = np.meshgrid (x, y)
omega0 =  np.exp (-X ** 2 - Y ** 2 / 20).flatten ()   # Initial vorticity

###
# Step 3: Define spectral k values
###

kx = (2 * np.pi / L) * np.concatenate ((np.arange (0, m / 2), np.arange (-m / 2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / L) * np.concatenate ((np.arange (0, m / 2), np.arange (-m / 2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid (kx, ky)
K = KX ** 2 + KY ** 2

###
# Step 4: Construct matrices A, B, C
###

e0 = np.zeros((n, 1))  # a vector of zeros
e1 = np.ones((n, 1))   # a vector of ones
e2 = np.copy(e1)        # copy the one vector
e4 = np.copy(e0)        # copy the zero vector

for j in range(1, m + 1):
    e2[m * j - 1] = 0    # overwrite every m-th value with zero
    e4[m * j - 1] = 1    # overwrite every m-th value with one
    
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# A = dx^2 + dy^2  (modified for this part for streamfunction requirement)
diagonalsA = [
    e1.flatten(), e1.flatten(), e5.flatten(),
    e2.flatten(), -4 * e1.flatten(), e3.flatten(),
    e4.flatten(), e1.flatten(), e1.flatten()
]
offsets = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
A = (spdiags(diagonalsA, offsets, n, n).toarray())
A_modified = A
A_modified[0,0] = 2
A_modified = A_modified / dx**2
A = A / dx**2

### B = dx
diagonalsB = [
    e1.flatten(), -1*e1.flatten(), e0.flatten(), e1.flatten(), -1*e1.flatten()
]
offsets = [-(n - m), -m, 0, m, (n - m)]
B = (spdiags(diagonalsB, offsets, n, n).toarray()) / (2 * dx)

# C = dy
diagonalsC = [
    e5.flatten(), -1 * e2.flatten(), e3.flatten(), -1 * e4.flatten()
]
offsets = [-m + 1, -1, 1, m - 1]
C = (spdiags(diagonalsC, offsets, n, n).toarray()) / (2 * dy)

###
# Step 5: Define the PDE with FFTs
###

# Solve ∇²ψ = ω using FFT with np.fft
def solve_streamfunction(omega):
    omega_hat = np.fft.fft2(omega)
    psi_hat = - omega_hat / K
    psi = np.real(np.fft.ifft2(psi_hat))  # Transform back to real space
    return psi

# Compute the advection term [ψ, ω] using sparse matrices B and C.
def compute_advection(psi, omega, B, C):
    dpsi_dx = B.dot(psi)
    dpsi_dy = C.dot(psi)
    domega_dx = B.dot(omega)
    domega_dy = C.dot(omega)
    return dpsi_dx * domega_dy - dpsi_dy * domega_dx

# Compute the RHS of the vorticity equation:
#    ω_t = [ψ, ω] + ν ∇²ω
def vorticity_rhs(t, omega_flat, B, C, A, nu, m, n):
    omega = omega_flat.reshape((m, m))
    psi = solve_streamfunction(omega)
    psi_flat = psi.reshape(n)

    advection = compute_advection(psi_flat, omega_flat, B, C)
    diffusion = nu * A.dot(omega_flat)
    rhs1 = -1 * advection + diffusion

    return rhs1


###
# Step 6: Solve with solve_ivp and show the result
###

solution = solve_ivp(
    vorticity_rhs, tspan, omega0, t_eval=t_eval, args=(B, C, A_modified, nu, m, n), method="RK45"
)

omega_t = solution.y.reshape((m, m, len(t_eval)))
omega = solution.y
A1 = omega
print("Values of A1:", A1)

end_time = time.time()
elapsed_time = end_time - start_time
print("the time it takes to solve this with FFT is :" , elapsed_time , "seconds")

# Plotting results
rows, cols = 3, 3
time_indices = list(range(min(rows * cols, len(t_eval))))
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
fig.suptitle("Vorticity Evolution Using FFT", fontsize=16)

for idx, ax in enumerate(axes.flat):
    if idx < len(time_indices):
        time_idx = time_indices[idx]
        t = t_eval[time_idx]
        
        # Plot the vorticity field
        c = ax.contourf(X, Y, omega_t[:, :, time_idx], levels=50, cmap='viridis', extend='both')
        ax.set_title(f"Time: {t:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Colorbar
        cbar = fig.colorbar(c, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Vorticity')
    else:
        ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()



####### PROBLEM (b): 
####### Solve these equations where for the streamline (∇2ψ = ω) use a Fast Fourier Transform (NOTE: set
####### kx(0) = ky(0) = 10−6

###
# A\b
###

start_time = time.time()

def vorticity_rhs_direct(t, omega_flat, A, B, C, nu, m, n):
    # Reshape omega into 2D
    omega = omega_flat.reshape((m, m))
    
    # Solve ∇²ψ = ω using the direct solver
    psi_flat = solve(A, omega_flat)  # Use the direct solver to compute ψ
    psi = psi_flat.reshape((m, m))  # Reshape back to 2D

    # Compute advection term: -[ψ, ω]
    dpsi_dx = B.dot(psi_flat)
    dpsi_dy = C.dot(psi_flat)
    domega_dx = B.dot(omega_flat)
    domega_dy = C.dot(omega_flat)
    advection = dpsi_dx * domega_dy - dpsi_dy * domega_dx

    # Compute diffusion term: ν ∇²ω
    diffusion = nu * A.dot(omega_flat)

    # Total RHS: ω_t = -[ψ, ω] + ν ∇²ω
    rhs = -advection + diffusion
    return rhs

solution_direct = solve_ivp(
    vorticity_rhs_direct, tspan, omega0, t_eval=t_eval, args=(A_modified, B, C, nu, m, n), method="RK45"
)
omega_t_direct = solution_direct.y.reshape((m, m, len(t_eval)))
omega_direct = solution_direct.y
A2 = omega_direct
print("Values of A2" , A2)

end_time = time.time()
elapsed_time = end_time - start_time
print("the time it takes to solve this with Direct Solve method is :" , elapsed_time , "seconds")


# Plotting results
rows, cols = 3, 3
time_indices = list(range(min(rows * cols, len(t_eval))))
fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
fig.suptitle("Vorticity Evolution Using Direct Solver", fontsize=16)

for idx, ax in enumerate(axes.flat):
    if idx < len(time_indices):
        time_idx = time_indices[idx]
        t = t_eval[time_idx]
        c = ax.contourf(X, Y, omega_t_direct[:, :, time_idx], levels=20, cmap='viridis')
        ax.set_title(f"Time: {t:.1f}", fontsize=10)
        ax.set_xlim([-L / 2, L / 2])
        ax.set_ylim([-L / 2, L / 2])
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(c, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Vorticity', fontsize=8)
    else:
        ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()

###
#LU Decomposition
###

start_time = time.time()

# LU decomposition of A
P, l, U = lu(A)  # Decompose A into P, L, and U matrices

# Function to solve using LU decomposition
def solve_lu_system(A, omega_flat):
    """
    Solves the system Ax = b using LU decomposition, where
    A is decomposed into P, L, and U.
    """
    Pb = np.dot(P, omega_flat)  # Apply permutation matrix
    y = solve_triangular(l, Pb, lower=True)  # Forward substitution
    x = solve_triangular(U, y)  # Back substitution
    return x

# Define RHS using LU for solving streamfunction
def vorticity_rhs_lu(t, omega_flat, B, C, P, l, U, nu, m, n):
    """
    Compute the RHS of the vorticity equation using LU decomposition:
        ω_t = -[ψ, ω] + ν ∇²ω
    """
    # Solve for streamfunction ψ using LU
    psi_flat = solve_lu_system(A, omega_flat)

    # Compute advection and diffusion terms
    dpsi_dx = B.dot(psi_flat)
    dpsi_dy = C.dot(psi_flat)
    domega_dx = B.dot(omega_flat)
    domega_dy = C.dot(omega_flat)

    advection = dpsi_dx * domega_dy - dpsi_dy * domega_dx
    diffusion = nu * A_modified.dot(omega_flat)
    
    # Combine terms
    rhs = -advection + diffusion
    return rhs

# Solve IVP using solve_ivp with LU decomposition
solution_lu = solve_ivp(
    vorticity_rhs_lu, tspan, omega0, t_eval=t_eval, args=(B, C, P, l, U, nu, m, n), method="RK45"
)

# Reshape the solution
omega_t_lu = solution_lu.y.reshape((m, m, len(t_eval)))
omega_lu = solution_lu.y
A3 = omega_lu
print("Values of A3:",A3)

end_time = time.time()
elapsed_time = end_time - start_time
print("the time it takes to solve this with LU Decomposition is :" , elapsed_time , "seconds")

L = 20

# Plotting the results
rows, cols = 3, 3  # Adjust number of subplots
time_indices = list(range(min(rows * cols, len(t_eval))))

fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
fig.suptitle("Vorticity Evolution Over Time (LU Decomposition)", fontsize=16)

for idx, ax in enumerate(axes.flat):
    if idx < len(time_indices):
        time_idx = time_indices[idx]
        t = t_eval[time_idx]
        c = ax.contourf(X, Y, omega_t_lu[:, :, time_idx], levels=50, cmap='viridis')
        ax.set_title(f"Time: {t:.1f}", fontsize=10)
        ax.set_xlim([-float(L) / 2, float(L) / 2])  # Ensure scalar values
        ax.set_ylim([-float(L) / 2, float(L) / 2])  # Ensure scalar values
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(c, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Vorticity', fontsize=8)
    else:
        ax.axis('off')


plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()


####### PROBLEM (c): 
####### Try out these initial conditions with your favorite/fastest solver on the streamfunction equations.
####### • Two oppositely “charged” Gaussian vorticies next to each other, i.e. one with positive amplitude, the
####### other with negative amplitude.
####### • Two same “charged” Gaussian vorticies next to each other.
####### • Two pairs of oppositely “charged” vorticies which can be made to collide with each other.
####### • A random assortment (in position, strength, charge, ellipticity, etc.) of vorticies on the periodic domain.
####### Try 10-15 vorticies and watch what happens.

"""
    My favorite solver on the streamfunction is the LU decomposition. Because it is fastest and stable.
    """



###
# Two oppositely “charged” Gaussian vorticies next to each other, i.e. one with positive amplitude, the
# other with negative amplitude.
###

# Generate the opposite gaussian charging vortices
def generate_opposite_gaussians(X, Y, centers, amplitudes, width=1.0):
    omega = amplitudes[0] * np.exp(-((X - centers[0][0])**2 + (Y - centers[0][1])**2) / width)
    omega += amplitudes[1] * np.exp(-((X - centers[1][0])**2 + (Y - centers[1][1])**2) / width)
    return omega

omega0 = generate_opposite_gaussians(
    X, Y,
    centers=[(-5, 0), (5, 0)],  # Positions of the two vortices
    amplitudes=[1.0, -1.0],  # Positive and negative charges
    width=5.0  # Gaussian width
).flatten()

# Solve using LU decomposition
solution_lu = solve_ivp(
    vorticity_rhs_lu, tspan, omega0, t_eval=t_eval, args=(B, C, P, l, U, nu, m, n), method="RK45"
)
omega_t_lu_oppo = solution_lu.y.reshape((m, m, len(t_eval)))

###
# Two same “charged” Gaussian vorticies next to each other.
###

# Define function for generating two same-charged Gaussian vortices
def generate_same_gaussians(X, Y, centers, amplitude, width=1.0):
    omega = amplitude * np.exp(-((X - centers[0][0])**2 + (Y - centers[0][1])**2) / width)
    omega += amplitude * np.exp(-((X - centers[1][0])**2 + (Y - centers[1][1])**2) / width)
    return omega

# Set initial conditions for two same-charged Gaussian vortices
omega0_same = generate_same_gaussians(
    X, Y,
    centers=[(-5, 0), (5, 0)],  # Positions of the vortices
    amplitude=1.0,  # Same positive charge for both
    width=5.0  # Gaussian width
).flatten()

# Solve using LU decomposition
solution_lu_same = solve_ivp(
    vorticity_rhs_lu, tspan, omega0_same, t_eval=t_eval, args=(B, C, P, l, U, nu, m, n), method="RK45"
)
omega_t_lu_same = solution_lu_same.y.reshape((m, m, len(t_eval)))

###
# Two pairs of oppositely “charged” vorticies which can be made to collide with each other.
###

# Generate multiple pairs of oppositely charged Gaussian vortices.
def generate_multiple_opposite_gaussians(X, Y, centers, amplitudes, width=1.0):
    omega = np.zeros_like(X)
    for i in range(len(centers)):
        omega += amplitudes[i] * np.exp(-((X - centers[i][0])**2 + (Y - centers[i][1])**2) / width)
    return omega

# Define vortex pair centers and amplitudes
centers = [(-5, -5), (5, 5), (-5, 5), (5, -5)]  # Two pairs diagonally placed
amplitudes = [1.0, -1.0, -1.0, 1.0]  # Opposite charges for each pair

# Generate the vorticity field for the two pairs of vortices
omega0 = generate_multiple_opposite_gaussians(
    X, Y, 
    centers=centers, 
    amplitudes=amplitudes, 
    width=5.0
).flatten()

# Solve using LU decomposition
solution_lu = solve_ivp(
    vorticity_rhs_lu, tspan, omega0, t_eval=t_eval, args=(B, C, P, l, U, nu, m, n), method="RK45"
)
omega_t_lu_collision = solution_lu.y.reshape((m, m, len(t_eval)))


###
#A random assortment (in position, strength, charge, ellipticity, etc.) of vorticies on the periodic domain.
#Try 10-15 vorticies and watch what happens.
###

# Generate a random assortment of vortices.
def generate_random_vortices(X, Y, num_vortices=10, domain_length=20, amplitude_range=(-1, 1), width_range=(1, 5)):
    omega = np.zeros_like(X)
    for _ in range(num_vortices):
        # Random vortex parameters
        center_x = np.random.uniform(-domain_length / 2, domain_length / 2)
        center_y = np.random.uniform(-domain_length / 2, domain_length / 2)
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        width_x = np.random.uniform(width_range[0], width_range[1])
        width_y = np.random.uniform(width_range[0], width_range[1])
        
        # Add vortex to the field
        omega += amplitude * np.exp(-(((X - center_x)**2) / width_x + ((Y - center_y)**2) / width_y))
    return omega

# Generate random vortices
omega0 = generate_random_vortices(
    X, Y, 
    num_vortices=15, 
    domain_length=L, 
    amplitude_range=(-1, 1), 
    width_range=(1, 5)
).flatten()

# Solve using LU decomposition
solution_lu = solve_ivp(
    vorticity_rhs_lu, tspan, omega0, t_eval=t_eval, args=(B, C, P, l, U, nu, m, n), method="RK45"
)
omega_t_lu_random = solution_lu.y.reshape((m, m, len(t_eval)))

print("finish")



####### PROBLEM (d): 
####### Make a 2-D movie of the dynamics. Color and coolness are key here.

###
# Altogether
###
def create_combined_vorticity_movie(X, Y, omega_t_list, t_eval, titles, filename="combined_vortices.mp4"):
    """
    Create a combined 2-D animated movie of vorticity dynamics for multiple simulations.
    
    :param X, Y: Meshgrid arrays for spatial coordinates.
    :param omega_t_list: List of 3D arrays of vorticity [m x m x time] for each simulation.
    :param t_eval: Time points for the simulation.
    :param titles: List of titles for each subplot.
    :param filename: Output filename for the movie.
    """
    num_sims = len(omega_t_list)
    rows, cols = 2, 2  # Assuming 4 simulations to fit in a 2x2 grid
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    cbar_list = []
    plots = []

    # Initialize the plots
    for idx, ax in enumerate(axes.flat):
        if idx < num_sims:
            cont = ax.contourf(X, Y, omega_t_list[idx][:, :, 0], levels=50, cmap='plasma', extend='both')
            plots.append(cont)
            cbar = fig.colorbar(cont, ax=ax, shrink=0.8)
            cbar.set_label('Vorticity', fontsize=10)
            cbar_list.append(cbar)
            ax.set_title(titles[idx], fontsize=14)
            ax.set_xlim([-L / 2, L / 2])
            ax.set_ylim([-L / 2, L / 2])
        else:
            ax.axis('off')

    def update(frame):
        for idx, ax in enumerate(axes.flat):
            if idx < num_sims:
                for c in plots[idx].collections:
                    c.remove()
                plots[idx] = ax.contourf(X, Y, omega_t_list[idx][:, :, frame], levels=50, cmap='plasma', extend='both')
                ax.set_title(f"{titles[idx]}\nTime: {t_eval[frame]:.2f}", fontsize=12)
        return plots

    anim = FuncAnimation(fig, update, frames=len(t_eval), blit=False, repeat=False)
    anim.save(filename, writer='ffmpeg', fps=10)
    plt.close(fig)
    print(f"Saved combined animation to {filename}.")

# List of dynamics and titles
omega_t_list = [
    omega_t_lu_oppo,
    omega_t_lu_same,
    omega_t_lu_collision,
    omega_t_lu_random
]
titles = [
    "Oppositely Charged Gaussian Vortices",
    "Same-Charged Gaussian Vortices",
    "Colliding Gaussian Vortices",
    "Random Gaussian Vortices"
]

# Create and save the combined video
create_combined_vorticity_movie(X, Y, omega_t_list, t_eval, titles, filename="combined_vortices.mp4")
