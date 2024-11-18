import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp, simpson
from scipy.sparse.linalg import eigs as eigh
import math


################# (a)

def shooting_func(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tolerance = 1e-6
col = ['r', 'b', 'g', 'c', 'm']
L = 4
xspan = np.arange(-L, L + 0.1, 0.1) 
EigenF = np.zeros((len(xspan), 5))
EigenV = np.zeros(5)

epsilon_start = 0.1
for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = 0.2
    
    for j in range(1000):
        Y0 = [1, np.sqrt(L**2 - epsilon)]
        ### solve the problem with ivp ###
        sol = solve_ivp(shooting_func, [xspan[0], xspan[-1]], Y0, args=(epsilon,), t_eval=xspan)

        if abs(sol.y[1, -1] + np.sqrt(L**2 - epsilon) * sol.y[0, -1]) < tolerance:
            break

        if ((-1) ** (modes + 1) * (sol.y[1, -1] + np.sqrt(L**2 - epsilon) * sol.y[0, -1])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2
         
    EigenV[modes - 1] = epsilon
    epsilon_start = epsilon + 0.1
    norm = np.trapz(sol.y[0] * sol.y[0], xspan)
    eigenfuction = abs(sol.y[0] / np.sqrt(norm))
    EigenF[:, modes - 1] = eigenfuction
    plt.plot(xspan, eigenfuction, col[modes - 1])

A1 = EigenF
A2 = EigenV
#plt.show()

################# (b)

# Parameters
L = 4  # Boundaries of x
x = np.arange(-L, L + 0.1, 0.1)  # Points from -4 to 4 with a step of 0.1
dx = 0.1
N = len(x)
A = np.zeros((N-2, N-2))  # Create an N x N matrix of zeros


# Functions for direct solve method

# Set up Matrix A
for j in range(N-2):
    A[j, j] = -2 - (x[j + 1] ** 2) * (dx ** 2)
# Set up the off-diagonal entries (-1) to the left and right of the main diagonal
for i in range(N - 3): 
    A[i, i + 1] = 1  # Right of the main diagonal
    A[i + 1, i] = 1  # Left of the main diagonal

# Setting Values for Boundary Terms
A_2 = np.zeros((N-2, N-2))
A_2[0, 0] = 4 / 3
A_2[0, 1] = -1 / 3

A_3 = np.zeros((N-2, N-2))
A_3[N - 1 - 2, N - 2 - 2] = - 1 / 3
A_3[N - 1 - 2, N - 1 - 2] = 4 / 3

A = A + A_2 + A_3
A = A / (dx ** 2)

# Solve the eigenvalue problem for -A (to get positive eigenvalues for bound states)
eigenvalues, eigenfunctions = eigh(-A, k=5, which = 'SM')

phi_0 = (4 / 3) * eigenfunctions[0, :] - (1 / 3) * eigenfunctions[1, :]
phi_n = - (1 / 3) * eigenfunctions[-2, :] + (4 / 3) * eigenfunctions[-1, :]
eigenfunctions = np.vstack((phi_0, eigenfunctions, phi_n))

A3 = eigenfunctions
A4 = eigenvalues

#print(A3)

# Normalization
for i in range(5):
    norm = np.trapz(eigenfunctions[:, i] ** 2, x)
    eigenfunctions[:, i] = abs(eigenfunctions[:, i] / np.sqrt(norm))
    plt.plot(x, eigenfunctions[:, i])
plt.legend(
    ["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"],
    loc="upper left"
    )
#plt.show()



################# (c)

# Parameters
L = 2  # Boundaries of x
dx = 0.1
x = np.arange(-L, L + dx, dx)  # Points from -4 to 4 with a step of 0.1
tol = 1e-6
eigenF1, eigenF2 = np.zeros((len(x), 2)), np.zeros((len(x), 2))
eigenV1, eigenV2 = np.zeros(2), np.zeros(2)
gammaValues = [0.05, - 0.05]

# Defining Shooting Equation
def shootingEquation(x, phi, epsilon, gamma):
    return [phi[1], (gamma * phi[0]**2 + x**2 - epsilon) * phi[0]]

for gamma in gammaValues: # loop for the values gamma +-0.05
    epsilon_start = 0.1
    A = 1e-6 # values we need for initial guess
    for modes in range(1,3):  # loop for 2 columns of eigenfunciton and value
        dA = 0.01
        for i in range(1000): # loop for adjusting epsilon
            epsilon = epsilon_start
            deEpsilon = 0.2
            for j in range(1000): # loop for stepping on value of A
                # Initial Condition
                Y0 = [A, np.sqrt(L**2 - epsilon) * A]
                # ODE solve
                odeSolve = solve_ivp(lambda x, phi: shootingEquation(x, phi, epsilon, gamma),[x[0], x[-1]], Y0, t_eval=x)
                # extract the phi and x solutions
                phi_ans = odeSolve.y.T
                x_ans = odeSolve.t
                # Boundary Tolarence check
                if abs(phi_ans[-1, 1] + np.sqrt(L**2 - epsilon) * phi_ans[-1, 0]) < tol:
                    break
                # Adjust to steps of epsilon
                if (-1) ** (modes + 1) * (phi_ans[-1, 1] + np.sqrt(L**2 - epsilon) * phi_ans[-1, 0]) > 0:
                    epsilon += deEpsilon
                else:
                    epsilon -= deEpsilon
                    deEpsilon /= 2
            # Check whether the prob density is focused
            integral = simpson(phi_ans[:, 0]**2, x=x_ans)
            if abs(integral - 1) < tol:
                break
            # Step on A
            if integral < 1:
                A += dA
            else:
                A -= dA
                dA /= 2
        # stepping on epsilon
        epsilon_start = epsilon + 0.2
        # save the result
        if gamma > 0:
            eigenF1[:, modes - 1] = np.abs(phi_ans[:, 0])
            eigenV1[modes - 1] = epsilon
        else:
            eigenF2[:, modes - 1] = np.abs(phi_ans[:, 0])
            eigenV2[modes - 1] = epsilon
A5 = eigenF1
A6 = eigenV1
A7 = eigenF2
A8 = eigenV2

plt.plot(x, A5)
plt.plot(x, A7)
plt.legend(["$\\phi_1$", "$\\phi_2$"], loc="upper right")
#plt.show()


############ (d)

def shootingequation(x, phi, epsilon):
    return [phi[1], (x ** 2 - epsilon) * phi[0]]

L = 2
x = [-L, L]
epsilon = 1
A = 1
phi0 = [A, np.sqrt(L ** 2 - epsilon) * A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

dt45, dt23, dtRadua, dtBDF = [], [], [], []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}
    #solve the ivp problem
    ans45 = solve_ivp(shootingequation, x, phi0, method='RK45', args=(epsilon,), **options)
    ans23 = solve_ivp(shootingequation, x, phi0, method='RK23', args=(epsilon,), **options)
    ansRadua = solve_ivp(shootingequation, x, phi0, method='Radau', args=(epsilon,), **options)
    ansBDF = solve_ivp(shootingequation, x, phi0, method='BDF', args=(epsilon,), **options)
    # calculate average time steps
    dt45.append(np.mean(np.diff(ans45.t)))
    dt23.append(np.mean(np.diff(ans23.t)))
    dtRadua.append(np.mean(np.diff(ansRadua.t)))
    dtBDF.append(np.mean(np.diff(ansBDF.t)))

fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fitRadua = np.polyfit(np.log(dtRadua), np.log(tols), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)

A9 = np.array([fit45[0], fit23[0], fitRadua[0], fitBDF[0]])


######################  (e)

# HW 3 - Part e
# Define first five Gauss-Hermite polynomial
def H0(x):
    return np.ones_like(x)


def H1(x):
    return 2 * x


def H2(x):
    return 4 * (x ** 2) - 2


def H3(x):
    return 8 * (x ** 3) - 12 * x


def H4(x):
    return 16 * (x ** 4) - 48 * (x ** 2) + 12


# Define x range
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)
H = np.column_stack([H0(x), H1(x), H2(x), H3(x), H4(x)])
phi = np.zeros(H.shape)

# Normalize solutions and put in phi
for j in range(5):
    phi[:, j] = ((np.exp(- (x ** 2) / 2) * H[:, j]) /
                 np.sqrt(math.factorial(j) * (2 ** j) * np.sqrt(np.pi))
                 )

# Build up results matrix for comparison
erps_a = np.zeros(5)
erps_b = np.zeros(5)

er_a = np.zeros(5)
er_b = np.zeros(5)
print(A2)

# Compare eigenfunctions and eigenvalues
for j in range(5):
    erps_a[j] = simpson(((abs(A1[:, j])) - abs(phi[:, j])) ** 2, x=x)
    erps_b[j] = simpson(((abs(A3[:, j])) - abs(phi[:, j])) ** 2, x=x)

    er_a[j] = 100 * (abs(A2[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))
    er_b[j] = 100 * (abs(A4[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))

A10 = erps_a
A11 = er_a

A12 = erps_b
A13 = er_b

print("A10:")
print(A10)
print("A11:")
print(A11)







                

