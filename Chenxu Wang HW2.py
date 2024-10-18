import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shooting_func(y, x, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tolerance = 1e-6
col = ['r', 'b', 'g', 'c', 'm']
L = 4
xspan = np.arange(-L, L + 0.1, 0.1) 
EigenF = np.zeros((len(xspan), 5))
EigenV = np.zeros(5)

epsilon_start = 1
for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = 1
    
    for j in range(1000):
        boundaries = [1, np.sqrt(L**2 - epsilon)]
        sol = odeint(shooting_func, boundaries, xspan, args=(epsilon,))

        if abs(np.sqrt(L**2 - epsilon) * sol[-1, 0] + sol[-1, 1]) < tolerance:
            break

        if ((-1) ** (modes + 1) * (sol[-1, 1] + np.sqrt(L**2 - epsilon) * sol[-1, 0])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2
         
    EigenV[modes - 1] = epsilon
    epsilon_start = epsilon + 0.1
    eigenfuction = abs(sol[:, 0] / np.sqrt(np.trapz(sol[:, 0] * sol[:, 0], xspan)))
    EigenF[:, modes - 1] = eigenfuction
    plt.plot(xspan, eigenfuction, col[modes - 1])

A1 = EigenF
A2 = EigenV
print(A1)
print(A2)
plt.show()
