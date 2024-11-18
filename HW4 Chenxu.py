import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

############# Problem (a)

# Parameters
m = 8        # N value in x and y directions
n = m * m    # total size of matrix
L = 10
x = np.linspace(-L, L, m+1)
dx = x[2] - x[1]
dy = dx


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

### A = dx^2 + dy^2
diagonalsA = [
    e1.flatten(), e1.flatten(), e5.flatten(),
    e2.flatten(), -4 * e1.flatten(), e3.flatten(),
    e4.flatten(), e1.flatten(), e1.flatten()
]
offsets = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
A = (spdiags(diagonalsA, offsets, n, n).toarray()) / dx**2
A1 = A

plt.figure(1)
plt.spy(A)  # Visualize the sparsity structure of the matrix
plt.title("Matrix Structure A = $\partial_x^2 + \partial_y^2$")
#plt.show()

### B = dx
diagonalsB = [
    e1.flatten(), -1*e1.flatten(), e0.flatten(), e1.flatten(), -1*e1.flatten()
]
offsets = [-(n - m), -m, 0, m, (n - m)]
B = (spdiags(diagonalsB, offsets, n, n).toarray()) / (2 * dx)
A2 = B
plt.figure(2)
plt.spy(B)  # Visualize the sparsity structure of the matrix
plt.title("Matrix Structure B = $\partial_x$")
#plt.show()

### C = dy
diagonalsC = [
    e5.flatten(), -1 * e2.flatten(), e3.flatten(), -1 * e4.flatten()
]
offsets = [-m + 1, -1, 1, m - 1]
C = (spdiags(diagonalsC, offsets, n, n).toarray()) / (2 * dy)
A3 = C
plt.figure(3)
plt.spy(C)  # Visualize the sparsity structure of the matrix
plt.title("Matrix Structure C = $\partial_y$")
#plt.show()