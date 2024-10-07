import numpy as np


def f(x):
    return x * np.sin(3 * x) - np.exp(x)
def f_prime(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)
x = np.array([-1.6])
tolerance = 1e-6
ite_newton = 0
for j in range(1000):
    x = np.append(x, x[j] - f(x[j]) / f_prime(x[j]))
    ite_newton = ite_newton + 1
    if abs(x[j+1] - x[j]) < tolerance:
        break
A1 = x



a = -0.7
b = -0.4
x_mid = []
for j in range(1000):
    xmid = (a + b) / 2
    x_mid.append(xmid)
    f_mid = f(xmid)
    if(f_mid) > 0:
        a = xmid
    else:
        b = xmid
    if abs(f_mid) < 1e-6:
        break
ite_bisec = j+1
A2 = x_mid
A3 = np.array([ite_newton, ite_bisec])


A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

A4 = A + B
A5 = 3 * x - 4 * y
A6 = np.dot(A, x)
x_minus_y = x - y
A7 = np.dot(B, x_minus_y)
A8 = np.dot(D, x)
A9 = np.dot(D, y) + z
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)

