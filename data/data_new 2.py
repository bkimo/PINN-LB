# Ref: https://nbviewer.jupyter.org/github/numerical-mooc/numerical-mooc/blob/master/lessons/02_spacetime/02_01_1DConvection.ipynb
# The square wave with inviscide burgers
# Forward Euler in time and backward Euler in space. First order approximation.

import numpy as np
import matplotlib.pyplot as plt 

# Set parameters.
nx = 491  # number of spatial discrete points
L = 2.0  # length of the 1D domain
dx = L / (nx - 1)  # spatial grid size
nt = 100  # number of time steps
dt = 0.002  # time-step size

x = np.linspace(0.0, L, num=nx)
u0 = np.ones(nx)
mask = np.where(np.logical_and(x >= 0.5, x <= 1.0))
u0[mask] = 2.0

# Compute the solution using Euler's method and array slicing.
u = u0.copy()
for n in range(1, nt):
    u[1:] = u[1:] - dt / dx * u[1:] * (u[1:] - u[:-1])

# Plot the solution after nt time steps
# along with the initial conditions.
plt.figure(figsize=(4.0, 4.0))
plt.xlabel('x')
plt.ylabel('u')
plt.grid()
plt.plot(x, u0, label='Initial',
            color='C0', linestyle='--', linewidth=2)
plt.plot(x, u, label='nt = {}'.format(nt),
            color='C1', linestyle='-', linewidth=2)
plt.legend()
plt.xlim(0.0, L)
plt.ylim(0.0, 2.5)
plt.show()