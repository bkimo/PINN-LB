################################################################
#
# Class 04: 1D-Inviscid Burgers' equation
# Compute an analytic solution of viscous Burgers' equation 
#    u = -2 * nu * (phi_prime/phi) + 4
# Author: Bong-Sik Kim
# Date: 12/01/2021
#
################################################################



import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify as lbd

x, nu, t = sp.symbols('x nu t')

phi = (sp.exp(-(x-4*t)**2/(4*nu*(t+1))) + 
       sp.exp(-(x-4*t-2*np.pi)**2 / (4*nu*(t+1))) )
phi_prime = phi.diff(x)

u = -2 * nu * (phi_prime/phi) + 4
u_lamb = lbd((t,x,nu), u)

# Set parameters for training data generation
nx = 256  # number of spatial grid points
L = 2.0 * np.pi  # length of the domain
dx = L / (nx - 1)  # spatial grid size
nu = 0.07  # viscosity
nt = 500  # number of time steps to compute
sigma = 0.1  # CFL limit
dt = sigma * dx**2 / nu  # time-step size

# Discretize the domain.
x = np.linspace(0.0, L, num=nx)

# Set the initial conditions.
t = 0.0
u0 = np.array([u_lamb(t, xi, nu) for xi in x])

# Compute the analytical solution.
#u_analytical = np.array([u_lamb(nt * dt, xi, nu) for xi in x])

# Compute the history of the analytical solution.
u_history = [np.array([u_lamb(n * dt, xi, nu) for xi in x])
                for n in range(nt)]
#Convert u_history into numpy array
#u_sol = np.array(u_history)

# Generate time values for saving
t_value = np.array([n*dt for n in range(nt)]) 
#m,n = u_sol.shape  # (m, n) = shape(t,u)
#print(m,n)
#print(t_value[499])

#u_exact = np.array([u_sol[i,:] for i in range(m)])
np.save("data2_x", x)
np.save("data2_t", t_value)
#np.save("data2_u", u_exact)
np.save("data2_u", u_history)

# Plot the analytical solution.
plt.figure(figsize=(6.0, 4.0))
plt.xlabel('x')
plt.ylabel('u')
plt.grid()
#plt.plot(x, u_sol[0,:], label='Initial',
 #           color='C0', linestyle='-', linewidth=2)
plt.plot(x, u_history[0], label='u(x,0)',
            color='C0', linestyle='-', linewidth=2)
#plt.plot(x, u_sol[-1,:], label='Analytical',
#            color='C1', linestyle='--', linewidth=2)
plt.plot(x, u_history[-1], label='u(x,t)',
            color='C1', linestyle='--', linewidth=2)
plt.legend()
plt.xlim(0.0, L)
plt.ylim(0.0, 10.0)

plt.show()


