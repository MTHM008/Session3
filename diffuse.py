import numpy as np
import matplotlib.pylab as plt

def hat(x):
    """ This is the 'hat' initial condition used in the diffusion problem"""
    nx = len(x)
    zeros = np.zeros(nx)
    ones = np.ones(nx)
    ic = np.where(np.logical_or(x <= 0.25, x >= 0.75), zeros, ones)
    return ic

# This program solves the linear diffusion equation using a FTBS scheme

K = 1.0
nx = 20
x = np.linspace(0., 1., nx)
dx = x[1]-x[0]

dt = .001

# From the class notes the time step restriction is 2 k dt/dx/dx <= 1.
dt = .9*dx*dx/2./K
print('dt = ', dt)

nu = (K*dt)/(dx*dx)

# Set initial condition
phi = hat(x)

plt.figure(1)
plt.plot(x, phi, 'bs')

nsteps = 100

# Loop over the time steps
for istep in range(nsteps):

    phi_np1 = phi + nu*(np.roll(phi, -1) - 2.0*phi + np.roll(phi, 1))        
    phi = np.copy(phi_np1)

    if istep % 10 == 0:
        plt.plot(x, phi)

plt.show()
    


