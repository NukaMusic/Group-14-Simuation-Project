'''Simulation'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

t_max = 5  # simulation time in seconds
dt = 0.25  # step size
N = 2 ** 10  # Number of particles
D = 0.01  # diffusivity

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

X = np.random.uniform(x_min, x_max, size=N)
Y = np.random.uniform(y_min, y_max, size=N)

X_i = np.copy(X)
Y_i = np.copy(Y)

def animate(time, X, Y, X_i, Y_i, fig):
I doctest
dummy_threadingdo
god
    X = X_i + np.random.normal(0, 1, size=(N,)) * np.sqrt(2 * D * dt) #Lagrange Diffusion
    Y = Y_i + np.random.normal(0, 1, size=(N,)) * np.sqrt(2 * D * dt) #Lagrange Diffusion
    X_i = np.copy(X)
    Y_i = np.copy(Y)
    fig.suptitle("Frame: " + str(time), fontsize=20)
    points.set_data(X, Y)
    return points,

fig = plt.figure()
ax = plt.axes()
ax.axis('scaled')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
points, = ax.plot(X, Y, 'o', markersize= 0.5)
fig.suptitle("Frame: ", fontsize=20)

anim = animation.FuncAnimation(fig, animate, fargs=(X, Y, X_i, Y_i, fig,), frames=100, repeat=False, interval=100)

plt.show()
