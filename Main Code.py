'''Simulation'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

database_file = 'velocityCMM3.dat'

pos = np.genfromtxt(database_file, usecols=(0, 1))
vel = np.genfromtxt(database_file, usecols=(2, 3))
row_length = np.sqrt(len(pos)).astype(int)
column_length = row_length

pos = np.round((pos - np.amin(pos)) / (np.amax(pos) - np.amin(pos)) * (row_length - 1)).astype(int)

velocities = np.empty(shape=(row_length * column_length, 2))

for coordinates in pos:
    index = coordinates[1] + coordinates[0] * row_length
    velocities[index] = vel[index]

t_max = 5  # simulation time in seconds
dt = 0.01 # step size
N = 2 ** 10  # Number of particles
D = 0.01  # diffusivity

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

x = np.random.uniform(x_min, x_max, size=N)
y = np.random.uniform(y_min, y_max, size=N)

x_i = np.copy(x)
y_i = np.copy(y)

def get_velocities(x, y):
    x_coordinates = np.round((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length - 1)).astype(int)
    y_coordinates = np.round((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length - 1)).astype(int)
    x_velocities = np.empty(shape=N)
    y_velocities = np.empty(shape=N)
    for i in range(N):
        velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
        x_velocities[i] = velocities[velocity_index][0]
        y_velocities[i] = velocities[velocity_index][1]
    return x_velocities, y_velocities

def animate(time, x, y, points, fig):
    v_x, v_y = get_velocities(x, y)
    x += v_x * dt #+ np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,)) #Lagrange Diffusion
    y += v_y * dt #+ np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,)) #Lagrange Diffusion
    for i in range(N):
        if x[i] > x_max:
            x[i] = 2 * x_max - x[i]
        elif x[i] < x_min:
            x[i] = 2 * x_min - x[i]
        if y[i] > y_max:
            y[i] = 2 * y_max - y[i]
        elif y[i] < y_min:
            y[i] = 2 * y_min - y[i]

    fig.suptitle("Frame: " + str(time), fontsize=20)
    points.set_data(x, y)
    return points

fig = plt.figure()
ax = plt.axes()
ax.axis('scaled')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
points, = ax.plot(x, y, 'o', markersize= 0.5)
fig.suptitle("Frame: ", fontsize=20)

anim = animation.FuncAnimation(fig, animate, fargs=(x, y, points, fig), frames=1000, repeat=False, interval=1)

plt.show()