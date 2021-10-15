'''Simulation'''

import numpy as np
import matplotlib.pyplot as plt
import time

starttime = time.time()

print(time.time() - starttime)

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

t_max = 0.12  # simulation time in seconds
dt = 0.001 # step size
N = 2 ** 16  # Number of particles
D = 0.1  # diffusivity
phi1 = np.ones(N) #Array of ones for phi
phi0 = np.zeros(N) #Array of zeros for phi
blue = np.full(N, 'b') #Array of blue for graphing
red = np.full(N, 'r') #Array of red for graphing

#initial conditions
#1: 1D problem
#2: middle patch
#3: side patch
init_type = 3

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

x = np.random.uniform(x_min, x_max, size=N)
y = np.random.uniform(y_min, y_max, size=N)

if init_type == 2:
    phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, phi1, phi0)

if init_type == 3:
    phi = np.where(x<0, phi1, phi0)

print(time.time() - starttime)

for i in range(N):
    col = np.where(phi == 1, blue, red)

print(time.time() - starttime)

plt.scatter(x, y, color=col, s=0.1)
plt.show()

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

print(time.time() - starttime)

for i in np.arange(0, t_max, dt):
    v_x, v_y = get_velocities(x, y)
    x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    #Walls (needs heavy optimization)
    for i in range(N):
        if x[i] > x_max:
            x[i] = 2 * x_max - x[i]
        elif x[i] < x_min:
            x[i] = 2 * x_min - x[i]
        if y[i] > y_max:
            y[i] = 2 * y_max - y[i]
        elif y[i] < y_min:
            y[i] = 2 * y_min - y[i]

plt.scatter(x, y, color=col, s=0.1)

print(time.time() - starttime)

plt.show()
