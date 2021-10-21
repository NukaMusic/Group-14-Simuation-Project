"""Simulation

Written by Group 14:
Nuka Miettinen Langgård
Conor Shannon
2021
"""

import numpy as np
from scipy import spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
import time  # for debugging and optimization
from vars_init_conditions import *

starttime = time.time()

print(time.time() - starttime)

pos = np.genfromtxt(vel_file, usecols=(0, 1))  # position array
vel = np.genfromtxt(vel_file, usecols=(2, 3))  # velocity array
row_length = np.sqrt(len(pos)).astype(int)
column_length = row_length  # currently assumes square

x = np.random.uniform(x_min, x_max, size=N)  # initial x-positions
y = np.random.uniform(y_min, y_max, size=N)  # initial y-positions

phi1 = np.ones(N)  # Array of ones for where function
phi0 = np.zeros(N)  # Array of zeros for where function
blue = np.full(N, 'b')  # Array of blue for where function
red = np.full(N, 'r')  # Array of red for where function
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['r', 'lime', 'b'], 256)  # colormap for graphing

if init_type == 2:
    phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, phi1, phi0)

if init_type == 3:
    phi = np.where(x < 0, phi1, phi0)

# Create a mesh and find the average phi values within it
def getavrphimesh(x, y):
    x_gran = np.round((x - x_min) / (x_max - x_min) * (Nx-1)).astype(int)  # figures out which grid square (granular
    y_gran = np.round((y - y_min) / (y_max - y_min) * (Ny-1)).astype(int)  # coordinate) each point fits into
    grancoord = np.column_stack((x_gran, y_gran))  # array of each point's granular coordinate
    unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
    avrphi = np.bincount(ids, phi)/count
    avrphi = np.rot90(np.reshape(avrphi, [Nx, Ny]))
    return avrphi

def get_velocities(x, y):  # given a coordinate, tells us what nearest velocity vector is
    distance, index = spatial.cKDTree(pos).query(np.column_stack((x, y)), workers=-1)
    x_velocities = vel[index][:, 0]
    y_velocities = vel[index][:, 1]
    return x_velocities, y_velocities


# Visualize the data
def visualize(viz_type):
    if viz_type == 1:
        for i in range(N):
            col = np.where(phi == 1, blue, red)  # create array of colours for each point
        plt.scatter(x, y, color=col, s=0.1)
        plt.show()

    if viz_type == 2:
        avphi = getavrphimesh(x, y)
        plt.imshow(avphi, interpolation='nearest', cmap=cmap,
                   extent=(x_min, x_max, y_min, y_max))  # interpolate = ?, cmap = colour map, extent changes graph size
        plt.colorbar()  # colour map legend
        plt.show()  # plot it!


visualize(viz_type)

print(time.time() - starttime)

for i in np.arange(0, (t_max+dt), dt):
    if vel_type == 1:
        v_x, v_y = get_velocities(x, y)
        x += v_x * dt
        y += v_y * dt
    x += np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N)  # Lagrange Diffusion and advection
    y += np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N)  # Lagrange Diffusion and advection
    # Walls
    x = np.where(x > x_max, 2 * x_max - x, x)  # if point is beyond wall, update
    x = np.where(x < x_min, 2 * x_min - x, x)  # position to bounce off wall as
    y = np.where(y > y_max, 2 * y_max - y, y)  # far as it went beyond the wall
    y = np.where(y < y_min, 2 * y_min - y, y)

visualize(viz_type)

print(time.time() - starttime)
