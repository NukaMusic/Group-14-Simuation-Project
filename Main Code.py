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

x = np.random.uniform(x_min, x_max, size=N)  # initial x-positions
y = np.random.uniform(y_min, y_max, size=N)  # initial y-positions
pos = np.genfromtxt(vel_file, usecols=(0, 1))  # position array for velocity field
vel = np.genfromtxt(vel_file, usecols=(2, 3))  # velocity array for velocity field

ones = np.ones(N)  # Array of ones for where function
zeros = np.zeros(N)  # Array of zeros for where function
blue = np.full(N, 'b')  # Array of blue for where function
red = np.full(N, 'r')  # Array of red for where function
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['r', 'lime', 'b'], 256)  # colormap for graphing

oneD_ref = np.genfromtxt('reference_solution_1D.dat')

if init_type == 1:
    phi = np.where(x <= 0, ones, zeros)
    y_min = -0.001
    y_max = 0.001
    Ny = 1
    vel_type = 0

if init_type == 2:
    phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, ones, zeros)

if init_type == 3:
    phi = np.where(x < 0, ones, zeros)

# Velocity data position resolution (distance between velocity field points
x_posres = (np.max(pos[:, 0]) - np.min(pos[:, 0])) / (len(np.unique(pos[:, 0]).astype(int)) - 1)
y_posres = (np.max(pos[:, 1]) - np.min(pos[:, 1])) / (len(np.unique(pos[:, 1]).astype(int)) - 1)

maxdist = np.sqrt(x_posres ** 2 + y_posres ** 2)  # maximum allowable distance for a particle to be from a vel coord


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
    x_velocities = np.where(distance > maxdist, zeros, x_velocities)
    y_velocities = np.where(distance > maxdist, zeros, y_velocities)
    return x_velocities, y_velocities


# Visualize the data
def visualize(init_type, viz_type):
    if init_type == 1:
        avphi = getavrphimesh(x, y)
        plt.scatter(np.linspace(x_min, x_max, Nx), avphi[0], s=0.5)
        plt.plot(oneD_ref[:, 0], oneD_ref[:, 1])
        plt.title('1D Particle Distribution', fontdict=None, loc='center', pad=None) #Plot Titles
        plt.show()
    if init_type == 2 or init_type == 3:
        if viz_type == 1:
            for i in range(N):
                col = np.where(phi == 1, blue, red)  # create array of colours for each point
            plt.scatter(x, y, color=col, s=0.1)
            plt.title('2D Particle Location Visualisation at '+str(t)+' s', fontdict=None, loc='center', pad=None) #Plot Titles
            plt.show()

        if viz_type == 2:
            avphi = getavrphimesh(x, y)
            plt.imshow(avphi, interpolation='nearest', cmap=cmap,
                       extent=(x_min, x_max, y_min, y_max))  # interpolate = ?, cmap = colour map, extent changes graph size
            plt.colorbar(label='Concentration')  # colour map legend
            plt.title('2D Particle Concentration Representation at '+str(t)+' s', fontdict=None, loc='center', pad=None) #Plot Titles
            plt.show()  # plot it!


if init_type == 2 or init_type == 3:
    t=0 #initialises t for title
    visualize(init_type, viz_type)

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
    t=t_max

visualize(init_type, viz_type)

print(time.time() - starttime)
