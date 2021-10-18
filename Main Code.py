'''Simulation

Written by Group 14:
Nuka Miettinen Langgård
Conor Shannon
2021
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time #for debugging and optimization

starttime = time.time()

print(time.time() - starttime)

database_file = 'velocityCMM3.dat' """import file object
velocity field must have same minimum and maximum bounds
#or the grid will stretch"""

pos = np.genfromtxt(database_file, usecols=(0, 1)) #position array
vel = np.genfromtxt(database_file, usecols=(2, 3)) #velocity array
row_length = np.sqrt(len(pos)).astype(int) 
column_length = row_length #currently assumes square

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

t_max = 0.4  # simulation time in seconds
dt = 0.0005  # step size
N = 2 ** 16  # Number of particles
D = 0.01  # diffusivity
Nx = 2  # Euler grid size x
Ny = 2  # Euler grid size y

x = np.random.uniform(x_min, x_max, size=N) #init x-positions
y = np.random.uniform(y_min, y_max, size=N) #init y-positions
velocities = np.empty(shape=(row_length * column_length, 2)) 
#create empty velocity array

phi1 = np.ones(N)  # Array of ones for where function
phi0 = np.zeros(N)  # Array of zeros for where function
# blue = np.full(N, 'b')  # Array of blue for where function
# red = np.full(N, 'r')  # Array of red for where function
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['r', 'lime', 'b'], 256)  # colormap for graphing

# initial conditions
# 1: 1D problem
# 2: middle patch
# 3: side patch
init_type = 2

if init_type == 2:
    phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, phi1, phi0)

if init_type == 3:
    phi = np.where(x < 0, phi1, phi0)


# create a mesh and find the average phi values within it
def getavrphimesh(x, y):
    x_gran = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * Nx).astype(int) #figures out which grid square (granular
    y_gran = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * Ny).astype(int) #coordinate) each point fits into
    grancoord = np.column_stack((x_gran, y_gran)) #array of each point's granular coordinate
    unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
<<<<<<< Updated upstream
    avrphi = np.bincount(ids, phi)/count
    avrphi = np.delete(avrphi, [0, 1])
    avrphi = np.reshape(avrphi, [Nx, Ny])
    return avrphi
=======
    avrphi = np.bincount(ids, phi)/count #get average concentration (phi) of each granular coordinate
    avrphi = np.delete(avrphi, [-1, -2]) #bug fix
    avrphi = np.reshape(avrphi, [Nx, Ny]) #reshape 1D avrphi array into 2D
    avrphi = np.rot90(avrphi, 1) #rotate the reshaped array (clunky)
    return avrphi 
>>>>>>> Stashed changes


avphi = getavrphimesh(x, y)

# for i in range(N):
#     col = np.where(phi == 1, blue, red) #create array of colours for each point

print(time.time() - starttime)

# plt.scatter(x, y, color=col, s=0.1)

plt.imshow(avphi, interpolation='nearest', cmap=cmap, origin='lower', extent=(x_min, x_max, y_min, y_max))
#interpolate = ?, cmap = colour map, origin = 'Lower' -> (0,0) at bottom left, extent changes graph size
plt.colorbar() #colour map legend
plt.show() #plot it!


def get_velocities(x, y): #given a coordinate, tells us what nearest velocity vector is
    x_coordinates = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length-1)).astype(int) #same indexing
    y_coordinates = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length-1)).astype(int) #as avrphi function
    x_velocities = np.empty(shape=N) #empty arrays to receive velocity data
    y_velocities = np.empty(shape=N)
    for i in range(N): #turns our two vel arrays into a 1D array
        velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
        x_velocities[i] = vel[velocity_index][0]
        y_velocities[i] = vel[velocity_index][1]
    return x_velocities, y_velocities 

print(time.time() - starttime)

for i in np.arange(0, (t_max+dt), dt): 
    v_x, v_y = get_velocities(x, y)
    x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    #Walls
    x = np.where(x > x_max, 2 * x_max - x, x) #if point is beyond wall, update 
    x = np.where(x < x_min, 2 * x_min - x, x) #position to bounce off wall as
    y = np.where(y > y_max, 2 * y_max - y, y) #far as it went beyond the wall
    y = np.where(y < y_min, 2 * y_min - y, y)

print(time.time() - starttime)

avphi = getavrphimesh(x, y) #same plot business as above
plt.imshow(avphi, interpolation='nearest', cmap=cmap, origin='lower', extent=(x_min, x_max, y_min, y_max))
plt.colorbar()
plt.show()

# plt.scatter(x, y, color=col, s=0.1)

print(time.time() - starttime)