'''Simulation

Written by Group 14:
Nuka Miettinen Langgård
2021
'''

import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp

class Simulation:
    """
        Runs simulation.

        database_file: Path to data.
    """

    database_file: str

    def __init__(self, database_file: str):
        """
            Init class.
        """
        self.debug = False
        self.database_file = database_file

    def start_simulation(self, x_min, x_max, y_min, y_max, t_max, dt, exp, D, Nx, Ny):  
        """
            Start simulation; option to change default values.
        """

        # 2^(exp) = N
        N = 2 ** exp

        print("[" + mp.current_process().name + "] Simulation running...")

        starttime = time.time()

        if self.debug:
            print(time.time() - starttime)

        pos = np.genfromtxt(self.database_file, usecols=(0, 1))
        vel = np.genfromtxt(self.database_file, usecols=(2, 3))
        row_length = np.sqrt(len(pos)).astype(int)
        column_length = row_length

        x = np.random.uniform(x_min, x_max, size=N)
        y = np.random.uniform(y_min, y_max, size=N)
        velocities = np.empty(shape=(row_length * column_length, 2))

        phi1 = np.ones(N)  # Array of ones for where function
        phi0 = np.zeros(N)  # Array of zeros for where function
        # blue = np.full(N, 'b')  # Array of blue for where function
        # red = np.full(N, 'r')  # Array of red for where function
        
        # initial conditions
        # 1: 1D problem
        # 2: middle patch
        # 3: side patch
        init_type = 3

        if init_type == 2:
            phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, phi1, phi0)

        if init_type == 3:
            phi = np.where(x < 0, phi1, phi0)

        if self.debug:
            print(time.time() - starttime)

        # create a mesh and find the average phi values within it
        def getavrphimesh(x, y):
            x_gran = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * Nx).astype(int)
            y_gran = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * Ny).astype(int)
            grancoord = np.column_stack((x_gran, y_gran))
            unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
            avrphi = np.bincount(ids, phi)/count
            avrphi = np.delete(avrphi, [-1, -2])
            avrphi = np.reshape(avrphi, [Nx, Ny])
            avrphi = np.rot90(avrphi, 1)
            return avrphi

        avphi = getavrphimesh(x, y)

        # for i in range(N):
        #     col = np.where(phi == 1, blue, red)

        if self.debug:
            print(time.time() - starttime)

        # plt.scatter(x, y, color=col, s=0.1)

        def get_velocities(x, y):
            x_coordinates = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length-1)).astype(int)
            y_coordinates = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length-1)).astype(int)
            x_velocities = np.empty(shape=N)
            y_velocities = np.empty(shape=N)
            for i in range(N):
                velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
                x_velocities[i] = vel[velocity_index][0]
                y_velocities[i] = vel[velocity_index][1]
            return x_velocities, y_velocities

        if self.debug:
            print(time.time() - starttime)

        for i in np.arange(0, t_max, dt):
            v_x, v_y = get_velocities(x, y)
            x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
            y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
            #Walls
            x = np.where(x > x_max, 2 * x_max - x, x)
            x = np.where(x < x_min, 2 * x_min - x, x)
            y = np.where(y > y_max, 2 * y_max - y, y)
            y = np.where(y < y_min, 2 * y_min - y, y)

        if self.debug:
            print(time.time() - starttime)

        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['r', 'lime', 'b'], 256)  # colormap for graphing

        print("[" + mp.current_process().name + "] Simulation done.")
        
        plt.imshow(getavrphimesh(x, y), interpolation='nearest', cmap=cmap, origin='lower', extent=(x_min, x_max, y_min, y_max))
        plt.colorbar()
        plt.show() 
