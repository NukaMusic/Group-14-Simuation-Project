'''Simulation'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Simulation:

    def __init__(self, database_file):
        pos = np.genfromtxt(database_file, usecols=(0, 1))
        vel = np.genfromtxt(database_file, usecols=(2, 3))
        self.row_length = np.floor(np.sqrt(len(pos))).astype(int)
        self.column_length = self.row_length

        pos = np.round((pos - np.amin(pos)) / (np.amax(pos) - np.amin(pos)) * (self.row_length - 1)).astype(int)

        self.velocities = np.empty(shape=(self.row_length * self.column_length, 2))

        for coordinates in pos:
            index = coordinates[1] + coordinates[0] * self.row_length
            self.velocities[index] = vel[index]

        self.t_max = 5  # simulation time in seconds
        self.dt = 0.01 # step size
        self.N = 2 ** 10  # Number of particles
        self.D = 0.01  # diffusivity

        # Domain size
        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1

        self.x = np.random.uniform(self.x_min, self.x_max, size=self.N)
        self.y = np.random.uniform(self.y_min, self.y_max, size=self.N)

        self.x_i = np.copy(self.x)
        self.y_i = np.copy(self.y)

    def get_velocities(self, x, y):
        x_coordinates = np.round((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (self.row_length - 1)).astype(int)
        y_coordinates = np.round((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (self.row_length - 1)).astype(int)
        x_velocities = np.empty(shape=self.N)
        y_velocities = np.empty(shape=self.N)
        for i in range(self.N):
            velocity_index = y_coordinates[i] + x_coordinates[i] * self.row_length
            x_velocities[i] = self.velocities[velocity_index][0]
            y_velocities[i] = self.velocities[velocity_index][1]
        assert x_coordinates.size == x_velocities.size
        assert y_coordinates.size == y_velocities.size
        return x_velocities, y_velocities



    def animate(self, time):
        x_v, y_v = self.get_velocities(np.copy(self.x), np.copy(self.y))
        self.x = self.x_i + x_v * self.dt #+ np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=(self.N,)) #Lagrange Diffusion
        self.y = self.y_i + y_v * self.dt #+ np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=(self.N,)) #Lagrange Diffusion
        for i in range(self.N):
            if self.x[i] > self.x_max:
                self.x[i] = 2 * self.x_max - self.x[i]
            elif self.x[i] < self.x_min:
                self.x[i] = 2 * self.x_min - self.x[i]
            if self.y[i] > self.y_max:
                self.y[i] = 2 * self.y_max - self.y[i]
            elif self.y[i] < self.y_min:
                self.y[i] = 2 * self.y_min - self.y[i]

        self.x_i = np.copy(self.x)
        self.y_i = np.copy(self.y)


        self.fig.suptitle("Frame: " + str(time), fontsize=20)
        self.points.set_data(self.x, self.y)
        return self.points

    def run(self):

        self.fig = plt.figure()
        ax = plt.axes()
        ax.axis('scaled')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        self.points, = ax.plot(self.x, self.y, 'o', markersize= 0.5)
        self.fig.suptitle("Frame: ", fontsize=20)

        # fargs=(self)
        anim = animation.FuncAnimation(self.fig, self.animate, frames=1000, repeat=False, interval=1)

        plt.show()

sim = Simulation('velocityCMM3.dat')
sim.run()