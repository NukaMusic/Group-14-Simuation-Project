"""
Simulation

Written by Group 14:
Nuka Miettinen Langgård
Conor Shannon
Oscar Jiang
2021
"""

import numpy as np
from scipy import spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
import time  # for debugging and optimization
import multiprocessing as mp
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="Simulation", width=820, height=690)
dpg.setup_dearpygui()


def call_simulation(path, min_x, max_x, min_y, max_y, euler_x, euler_y, time, step_size, num_particles, diff, init_type,
                    viz_type, use_vel):
    temp = Simulation(path, min_x, max_x, min_y, max_y, euler_x, euler_y, time, step_size, num_particles, diff,
                      init_type, viz_type, use_vel)
    temp.start_simulation(init_type, viz_type)


def sim_callback():
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=call_simulation, args=(
        dpg.get_value("path"), dpg.get_value("min_x"),
        dpg.get_value("max_x"), dpg.get_value("min_y"),
        dpg.get_value("max_y"), float(dpg.get_value("time")),
        float(dpg.get_value("step_size")), dpg.get_value("num_particles"),
        float(dpg.get_value("diff")), dpg.get_value("euler_x"), dpg.get_value("euler_y"),
        dpg.get_value("init_type"), dpg.get_value("viz_type"), dpg.get_value("use_vel")))
    p.start()


# Creating Labels for the Input Parameters
with dpg.window(label="Parameters", width=800):
    dpg.add_text("File name of velocity field")
    dpg.add_input_text(tag="path", default_value="velocityCMM3.dat", label=" (.dat)")
    dpg.add_text("Domain")
    dpg.add_input_int(tag="min_x", default_value=-1, label=" (min. x)")
    dpg.add_input_int(tag="max_x", default_value=1, label=" (max. x)")
    dpg.add_input_int(tag="min_y", default_value=-1, label=" (min. y)")
    dpg.add_input_int(tag="max_y", default_value=1, label=" (max. y)")
    dpg.add_text("Euler Grid Size")
    dpg.add_input_int(tag="euler_x", default_value=16, label=" (x)")
    dpg.add_input_int(tag="euler_y", default_value=16, label=" (y)")
    dpg.add_text("Simulated time")
    dpg.add_input_text(tag="time", decimal=True, default_value=0.2, label=" second(s)")
    dpg.add_text("Step Size")
    dpg.add_input_text(tag="step_size", decimal=True, default_value=0.0005)
    dpg.add_text("# of Particles")
    dpg.add_input_int(tag="num_particles", default_value=100000, label="linear")  # dpg.add_drag_int for a slider
    dpg.add_text("Diffusivity")
    dpg.add_input_text(tag="diff", default_value=0.01, decimal=True)
    dpg.add_text("Init Type: 1 for 1D problem (overrides y_min, y_max, Ny, D, t_max and use_vel), 2 for middle patch,")
    dpg.add_text("3 for side patch, 4 for engineering simulation (overrides nearly all variables")
    dpg.add_input_int(tag="init_type", default_value=3, label=" Initial Cond.")
    dpg.add_text("Vis Type: 1 for particles, 2 for Concentration field")
    dpg.add_input_int(tag="viz_type", default_value=2, label=" Vis. Display ")
    dpg.add_text("Vel Type: 0 for No velocity field, 1 for the previously defined velocity field")
    dpg.add_input_int(tag="use_vel", default_value=1, label=" Vel. Field Conditions")  # (change this to be a true and false button maybe?)
    dpg.add_text("Done?")
    dpg.add_button(label="Run Simulation", callback=sim_callback)


class Simulation:
    """
        Runs simulation.

        velocity_field: Path to data.
    """

    def __init__(self, velocity_field, x_min, x_max, y_min, y_max, t_max, dt, N, D, Nx, Ny, init_type, viz_type, use_vel):
        """
        Init class.
        """
        self.debug = False
        self.velocity_field = velocity_field
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_max = t_max
        self.dt = dt
        self.N = N
        self.D = D
        self.Nx = Nx
        self.Ny = Ny
        self.init_type = init_type
        self.viz_type = viz_type
        self.use_vel = use_vel
        self.pos = np.genfromtxt(self.velocity_field, usecols=(0, 1))  # position array for velocity field
        self.vel = np.genfromtxt(self.velocity_field, usecols=(2, 3))  # velocity array for velocity field
        self.t = 0  # initialises t for titles
        self.ones = np.ones(self.N)  # Array of ones for where function
        self.zeros = np.zeros(self.N)  # Array of zeros for where function
        self.blue = np.full(self.N, 'b')  # Array of blue for where function
        self.red = np.full(self.N, 'r')  # Array of red for where function
        self.oneD_ref = np.genfromtxt('reference_solution_1D.dat')
        # Velocity data position resolution (distance between velocity field points
        self.x_posres = (np.max(self.pos[:, 0]) - np.min(self.pos[:, 0]))/(len(np.unique(self.pos[:, 0]).astype(int))-1)
        self.y_posres = (np.max(self.pos[:, 1]) - np.min(self.pos[:, 1]))/(len(np.unique(self.pos[:, 1]).astype(int))-1)
        self.maxdist = np.sqrt(self.x_posres ** 2 + self.y_posres ** 2)  # maximum allowable distance for a particle to be from a vel coord
        self.x = np.random.uniform(self.x_min, self.x_max, size=self.N)  # initial x-positions
        self.y = np.random.uniform(self.y_min, self.y_max, size=self.N)  # initial y-positions

    def get_initial_values(self, init_type):
        if init_type == 1:
            self.phi = np.where(self.x <= 0, self.ones, self.zeros)
            self.y_min = -0.001
            self.y_max = 0.001
            self.Ny = 1
            self.D = 0.1
            self.use_vel = 0
            self.t_max = 0.2

        elif init_type == 2:
            self.phi = np.where(np.sqrt(self.x ** 2 + self.y ** 2) < 0.3, self.ones, self.zeros)
            self.cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['r', 'lime', 'b'])  # colormap for graphing
        elif init_type == 3:
            self.phi = np.where(self.x < 0, self.ones, self.zeros)
            self.cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['r', 'lime', 'b'])  # colormap for graphing
        elif init_type == 4:
            self.D = 0.1
            self.x_min = -1
            self.x_max = 1
            self.y_min = -1
            self.y_max = 1
            self.Nx = self.Ny = 64
            self.N = 150000
            self.use_vel = 1
            self.cmap = mpl.colors.LinearSegmentedColormap.from_list('eng_colmap', [(0, 'r'), (0.29999, 'lime'),
                                                                    (0.3, 'b'), (1, 'b')])
                                                                    # colormap for engineering simulation
            self.t_max = 10
            self.viz_type = 2
            self.x = np.random.uniform(self.x_min, self.x_max, size=self.N)  # initial x-positions
            self.y = np.random.uniform(self.y_min, self.y_max, size=self.N)  # initial y-positions
            self.ones = np.ones(self.N)  # Array of ones for where function
            self.zeros = np.zeros(self.N)  # Array of zeros for where function
            self.phi = np.where(np.sqrt((self.x - 0.4) ** 2 + (self.y - 0.4) ** 2) < 0.1, self.ones, self.zeros)

    # Create a mesh and find the average phi values within it
    def getavrphimesh(self):
        x_gran = np.round((self.x - self.x_min) / (self.x_max - self.x_min) * (self.Nx - 1)).astype(int)  # figures out which grid square (granular
        y_gran = np.round((self.y - self.y_min) / (self.y_max - self.y_min) * (self.Ny - 1)).astype(int)  # coordinate) each point fits into
        grancoord = np.column_stack((x_gran, y_gran))  # array of each point's granular coordinate
        unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
        avrphi = np.bincount(ids, self.phi) / count
        avrphi = np.rot90(np.reshape(avrphi, [self.Nx, self.Ny]))
        return avrphi

    def get_velocities(self):  # given a coordinate, tells us what nearest velocity vector is
        distance, index = spatial.cKDTree(self.pos).query(np.column_stack((self.x, self.y)), workers=-1)
        x_velocities = self.vel[index][:, 0]
        y_velocities = self.vel[index][:, 1]
        x_velocities = np.where(distance > self.maxdist, self.zeros, x_velocities)
        y_velocities = np.where(distance > self.maxdist, self.zeros, y_velocities)
        return x_velocities, y_velocities

    # Visualize the data
    def visualize(self, init_type, viz_type):

        if init_type == 1:
            avphi = self.getavrphimesh()
            plt.plot(self.oneD_ref[:, 0], self.oneD_ref[:, 1], color='r')
            plt.scatter(np.linspace(self.x_min, self.x_max, self.Nx), avphi[0], s=15, marker='.', color='b')
            plt.plot(np.linspace(self.x_min, self.x_max, self.Nx), avphi[0], color='b')
            plt.legend(['Reference Solution', 'Simulation'], loc='upper right')
            plt.title('1D Particle Distribution', fontdict=None, loc='center', pad=None)  # Plot Titles
            plt.xlabel('x')
            plt.ylabel('Concentration, ϕ ')
            plt.show()

        else:
            if viz_type == 1:
                col = np.where(self.phi == 1, self.blue, self.red)  # create array of colours for each point
                plt.scatter(self.x, self.y, color=col, s=0.1)
                plt.title('2D Particle Location Visualisation at ' + str(round(self.t / self.dt) * self.dt) + 's', fontdict=None,
                          loc='center', pad=20)  # Plot Titles
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()

            if viz_type == 2:
                if self.init_type != 4:
                    avphi = self.getavrphimesh()
                if self.init_type == 4:
                    if self.t == 0:
                        avphi = self.getavrphimesh()
                    averphi = self.getavrphimesh()
                    avphi = np.where(averphi >= 0.3, np.ones((self.Nx, self.Ny)), avphi)
                plt.imshow(avphi, interpolation='nearest', cmap=self.cmap,
                           extent=(self.x_min, self.x_max, self.y_min,
                                   self.y_max))  # interpolate = ?, cmap = colour map, extent changes graph size
                plt.colorbar(label='Concentration, ϕ')  # colour map legend
                plt.title('2D Particle Concentration Representation at ' + str(round(self.t / self.dt) * self.dt) + 's',
                          fontdict=None, loc='center', pad=20)  # Plot Titles
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()  # plot it!

    def start_simulation(self, init_type, viz_type):

        starttime = time.time()

        print("[" + mp.current_process().name + "] Simulation running...")

        self.get_initial_values(init_type)

        if self.init_type != 1:
            self.visualize(init_type, viz_type)

        if self.debug:
            print(time.time() - starttime)

        for _ in np.arange(0, self.t_max, self.dt):
            if self.use_vel == 1:
                v_x, v_y = self.get_velocities()
                self.x += v_x * self.dt
                self.y += v_y * self.dt
            self.x += np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=self.N)  # Lagrange Diffusion and advection
            self.y += np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=self.N)  # Lagrange Diffusion and advection
            # Walls
            self.x = np.where(self.x > self.x_max, 2 * self.x_max - self.x, self.x)  # if point is beyond wall, update
            self.x = np.where(self.x < self.x_min, 2 * self.x_min - self.x, self.x)  # position to bounce off wall as
            self.y = np.where(self.y > self.y_max, 2 * self.y_max - self.y, self.y)  # far as it went beyond the wall
            self.y = np.where(self.y < self.y_min, 2 * self.y_min - self.y, self.y)
            self.t += self.dt  # t for titles
            if self.init_type != 1:
                if round(self.t % 0.05, 6) == 0:
                    print(self.t)
                    self.visualize(init_type, viz_type)

        if self.init_type == 1:
            self.visualize(init_type, viz_type)

        if self.debug:
            print(time.time() - starttime)

        print("[" + mp.current_process().name + "] Simulation done.")


# Prevents the Child program from relaunching the GUI
if mp.current_process().name == 'MainProcess':
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
