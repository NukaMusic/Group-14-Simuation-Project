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
dpg.create_viewport(title="Simulation", width=900, height=800)
dpg.setup_dearpygui()


def call_simulation(path, a, b, c, d, e, f, g, h, i, j, k, l, m):
    temp = Simulation(path)
    temp.start_simulation(a, b, c, d, e, f, g, h, i, j, k, l, m)


def sim_callback():
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=call_simulation, args=(
        dpg.get_value("path"), dpg.get_value("min_x"),
        dpg.get_value("max_x"), dpg.get_value("min_y"),
        dpg.get_value("max_y"), float(dpg.get_value("time")),
        float(dpg.get_value("step_size")), dpg.get_value("num_particles"),
        float(dpg.get_value("diff")), dpg.get_value("euler_x"), dpg.get_value("euler_y"),
        dpg.get_value("init_type"), dpg.get_value("viz_type"), dpg.get_value("vel_type")))
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
    dpg.add_text("# of Particles (2^(your_value_below))")
    dpg.add_input_int(tag="num_particles", default_value=14, label=" (exp.)")  # dpg.add_drag_int for a slider
    dpg.add_text("Diffusivity")
    dpg.add_input_text(tag="diff", default_value=0.01, decimal=True)
    dpg.add_text("Init Type: 1 for 1D problem (overrides y_min, y_max, Ny, D, t_max and vel_type), 2 for middle patch, 3 for side patch")
    dpg.add_input_int(tag="init_type", default_value=3, label=" Initial Cond.")
    dpg.add_text("Vis Type: 1 for particles, 2 for Concentration field")
    dpg.add_input_int(tag="viz_type", default_value=2, label=" Vis. Display ")
    dpg.add_text("Vel Type: False(0) for No velocity field, True(1) for the previously defined velocity field")
    dpg.add_input_int(tag="vel_type", default_value=True, label=" Vel. Field Conditions")  # (change this to be a true and false button maybe?)
    dpg.add_text("Done?")
    dpg.add_button(label="Run Simulation", callback=sim_callback)


class Simulation:
    """
        Runs simulation.

        database_file: Path to data.
    """

    def __init__(self, database_file):
        """
        Init class.
        """
        self.debug = False
        self.database_file = database_file

    def start_simulation(self, x_min, x_max, y_min, y_max, t_max, dt, exp, D, Nx, Ny, init_type, viz_type, vel_type):
        """
        Start simulation; option to change default values.
        """
        starttime = time.time()

        if self.debug:
            print(time.time() - starttime)

        print("[" + mp.current_process().name + "] Simulation running...")

        N = 2 ** exp
        x = np.random.uniform(x_min, x_max, size=N)  # initial x-positions
        y = np.random.uniform(y_min, y_max, size=N)  # initial y-positions
        pos = np.genfromtxt(self.database_file, usecols=(0, 1))  # position array for velocity field
        vel = np.genfromtxt(self.database_file, usecols=(2, 3))  # velocity array for velocity field
        t = 0  # initialises t for titles

        ones = np.ones(N)  # Array of ones for where function
        zeros = np.zeros(N)  # Array of zeros for where function
        blue = np.full(N, 'b')  # Array of blue for where function
        red = np.full(N, 'r')  # Array of red for where function
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['r', 'lime', 'b'])  # colormap for graphing
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('eng_colmap', [(0, 'r'), (0.29999, 'lime'), (0.3, 'b'), (1, 'b')])  # colormap for engineering simulation

        oneD_ref = np.genfromtxt('reference_solution_1D.dat')

        if init_type == 1:
            phi = np.where(x <= 0, ones, zeros)
            y_min = -0.001
            y_max = 0.001
            Ny = 1
            D = 0.1
            vel_type = 0
            t_max = 0.2

        if init_type == 2:
            phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, ones, zeros)

        if init_type == 3:
            phi = np.where(x < 0, ones, zeros)

        # Velocity data position resolution (distance between velocity field points
        x_posres = (np.max(pos[:, 0]) - np.min(pos[:, 0])) / (len(np.unique(pos[:, 0]).astype(int)) - 1)
        y_posres = (np.max(pos[:, 1]) - np.min(pos[:, 1])) / (len(np.unique(pos[:, 1]).astype(int)) - 1)

        maxdist = np.sqrt(
            x_posres ** 2 + y_posres ** 2)  # maximum allowable distance for a particle to be from a vel coord

        if self.debug:
            print(time.time() - starttime)

        # Create a mesh and find the average phi values within it
        def getavrphimesh(x, y):
            x_gran = np.round((x - x_min) / (x_max - x_min) * (Nx - 1)).astype(
                int)  # figures out which grid square (granular
            y_gran = np.round((y - y_min) / (y_max - y_min) * (Ny - 1)).astype(int)  # coordinate) each point fits into
            grancoord = np.column_stack((x_gran, y_gran))  # array of each point's granular coordinate
            unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
            avrphi = np.bincount(ids, phi) / count
            avrphi = np.rot90(np.reshape(avrphi, [Nx, Ny]))
            return avrphi

        if self.debug:
            print(time.time() - starttime)

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
                plt.plot(oneD_ref[:, 0], oneD_ref[:, 1], color='r')
                plt.scatter(np.linspace(x_min, x_max, Nx), avphi[0], s=15, marker='.', color='b')
                plt.plot(np.linspace(x_min, x_max, Nx), avphi[0], color='b')
                plt.legend(['Reference Solution', 'Simulation'], loc='upper right')
                plt.title('1D Particle Distribution', fontdict=None, loc='center', pad=None)  # Plot Titles
                plt.xlabel('x')
                plt.ylabel('Concentration, ϕ ')
                plt.show()

            if init_type == 2 or init_type == 3:
                if viz_type == 1:
                    for i in range(N):
                        col = np.where(phi == 1, blue, red)  # create array of colours for each point
                    plt.scatter(x, y, color=col, s=0.1)
                    plt.title('2D Particle Location Visualisation at ' + str(round(t / dt) * dt) + ' s', fontdict=None,
                              loc='center', pad=20)  # Plot Titles
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.show()

                if viz_type == 2:
                    avphi = getavrphimesh(x, y)
                    plt.imshow(avphi, interpolation='nearest', cmap=cmap2,
                               extent=(x_min, x_max, y_min,
                                       y_max))  # interpolate = ?, cmap = colour map, extent changes graph size
                    plt.colorbar(label='Concentration, ϕ')  # colour map legend
                    plt.title('2D Particle Concentration Representation at ' + str(t) + ' s',
                              fontdict=None, loc='center', pad=20)  # Plot Titles
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.show()  # plot it!

        if init_type == 2 or init_type == 3:
            visualize(init_type, viz_type)

        if self.debug:
            print(time.time() - starttime)
        init_type = 3
        viz_type =2
        for step in np.arange(0, t_max, dt):
            if vel_type:
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
            t += dt  # t for titles
            if init_type == 2 or init_type == 3:
                if round(t % 0.05, 6) == 0:
                    visualize(init_type, viz_type)

        if init_type == 1:
            visualize(init_type, viz_type)

        if self.debug:
            print(time.time() - starttime)

        print("[" + mp.current_process().name + "] Simulation done.")


# Prevents the Child program from relaunching the GUI
if mp.current_process().name == 'MainProcess':
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
