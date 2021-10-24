'''Simulation

Written by Group 14:
Oscar Jiang
2021
'''

import dearpygui.dearpygui as dpg

import Main_Code
import multiprocessing as mp

# Shaping the Gui
dpg.create_context()
dpg.create_viewport(title="Simulation", width=500, height=800)
dpg.setup_dearpygui()

# Opening the Main Code and importing the code as a library for the gui to run
def call_simulation(path, a, b, c, d, e, f, g, h, i, j):
    temp = Main_Code.Simulation(path)
    temp.start_simulation(a, b, c, d, e, f, g, h, i, j)

def sim_callback(sender, data):
    
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=call_simulation, args=(
        dpg.get_value("path"), dpg.get_value("min_x"), 
        dpg.get_value("max_x"), dpg.get_value("min_y"), 
        dpg.get_value("max_y"), float(dpg.get_value("time")), 
        float(dpg.get_value("step_size")), dpg.get_value("num_particles"), 
        float(dpg.get_value("diff")), dpg.get_value("euler_x"), dpg.get_value("euler_y")))
    p.start()
    
# Setting up the gui interactions
with dpg.window(label="Parameters", width=300):
    dpg.add_text("Location of Simulation Data")
    dpg.add_input_text(tag="path", default_value="velocityCMM3.dat", label=" (.dat)")
    dpg.add_text("Domain")
    dpg.add_input_int(tag="min_x", default_value=-1, label=" (min. x)")
    dpg.add_input_int(tag="max_x", default_value=1, label=" (max. x)")
    dpg.add_input_int(tag="min_y", default_value=-1, label=" (min. y)")
    dpg.add_input_int(tag="max_y", default_value=1, label=" (max. y)")
    dpg.add_text("Euler Grid Size")
    dpg.add_input_int(tag="euler_x", default_value=65, label=" (x)")
    dpg.add_input_int(tag="euler_y", default_value=65, label=" (y)")
    dpg.add_text("Maximum Runtime")
    dpg.add_input_text(tag="time", decimal=True, default_value=0.4, label=" second(s)")
    dpg.add_text("Step Size")
    dpg.add_input_text(tag="step_size", decimal=True, default_value=0.0005)
    dpg.add_text("# of Particles (2^(your_value_below))")
    dpg.add_drag_int(tag="num_particles", min_value=1, default_value=16, max_value=50, label=" (exp.)")
    dpg.add_text("Diffusivity")
    dpg.add_input_text(tag="diff", default_value=0.01, decimal=True, label=" m^2/s")
    dpg.add_text("Done?")
    dpg.add_button(label="Run Simulation", callback=sim_callback)

# Making sure that the child program does not open the gui again
if mp.current_process().name == 'MainProcess':
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()