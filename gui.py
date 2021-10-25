import dearpygui.dearpygui as dpg
import Main_Code
import multiprocessing as mp

dpg.create_context()
dpg.create_viewport(title="Simulation", width=900, height=800)
dpg.setup_dearpygui()

def call_simulation(path, a, b, c, d, e, f, g, h, i, j, k, l, m):
    temp = Main_Code.Simulation(path)
    temp.start_simulation(a, b, c, d, e, f, g, h, i, j, k , l, m)

def sim_callback(sender, data):
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
    dpg.add_text("Location of Simulation Data")
    dpg.add_input_text(tag="path", default_value="velocityCMM3.dat", label=" (.dat)")
    dpg.add_text("Domain")
    dpg.add_input_int(tag="min_x", default_value=-1, label=" (min. x)")
    dpg.add_input_int(tag="max_x", default_value=1, label=" (max. x)")
    dpg.add_input_int(tag="min_y", default_value=-1, label=" (min. y)")
    dpg.add_input_int(tag="max_y", default_value=1, label=" (max. y)")
    dpg.add_text("Euler Grid Size")
    dpg.add_input_int(tag="euler_x", default_value=64, label=" (x)")
    dpg.add_input_int(tag="euler_y", default_value=64, label=" (y)")
    dpg.add_text("Maximum Runtime")
    dpg.add_input_text(tag="time", decimal=True, default_value=0.2, label=" second(s)")
    dpg.add_text("Step Size")
    dpg.add_input_text(tag="step_size", decimal=True, default_value=0.0005)
    dpg.add_text("# of Particles (2^(your_value_below))")
    dpg.add_input_int(tag="num_particles", default_value=17, label=" (exp.)") # dpg.add_drag_int for a slider
    dpg.add_text("Diffusivity")
    dpg.add_input_text(tag="diff", default_value=0.01, decimal=True)
    dpg.add_text("Init Type: 1 for 1D problem (overrides y_min, y_max, Ny, and vel_type, 2 for middle patch, 3 for side patch)")
    dpg.add_input_int(tag="init_type", default_value=1, label=" Initial Cond.")
    dpg.add_text("Vis Type: 1 for particles, 2 for Concentration field")
    dpg.add_input_int(tag="viz_type", default_value=1, label=" Vis. Display ")
    dpg.add_text("Vel Type: 0 for No velocity field, 1 for the previously defined velocity")
    dpg.add_input_int(tag="vel_type", default_value=0, label=" Vel. Field Conditions")
    dpg.add_text("Done?")
    dpg.add_button(label="Run Simulation", callback=sim_callback)

# Prevents the Child program from relaunching the GUI
if mp.current_process().name == 'MainProcess':
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()