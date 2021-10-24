"""Variables and initialization conditions for the main code
"""

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

t_max = 0.2 # simulation time in seconds
dt = 0.0005  # step size
N = 2 ** 12  # Number of particles
D = 0.1  # diffusivity
Nx = 16  # Euler grid size x
Ny = 16  # Euler grid size y

# Velocity field file name
# Note: import file object velocity field must have same minimum and maximum bounds or the velocity field will stretch
vel_file = 'velocityCMM3.dat'

# initial conditions
# 1: 1D problem (overrides y_min, y_max, Ny, and vel_type)
# 2: middle patch
# 3: side patch
init_type = 1

# Visualization type
# 1: Particles
# 2: Concentration field
viz_type = 1

# velocity
# 0: No velocity field
# 1: velocity field read from file defined earlier
vel_type = 0
