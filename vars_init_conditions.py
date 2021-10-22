"""Variables and initialization conditions for the main code
"""

# Domain size
x_min = -2
x_max = 2
y_min = -2
y_max = 2

t_max = 0.1  # simulation time in seconds
dt = 0.0005  # step size
N = 2 ** 15  # Number of particles
D = 0.01  # diffusivity
Nx = 64  # Euler grid size x
Ny = 64  # Euler grid size y


# Velocity field file name
# Note: import file object velocity field must have same minimum and maximum bounds or the velocity field will stretch
vel_file = 'velocityCMM3.dat'

# initial conditions
# 1: 1D problem
# 2: middle patch
# 3: side patch
init_type = 3

# Visualization type
# 1: Particles
# 2: Concentration field
viz_type = 1

# velocity
# 0: No velocity field
# 1: velocity field read from file defined earlier
vel_type = 1
