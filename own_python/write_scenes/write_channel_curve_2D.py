# IMPORTS AND SETUP
import json
import os
import sys
import numpy as np

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *

#--------------------------#
#   SIMULATION PARAMETERS  #
#--------------------------#
r = 0.02                  # Particle radius
U_0 = 5                   # Initial velocity
t_end = 30                # End time
timeStepSize = 0.001      # Time step
FPS = 25                  # Export FPS
clean_output = True       # Clean output folder

# PHYSICAL PARAMETERS
rho_0 = 1000              # Reference density
sim2D = True              # 2D simulation
mapInvert = False         # Invert domain
attr = "velocity;angular velocity;p / rho^2;density"  # Export attributes

# SPH PARAMETERS
xsph_fluid = 0.2          # Fluid XSPH
xsph_boundary = 0.7       # Boundary XSPH
viscosity_boudary = 2     # Boundary viscosity

# NUMERICAL METHODS
simulationMethod = 4      # DFSPH
viscosityMethod = 6       # Weiler et al. 2018
nu = 1e-6                 # Viscosity coefficient
visc_boundary = 2
vorticityMethod = 1       # Micropolar model
vorticity = 0.25
viscosityOmega = 0.1
inertiaInverse = 0.5
boundaryHandlingMethod = 2 # Volume maps
elasticityMethod = 0       # None
dragMethod = 2             # Gissler et al. 2017
drag = 1.0


#----------------------------#
#   GEOMETRIC CONFIGURATION  #
#----------------------------#

Ly = 0.1                   # Half-wall height
Lz = 1.0                   # Depth
Lx_narrowing = 3.0         # Narrowing section length
Lx_horiz = 25.0            # Horizontal section length
x_start = -2.0             # Initial x position
Ly_init = 2                # Initial half-width

# ANGLE CONFIGURATION
alpha_tot = np.radians(15) # Total narrowing angle
nb_narrow = 5              # Number of narrowing elements
nb_horiz = 2               # Number of horizontal elements

# Calculate x length for narrowing
Lx = calculate_lx_for_total_length(Lx_narrowing, nb_narrow, alpha_tot)

# CREATE GEOMETRY
bottom_narrowing, top_narrowing = create_narrowing(Lx, Ly, Lz, alpha_tot, nb_narrow, x_start, Ly_init)
bottom_straight, top_straight = create_straight_channel(Lx_horiz, Ly, Lz, bottom_narrowing, top_narrowing, nb_horiz)

# Calculate total length
first_x = bottom_narrowing[0]["translation"][0] - (Lx/2)
last_x = bottom_narrowing[-1]["translation"][0] + (Lx/2)
Lx_tot_narrow = last_x - first_x
Lx_tot_horiz = nb_horiz * Lx_horiz
total_length = Lx_tot_narrow + Lx_tot_horiz

# Common parameters for rigid bodies
common_params = {
    "geometryFile": "../models/UnitBox.obj",
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": False,
    "mapInvert": mapInvert,
    "mapThickness": 0.0,
    "mapResolution": [60, 10, 10],
    "samplingMode": 1,
    "color": [0.1, 0.4, 0.6, 1.0]
}

# Create rigid bodies
rigid_bodies = add_rectangles(common_params, bottom_narrowing, top_narrowing, bottom_straight, top_straight)

# EMITTER AND FLUID BLOCK
# Check final diameter
D = 2 * (top_narrowing[-1]["translation"][1] - bottom_narrowing[-1]["translation"][1]) - Ly

# Emitter configuration
emitter_distance = 0
emitter_x = bottom_narrowing[0]["translation"][0] - Lx/2 - emitter_distance
emitter_width = 0.5
emitter_height = 2*Ly_init

emitter_config = {
    "physical_width": emitter_width,
    "physical_height": emitter_height,
    "width": int(emitter_width / (2 * r)),
    "height": int(emitter_height / (2 * r)),
    "translation": [emitter_x, 0, 0],
    "rotation_axis": [0, 0, 1],
    "rotation_angle": 0,
    "velocity": U_0,
    "type": 0,
    "end_time": 4
}

# Bounding box for particle reuse
leftmost_x = emitter_x - emitter_width/2
rightmost_x = bottom_straight[-1]["translation"][0] + Lx_horiz/2
bottom_y = bottom_narrowing[0]["translation"][1] - Ly/2
top_y = top_narrowing[0]["translation"][1] + Ly/2

margin = 5*r
box_min = [leftmost_x - margin, bottom_y - margin, -Lz/2 - margin]
box_max = [rightmost_x*0.95 + margin, top_y + margin, Lz/2 + margin]

# Fluid block dimensions
fluid_x1 = bottom_narrowing[-1]["translation"][0] + Lx/2
fluid_y1 = bottom_straight[0]["translation"][1] + Ly/2 + 2*r
fluid_x2 = bottom_straight[-1]["translation"][0] + Lx_horiz/2
fluid_y2 = top_straight[0]["translation"][1] - Ly/2 - 2*r

# SIMULATION DATA STRUCTURE
data = {
    "Configuration": {
        "particleAttributes": attr,
        "timeStepSize": timeStepSize,
        "sim2D": sim2D,
        "stopAt": t_end,
        "numberOfStepsPerRenderUpdate": 4,
        "enableVTKExport": True,
        "enableRigidBodyVTKExport": True,
        "dataExportFPS": FPS,
        "particleRadius": r,
        "density0": rho_0,
        "simulationMethod": simulationMethod,
        "gravitation": [0, 0, 0],
        "cflMethod": 2,
        "cflFactor": 0.6,
        "cflMaxTimeStepSize": 0.01,
        "maxIterations": 2000,
        "maxError": 0.005,
        "maxIterationsV": 500,
        "maxErrorV": 0.01,
        "stiffness": 100000,
        "exponent": 7,
        "velocityUpdateMethod": 0,
        "enableDivergenceSolver": True,
        "boundaryHandlingMethod": boundaryHandlingMethod
    },

    "Materials": [
        {
            "id": "Fluid",
            "viscosityMethod": viscosityMethod,
            "viscosity": nu,
            "xsphBoundary": xsph_boundary,
            "xsph": xsph_fluid,
            "viscosityBoundary": visc_boundary,
            "dragMethod": dragMethod,
            "drag": drag,
            "vorticityMethod": vorticityMethod,
            "vorticity": vorticity,
            "viscosityOmega": viscosityOmega,
            "inertiaInverse": inertiaInverse,
            "colorMapType": 1,
            "maxEmitterParticles": 10000000,
            "emitterReuseParticles": True,
            "emitterBoxMin": box_min,
            "emitterBoxMax": box_max
        }
    ],
    
    "Emitters": [
        {
            "width": int(emitter_width / (2 * r)),
            "height": int(emitter_height / (2 * r)),
            "translation": [emitter_x, 0, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0,
            "velocity": U_0,
            "type": 0,
            "emitEndTime": t_end
        }
    ],
    
    "FluidBlocks": [
        {
            "denseMode": 0,
            "start": [fluid_x1, fluid_y1, -0.5],
            "end": [fluid_x2, fluid_y2, 0.5],
            "scale": [1, 1, 1]
        }
    ],
    
    "RigidBodies": rigid_bodies
}

# DISPLAY AND EXPORT
# Print simulation information
print("\n--- Channel Information ---")
print(f"Initial width: {2 * Ly_init:.3f} m")
print(f"Final width: {D:.3f} m")
print(f"Narrowing ratio: {2 * Ly_init / D:.2f}")
print(f"Target narrowing length: {Lx_narrowing} m")
print(f"Actual narrowing length: {Lx_tot_narrow} m")
print(f"Total channel length: {total_length} m")

print("\n--- Emitter Configuration ---")
print(f"Position: {emitter_config['translation']}")
print(f"Dimensions: width = {emitter_config['physical_width']}m, height = {emitter_config['physical_height']}m")
print(f"Velocity: {emitter_config['velocity']} m/s")

print("\n--- Fluid Block Configuration ---")
print(f"Start: [{fluid_x1}, {fluid_y1}, -0.5]")
print(f"End: [{fluid_x2}, {fluid_y2}, 0.5]")

print("\n--- Particle Reuse Bounding Box ---")
print(f"Min: {box_min}")
print(f"Max: {box_max}\n")

# Define output paths
json_path = "SPlisHSPlasH/data/Scenes/channel_curve_2D.json"
output_path = "SPlisHSPlasH/bin/output/channel_curve_2D"

# Clean output directory
if clean_output:
    clean_files(output_path)

# Ensure directory exists
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# Write JSON file
with open(json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data written to '{json_path}'")