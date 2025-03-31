import json
import os
import sys
import numpy as np

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *


#---------------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#---------------------------------------------------------------------------------------

# General parameters
r = 0.01                  # Particle radius
sim2D = True               # 2D simulation
timeStepSize = 0.0005       # Time step
mapInvert = False          # Map inversion
angle = np.radians(30)     # Inclination angle of oblique obstacles (30 degrees)
U = 38.46

#---------------------------------------------------------------------------------------
# GEOMETRIC CONFIGURATION
#---------------------------------------------------------------------------------------

# Wall dimensions
Lx = 3.0                   # Length
Ly = 0.1                   # Thickness
Lz = 1.0                   # Depth
Lx2 = 1.5 * Lx             # Length of horizontal walls

# Oblique obstacles positions
ox, oy, oz = 1.85, 1.0, 0.0
obstacle_bottom_pos = [-ox, -oy, oz]  # Position of bottom obstacle
obstacle_top_pos = [-ox, oy, oz]      # Position of top obstacle

# Obstacles rotation angles
angle_bottom = angle       # Positive angle for bottom obstacle
angle_top = -angle         # Negative angle for top obstacle

#---------------------------------------------------------------------------------------
# CALCULATING HORIZONTAL WALL POSITIONS
#---------------------------------------------------------------------------------------
def calculate_wall_position(pos, angle, Lx, Lx2, Ly):
    new_x = pos[0] + (Lx/2) * np.cos(angle) - (Ly/2) * np.sin(angle) + Lx2/2
    return new_x

# Position of the bottom horizontal wall
x_wall_bottom = calculate_wall_position(obstacle_bottom_pos, angle_bottom, Lx, Lx2, Ly)
y_wall_bottom = obstacle_bottom_pos[1] + (Lx/2) * np.sin(angle_bottom) + (Ly/2) * np.cos(angle_bottom) - Ly/2
wall_bottom_pos = [round(x_wall_bottom, 2), round(y_wall_bottom, 2), 0]

# Position of the top horizontal wall
x_wall_top = calculate_wall_position(obstacle_top_pos, angle_bottom, Lx, Lx2, Ly)
y_wall_top = obstacle_top_pos[1] + (Lx/2) * np.sin(angle_top) - (Ly/2) * np.cos(angle_top) + Ly/2
wall_top_pos = [round(x_wall_top, 2), round(y_wall_top, 2), 0]

#---------------------------------------------------------------------------------------
# CALCULATING FLUID BLOCK POSITION
#---------------------------------------------------------------------------------------

x1 = round(wall_bottom_pos[0] - Lx2/2, 2)
y1 = round(wall_bottom_pos[1] + Ly/2 + r, 2)
z1 = 0

x2 = round(wall_top_pos[0] + Lx2/2, 2)
y2 = round(wall_top_pos[1] - Ly/2 - r, 2)
z2 = 0

print(f"Bottom wall position: {wall_bottom_pos}")
print(f"Fluid block - Point 1: ({x1}, {y1}, {z1})")
print(f"Fluid block - Point 2: ({x2}, {y2}, {z2})")

#---------------------------------------------------------------------------------------
# EMITTER CONFIGURATION
#---------------------------------------------------------------------------------------

# Calculate emitter position and dimensions
emitter_x = round(-ox - (Lx/2)*np.cos(angle) - (Ly/2)*np.sin(angle), 2)
emitter_height = round(2*(oy + (Lx/2)*np.sin(angle) - (Ly/2)*np.cos(angle) - 5*r), 2)

emitter_config = {
    "physical_width": 0.5,                      # Width in meters
    "physical_height": emitter_height,          # Height in meters
    "width": int(0.5 / (2 * r)),                # Width in number of particles
    "height": int(emitter_height / (2 * r)),    # Height in number of particles
    "translation": [emitter_x, 0, 0],           # Position
    "rotation_axis": [1, 0, 0],                 # Rotation axis
    "rotation_angle": 0,                        # Rotation angle
    "velocity": U,                              # Initial particle velocity
    "type": 0,                                  # Emitter type (box)
    "end_time": 4                               # Emission time in seconds
}

# Limits for particle reuse
box_min = [-10, -10, -10]
box_max = [10, 10, 10]

#---------------------------------------------------------------------------------------
# RIGID BODIES DEFINITION
#---------------------------------------------------------------------------------------

# Common parameters for all rigid bodies
common_params = {
    "geometryFile": "../models/UnitBox.obj",
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": False,
    "mapInvert": mapInvert,
    "mapThickness": 0.0,
    "mapResolution": [60, 6, 10],
    "samplingMode": 1
}

# Rigid bodies definition
rigid_bodies = [
    # Bottom wall
    {
        **common_params,
        "translation": wall_bottom_pos,
        "rotationAxis": [1, 0, 0],
        "rotationAngle": 0,
        "scale": [Lx2, Ly, Lz]
    },
    # Top wall
    {
        **common_params,
        "translation": wall_top_pos,
        "rotationAxis": [1, 0, 0],
        "rotationAngle": 0,
        "scale": [Lx2, Ly, Lz]
    },
    # Bottom-left obstacle
    {
        **common_params,
        "translation": obstacle_bottom_pos,
        "rotationAxis": [0, 0, 1],
        "rotationAngle": angle_bottom,
        "scale": [Lx, Ly, Lz]
    },
    # Top-left obstacle
    {
        **common_params,
        "translation": obstacle_top_pos,
        "rotationAxis": [0, 0, 1],
        "rotationAngle": angle_top,
        "scale": [Lx, Ly, Lz]
    }
]

#---------------------------------------------------------------------------------------
# DATA STRUCTURE CREATION
#---------------------------------------------------------------------------------------

data = {
    "Configuration": {
        "timeStepSize": timeStepSize,
        "sim2D": sim2D,
        "stopAt": 4,
        "numberOfStepsPerRenderUpdate": 4,
        "enableVTKExport": True,
        "enableRigidBodyVTKExport": True,
        "dataExportFPS": 1000,
        "particleRadius": r,
        "density0": 1000,
        "simulationMethod": 4,
        "gravitation": [0, 0, 0],
        "cflMethod": 2,
        "cflFactor": 0.5,
        "cflMaxTimeStepSize": 0.0005,
        "maxIterations": 100,
        "maxError": 0.05,
        "maxIterationsV": 100,
        "maxErrorV": 0.1,
        "stiffness": 50000,
        "exponent": 7,
        "velocityUpdateMethod": 0,
        "enableDivergenceSolver": True,
        "boundaryHandlingMethod": 2
    },

    "Materials": [
        {
            "id": "Fluid",
            "viscosityMethod": 6,
            "viscosity": 0.01,
            "dragMethod": 2,
            "drag": 10.0,
            "vorticityMethod ":1,
            "vorticity": 0.1, 
			"viscosityOmega": 0.05,
			"inertiaInverse": 0.5,
            "colorMapType": 1,
            "maxEmitterParticles": 100000,
            "emitterReuseParticles": True,
            "emitterBoxMin": box_min,
            "emitterBoxMax": box_max
        }
    ],

    "Emitters": [
        {
            "width": emitter_config["width"],
            "height": emitter_config["height"],
            "translation": emitter_config["translation"],
            "rotationAxis": emitter_config["rotation_axis"],
            "rotationAngle": emitter_config["rotation_angle"],
            "velocity": emitter_config["velocity"],
            "type": emitter_config["type"],
            "emitEndTime": emitter_config["end_time"]
        }
    ],
    
    "FluidBlocks": [
        {
            "denseMode": 0,
            "start": [x1, y1, -0.5],
            "end": [x2, y2, 0.5],
            "scale": [1, 1, 1]
        }
    ],
    
    "RigidBodies": rigid_bodies
}

#---------------------------------------------------------------------------------------
# DISPLAYING INFORMATION AND EXPORTING
#---------------------------------------------------------------------------------------

# Display simulation information
print_simulation_info(emitter_config, rigid_bodies)

# Define output paths
json_path = "Code/data/Scenes/channel_oblique_2D.json"
output_path = "Code/bin/output/channel_oblique_2D"

# Clean output directory
clean_files(output_path)

# Ensure directory exists
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# Write to JSON file
with open(json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data written to '{json_path}'")