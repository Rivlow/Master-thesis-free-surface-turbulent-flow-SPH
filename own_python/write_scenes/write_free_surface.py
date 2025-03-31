#-------------------------------#
# 1. IMPORTS AND CONFIGURATION #
#-------------------------------#
import json
import os
import sys
import numpy as np

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *


#-----------------------------#
# 2. SIMULATION PARAMETERS    #
#-----------------------------#

# 2.1 General parameters
r = 0.01                   # Particle radius
U_0 = 2                     # Initial velocity
t_end = 15                  # Simulation end time
timeStepSize = 0.001        # Time step size
FPS = 40                    # Frames per second for export
clean_output = True         # Clean output directory

# 2.2 Physical parameters
rho_0 = 1000                # Reference density

# 2.3 Simulation parameters
sim2D = True                # 2D simulation
mapInvert = False           # Domain is outside the unit box
attr = "velocity;angular velocity;p / rho^2;density"  # Exported attributes

# 2.4 SPH parameters
xsph_fluid = 0.2            # XSPH for fluid
xsph_boundary = 0.7         # XSPH for boundaries
viscosity_boundary = 2      # Boundary viscosity


#-------------------------------#
# 3. NUMERICAL METHODS         #
#-------------------------------#

# 3.1 Pressure solver method (DFSPH = 4)
simulationMethod = 4  # Divergence-free smoothed particle hydrodynamics (DFSPH)

# 3.2 Viscosity method (Weiler et al. 2018 = 6)
viscosityMethod = 6                 
nu = 1e-6

# 3.3 Vorticity method (Micropolar model = 1)
vorticityMethod = 1
vorticity = 0.25
viscosityOmega = 0.1
inertiaInverse = 0.5

# 3.4 Boundary handling method (Volume maps = 2)
boundaryHandlingMethod = 2

# 3.5 Elasticity method (None = 0)
elasticityMethod = 0

# 3.6 Drag method (Gissler et al. 2017 = 2)
dragMethod = 2
drag = 1.0


#-------------------------------#
#  4. GEOMETRIC CONFIGURATION  #
#-------------------------------#

# 4.1 Rectangle dimensions
Ly = 0.1                    # Half-height of walls
Lz = 1.0                    # Depth
Lx_1 = 1.0                  # Length of first rectangle
Lx_2 = 1 * Lx_1             # Length of second rectangle

# 4.2 Circle parameters
Radius = 0.205              # Radius of the sphere
H = 0.05                    # Vertical offset parameter

# 4.3 Calculate positions
trans_first_rect = [Lx_1/2, -Ly/2, 0]                                          # First rectangle position
trans_circle = [Lx_1 + np.sqrt(Radius**2 - (H/2)**2), 0, 0]                    # Circle position
trans_second_rect = [1.5*Lx_1 + 2*np.sqrt(Radius**2 - (H/2)**2), -Ly/2, 0]     # Second rectangle position

# 4.4 Create rigid bodies array with mixed geometries
RigidBodies = [
    # First rectangle
    {
        "geometryFile": "../models/UnitBox.obj",
        "translation": trans_first_rect,
        "rotationAxis": [1, 0, 0],
        "rotationAngle": 0,
        "scale": [Lx_1, Ly, Lz],
        "color": [0.1, 0.4, 0.6, 1.0], 
        "isDynamic": False,
        "isWall": False,
        "mapInvert": mapInvert,
        "mapThickness": 0.0,
        "mapResolution": [20, 20, 20],
        "samplingMode": 1
    },
    # Second rectangle
    {
        "geometryFile": "../models/UnitBox.obj",
        "translation": trans_second_rect,
        "rotationAxis": [1, 0, 0],
        "rotationAngle": 0,
        "scale": [Lx_2, Ly, Lz],
        "color": [0.1, 0.4, 0.6, 1.0], 
        "isDynamic": False,
        "isWall": False,
        "mapInvert": mapInvert,
        "mapThickness": 0.0,
        "mapResolution": [20, 20, 20],
        "samplingMode": 1
    },
    # Circle
    {
        "geometryFile": "../models/sphere.obj",
        "translation": trans_circle,
        "rotationAxis": [1, 0, 0],
        "rotationAngle": np.radians(90),
        "scale": [Radius, Radius, Radius],
        "color": [0.1, 0.4, 0.6, 1.0], 
        "isDynamic": False,
        "isWall": False,
        "mapInvert": mapInvert,
        "mapThickness": 0.0,
        "mapResolution": [20, 20, 20],
        "samplingMode": 1
    }
]


#---------------------------#
# 5. FLUID CONFIGURATION   #
#---------------------------#

# 5.1 First fluid block dimensions
fluid_x11, fluid_y11 = Lx_1/2, 0
fluid_x21, fluid_y21 = Lx_1 - 2*r, 1.2*(Radius - H)

# 5.2 Calculate scene bounds with margin for the weir
box_min, box_max = calculate_scene_bounds(RigidBodies, margin=5*r)
rightmost_x = box_max[0] - (2*r)*10
width_of_weir = 5 * (2*r)*10

# 5.3 Second fluid block dimensions
fluid_x12, fluid_y12 = Lx_1 + 2*np.sqrt(Radius**2 - (H/2)**2), 0
fluid_x22, fluid_y22 = rightmost_x - 30*(2*r), 1.2*(Radius - H)

# 5.4 Add weir to rigid bodies
RigidBodies.append({
    "geometryFile": "../models/UnitBox.obj",
    "translation": [rightmost_x, 0.165, 0],  # 0.165 = 0.33/2 car le centre est à mi-hauteur
    "rotationAxis": [1, 0, 0],
    "rotationAngle": 0,
    "scale": [width_of_weir, 0.33, Lz],
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": False,
    "mapInvert": mapInvert,
    "mapThickness": 0.0,
    "mapResolution": [20, 20, 20],
    "samplingMode": 1
})


#-------------------------------#
# 6. EMITTER CONFIGURATION     #
#-------------------------------#

# 6.1 Emitter dimensions
emit_H = 0.41                 # Emitter height
emit_L = (2*r)*2              # Emitter width

# 6.2 Emitter setup
emitter_config = {
    "physical_width": emit_L,             
    "physical_height": emit_H,          
    "width": int(emit_L / (2 * r)),      
    "height": int(emit_H / (2 * r)),   
    "translation": [0, emit_H/2 + 4*r, 0],           
    "velocity": U_0,                           
    "type": 0,                                   
    "end_time": 4                               
}


#-------------------------------#
# 7. CALCULATE FINAL BOUNDS    #
#-------------------------------#

# Recalculate scene bounds with all geometries including the weir
final_box_min, final_box_max = calculate_scene_bounds(RigidBodies, margin=10*r)

# Add some padding to the bounds (extra 20% in each direction)
padding = [(final_box_max[i] - final_box_min[i]) * 0.2 for i in range(3)]
emitter_box_min = [final_box_min[i] - padding[i] for i in range(3)]
emitter_box_max = [final_box_max[i] + padding[i] for i in range(3)]

print("\n---------- Fluid Block Configuration ----------")
print(f"First fluid block - Start: [{fluid_x11}, {fluid_y11}, -0.5], End: [{fluid_x21}, {fluid_y21}, 0.5]")
print(f"Second fluid block - Start: [{fluid_x12}, {fluid_y12}, -0.5], End: [{fluid_x22}, {fluid_y22}, 0.5]")

print(f"\n---------- Scene Bounding Box ----------")
print(f"Box min: {final_box_min}")
print(f"Box max: {final_box_max}")
print(f"Emitter box min: {emitter_box_min}")
print(f"Emitter box max: {emitter_box_max}")


#-------------------------------#
# 8. DATA STRUCTURE CREATION   #
#-------------------------------#

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
        "gravitation": [0, -9.81, 0],
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
            "viscosityBoundary": viscosity_boundary,
            "dragMethod": dragMethod,
            "drag": drag,
            "vorticityMethod": vorticityMethod,
            "vorticity": vorticity, 
            "viscosityOmega": viscosityOmega,
            "inertiaInverse": inertiaInverse,
            "colorMapType": 1,
            "maxEmitterParticles": 10000000,
            "emitterReuseParticles": True,
            "emitterBoxMin": emitter_box_min,
            "emitterBoxMax": emitter_box_max
        }
    ],
    
    "Emitters": [
        {
            "width": emitter_config["width"],
            "height": emitter_config["height"],
            "translation": emitter_config["translation"],
            "velocity": emitter_config["velocity"],
            "type": emitter_config["type"],
            "emitEndTime": t_end
        }
    ],

    "FluidBlocks": [
        {
            "denseMode": 0,
            "start": [fluid_x11, fluid_y11, -0.5],
            "end": [fluid_x21, fluid_y21, 0.5],
            "scale": [1, 1, 1]
        },
        {
            "denseMode": 0,
            "start": [fluid_x12, fluid_y12, -0.5],
            "end": [fluid_x22, fluid_y22, 0.5],
            "scale": [1, 1, 1]
        }
    ],
    
    "RigidBodies": RigidBodies
}


#-------------------------------#
# 9. DISPLAY AND EXPORT        #
#-------------------------------#

# 9.1 Define output paths
json_path = "Code/data/Scenes/free_surface_2D.json"
output_path = "Code/bin/output/free_surface_2D"

# 9.2 Clean output directory
if clean_output:
    clean_files(output_path)

# 9.3 Ensure directory exists
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# 9.4 Write JSON file
with open(json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

# 9.5 Display simulation information
print(f"\n---------- Geometry Information ----------")
print(f"Rectangle dimensions: Length = {Lx_1}m, Height = {2*Ly}m, Depth = {Lz}m")
print(f"Sphere radius: {Radius}m")
print(f"Distance between rectangle centers: {trans_second_rect[0] - trans_first_rect[0]}m")

print(f"\n---------- Emitter Configuration ----------")
print(f"Position: {emitter_config['translation']}")
print(f"Physical dimensions: width = {emitter_config['physical_width']}m, height = {emitter_config['physical_height']}m")
print(f"Velocity: {emitter_config['velocity']} m/s")

print(f"\nData written to '{json_path}'")