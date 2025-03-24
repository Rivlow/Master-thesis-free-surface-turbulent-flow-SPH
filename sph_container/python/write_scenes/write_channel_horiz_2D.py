import json
import os
import sys
import numpy as np

# Configure project path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
from python.write_scenes.Tools_scenes import * 

#-----------------------#
# SIMULATION PARAMETERS #
#-----------------------#

# General parameters
r = 0.01
U_0 = 5
t_end = 5
timeStepSize = 0.001

# Physical parameters
rho_0 = 1000
wall_friction = 0

# Sim parameters
sim2D = True
mapInvert = False # False: domain is outside unitbox (inside if True)
export = True
FPS = 100


#------- Pressure solver -------#
'''
0: Weakly compressible SPH for free surface flows (WCSPH)
1: Predictive-corrective incompressible SPH (PCISPH)
2: Position based fluids (PBF)
3: Implicit incompressible SPH (IISPH)
4: Divergence-free smoothed particle hydrodynamics (DFSPH)
5: Projective Fluids (dynamic boundaries not supported yet)
6: Implicit compressible SPH (ICSPH)
'''
simulationMethod = 4

#-------- Viscosity -----------#
'''
0: None
1: Standard
2: Bender and Koschier 2017
3: Peer et al. 2015
4: Peer et al. 2016
5: Takahashi et al. 2015 (improved)
6: Weiler et al. 2018
'''
viscosityMethod = 6                 
nu = 1e-6

#-------- Vorticity -----------#
'''
0: None
1: Micropolar model
2: Vorticity confinement
'''
vorticityMethod = 1
vorticity = 0.1
viscosityOmega = 0.05
inertiaInverse = 0.5

#-------- Boundary handling -----------#
'''
0: particle-based boundaries (Akinci et al. 2012)
1: density maps (Koschier et al. 2017)
2: volume maps (Bender et al. 2019)
'''
boundaryHandlingMethod = 2

#-------- Elasticity -----------#
'''
0: None
1: Becker et al. 2009
2: Peer et al. 2018
'''
elasticityMethod = 0

#--------- Drag method -----------#
'''
0: None
1: Macklin et al. 2014
2: Gissler et al. 2017
'''
dragMethod = 2
drag = 10.0
#-------------------------#
# GEOMETRIC CONFIGURATION #
#-------------------------#

# Channel dimensions
Lx_wall = 5.0
diameter = 1.0
Ly_wall = 0.1
Lz = 1.0  # Depth in z-direction

# Wall positions
top_wall_y = diameter / 2
bottom_wall_y = -diameter / 2



#-------------------#
# CREATE RECTANGLES #
#-------------------#

# Common parameters for all rigid bodies
common_params = {
    "geometryFile": "../models/UnitBox.obj",
    "isDynamic": False,
    "isWall": False,
    "mapInvert": mapInvert,
    "mapThickness": 0.0,
    "mapResolution": [100, 20, 10],
    "samplingMode": 1,
    "color": [0.1, 0.4, 0.6, 1.0]
}

# Create top and bottom walls
rigid_bodies = [
    # Bottom wall
    {
        **common_params,
        "translation": [0, bottom_wall_y - Ly_wall/2, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [Lx_wall, Ly_wall, Lz]
    },
    # Top wall
    {
        **common_params,
        "translation": [0, top_wall_y + Ly_wall/2, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [Lx_wall, Ly_wall, Lz]
    }
]

#---------------------------------#
# CREATE EMITTER AND FLUID BLOCK  #
#---------------------------------#


# Fluid block dimensions
fluid_x1 = -Lx_wall / 3 + 10*r  # Middle of the channel
fluid_y1 = bottom_wall_y + r    # Just above bottom wall
fluid_x2 = Lx_wall/30  # Extend to the right
fluid_y2 = top_wall_y - r # Just below top wall

# Define bounding box for particle reuse

# Emitter configuration
emitter_width = int(0.2/2*r)
emitter_height = int((fluid_y2 - fluid_y1)/(2*r)) - 1
emitter_x = -Lx_wall / 2   # Position just outside the left edge

margin = 3*r
box_min = [emitter_x - margin, bottom_wall_y - margin, -Lz/2 - margin]
box_max = [Lx_wall/2 - 0.3, top_wall_y + margin, Lz/2 + margin]

#-------------------------#
# DATA STRUCTURE CREATION #
#-------------------------#

data = {
    "Configuration": {
        "timeStepSize": timeStepSize,
        "sim2D": sim2D,
        "stopAt": t_end,
        "numberOfStepsPerRenderUpdate": 4,
        "enableVTKExport": export,
        "enableRigidBodyVTKExport": True,
        "dataExportFPS": FPS,
        "particleAttributes":"velocity;density;pressure",
        "particleRadius": r,
        "density0": rho_0,
        "simulationMethod": simulationMethod,
        "gravitation": [0, 0, 0],
        "cflMethod": 2,
        "cflFactor": 0.6,
        "cflMaxTimeStepSize": 0.01,
        "maxIterations": 1000,
        "maxError": 0.05,
        "maxIterationsV": 100,
        "maxErrorV": 0.1,
        "stiffness": 50000,
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
            "xsphBoundary": 0.2, 
            "xsph": 0.025,  
            "viscosityBoundary": 2,
            "dragMethod": dragMethod,
            "drag": drag,
            "vorticityMethod": vorticityMethod,
            "vorticity": vorticity, 
            "viscosityOmega": viscosityOmega,
            "inertiaInverse": inertiaInverse,
            "colorMapType": 1,
            "emitterReuseParticles": True,
            "maxEmitterParticles": 10000000,
            "emitterBoxMin": box_min,
            "emitterBoxMax": box_max
        }
    ],

    "AnimationFields": [
        {
            "particleField": "velocity",
            "shapeType": 0,
            "translation": [-Lx_wall/2 + 0.2, 0.0, 0.0], 
            "scale": [0.4, 0.95, 1.0],  
            "expression_x": "5.0",     
            "expression_y": "0.0",
            "expression_z": "0.0",
            "startTime": 0,
            "endTime": 5.0              
        }
    ],
    "FluidBlocks": [
        {
            "denseMode": 0,
            "start": [fluid_x1, fluid_y1, -Lz/2 + r],
            "end": [fluid_x2, fluid_y2, Lz/2 - r],
            "scale": [1, 1, 1]
        }
    ],
    
    "Emitters": [
        {
            "width": emitter_width,
            "height": emitter_height,
            "translation": [emitter_x, 0, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0,
            "velocity": U_0,
            "type": 0,
            "emitEndTime": t_end
        }
    ],
    "RigidBodies": rigid_bodies
}


   
 
#--------------------------------------#
# DISPLAYING INFORMATION AND EXPORTING #
#--------------------------------------#

# Display simulation information
print("\n---------- Channel Information ----------")
print(f"Channel length: {Lx_wall} m")
print(f"Channel height: {diameter} m")
print(f"Wall thickness: {Ly_wall} m")

print("\n---------- Emitter Configuration ----------")
print(f"Position: [{emitter_x}, 0, 0]")
print(f"Physical dimensions: width = {emitter_width}m, height = {emitter_height}m")
print(f"Particles: width = {int(emitter_width / (2 * r))}, height = {int(emitter_height / (2 * r))}")
print(f"Velocity: {U_0} m/s")

print("\n---------- Fluid Block Configuration ----------")
print(f"Start: [{fluid_x1}, {fluid_y1}, {-Lz/2 + r}]")
print(f"End: [{fluid_x2}, {fluid_y2}, {Lz/2 - r}] \n")

# Define output paths
json_path = "Code/data/Scenes/channel_horiz_2D.json"
output_path = "Code/bin/output/channel_horiz_2D"

# Clean output directory
clean_files(output_path)

# Ensure directory exists
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# Write to JSON file
with open(json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data written to '{json_path}'")