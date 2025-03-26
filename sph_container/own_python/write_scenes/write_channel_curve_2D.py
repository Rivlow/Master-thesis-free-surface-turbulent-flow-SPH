import json
import os
import sys
import numpy as np

# Configure project path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
from own_python.write_scenes.Tools_scenes import * 

#-----------------------#
# SIMULATION PARAMETERS #
#-----------------------#

# General parameters
r = 0.02
U_0 = 5
t_end = 30
timeStepSize = 0.001
xsph_fluid = 0.2
xsph_boundary = 0.7
viscosity_boudary = 2

# Physical parameters
rho_0 = 1000

# Sim parameters
sim2D = True
mapInvert = False # False: domain is outside unitbox (inside if True)
FPS = 20
clean_output = True

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
nu = 0.5*1e-4

#-------- Vorticity -----------#
'''
0: None
1: Micropolar model
2: Vorticity confinement
'''
vorticityMethod = 1
vorticity = 0.25
viscosityOmega = 0.1
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
drag = 1.0

#-------------------------#
# GEOMETRIC CONFIGURATION #
#-------------------------#

# Wall dimensions
Ly = 0.1                   
Lz = 1.0                   
Lx_narrowing = 3.0          
Lx_horiz = 15.0          

x_start = -2.0         # initial x position of channel entry     
Ly_init = 2            # half distance between right narrow elem 

# Angle configuration
alpha_tot = np.radians(15)    
nb_narrow = 5               # nb elements for narrow section
nb_horiz = 2                  # nb elements for horiz section

Lx = calculate_lx_for_total_length(Lx_narrowing, nb_narrow, alpha_tot)

#-------------------#
# CREATE RECTANGLES #
#-------------------#

bottom_narrowing, top_narrowing = create_narrowing(Lx, Ly, Lz, alpha_tot, nb_narrow, x_start, Ly_init)
bottom_straight, top_straight = create_straight_channel(Lx_horiz, Ly, Lz, bottom_narrowing, top_narrowing, nb_horiz)

# Calculate total length (narrow section)
first_x = bottom_narrowing[0]["translation"][0] - (Lx/2)
last_x = bottom_narrowing[-1]["translation"][0] + (Lx/2)
Lx_tot_narrow = last_x - first_x
print(f"Target narrowing length: {Lx_narrowing} m")
print(f"Actual narrowing length: {Lx_tot_narrow} m")

Lx_tot_horiz = nb_horiz * Lx_horiz
total_length = Lx_tot_narrow + Lx_tot_horiz
print(f"Total channel length: {total_length} m")

# Common parameters for all rigid bodies
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

rigid_bodies = add_rectangles(common_params, bottom_narrowing, top_narrowing, bottom_straight, top_straight)

#---------------------------------#
# CREATE EMITTER AND FLUID BLOCK  #
#---------------------------------#

# Check final diameter D (in horiz section)
D = 2 * (top_narrowing[-1]["translation"][1] - bottom_narrowing[-1]["translation"][1]) - Ly

# Emitter dimensions/position
emitter_distance = 0  # ox [m]
emitter_x = bottom_narrowing[0]["translation"][0] - Lx/2 - emitter_distance
emitter_width = 0.5  # Width [m]
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

# Define bounding box for particle reuse based on actual geometry
# Calculate the leftmost, rightmost, bottom and top coordinates
leftmost_x = emitter_x - emitter_width/2
rightmost_x = bottom_straight[-1]["translation"][0] + Lx_horiz/2

bottom_y = bottom_narrowing[0]["translation"][1] - Ly/2
top_y = top_narrowing[0]["translation"][1] + Ly/2

# Add some margin to ensure we don't clip particles too early
margin = 5*r
box_min = [leftmost_x - margin, bottom_y - margin, -Lz/2 - margin]
box_max = [rightmost_x*0.95 + margin, top_y + margin, Lz/2 + margin]

print(f"\n---------- Particle Bounding Box ----------")
print(f"Box min: {box_min}")
print(f"Box max: {box_max}")

# Fluid block dimensions
fluid_x1 = bottom_narrowing[-1]["translation"][0] + Lx/2  
fluid_y1 = bottom_straight[0]["translation"][1] + Ly/2 + r  
fluid_x2 = bottom_straight[-1]["translation"][0] + Lx_horiz/2 
fluid_y2 = top_straight[0]["translation"][1] - Ly/2 - r  

#-------------------------#
# DATA STRUCTURE CREATION #
#-------------------------#

data = {
    "Configuration": {
        "particleAttributes": "velocity;angular velocity;pressure;density;velocity difference",
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
        "maxIterations": 1000,
        "maxError": 0.01,
        "maxIterationsV": 100,
        "maxErrorV": 0.05,
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
            "viscosityBoundary": 1.0,
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

#--------------------------------------#
# DISPLAYING INFORMATION AND EXPORTING #
#--------------------------------------#

# Display simulation information
#print_simulation_info(rigid_bodies, Lx, nb_narrow)
print("\n---------- Channel Information ----------")
print(f"Initial channel width: {2 * Ly_init:.3f} m")
print(f"Final channel width: {D:.3f} m")
print(f"Narrowing ratio: {2 * Ly_init / D:.2f}")

print("\n---------- Emitter Configuration ----------")
print(f"Position: {emitter_config['translation']}")
print(f"Physical dimensions: width = {emitter_config['physical_width']}m, height = {emitter_config['physical_height']}m")
print(f"Velocity: {emitter_config['velocity']} m/s")

print("\n---------- Fluid Block Configuration ----------")
print(f"Start: [{fluid_x1}, {fluid_y1}, -0.5]")
print(f"End: [{fluid_x2}, {fluid_y2}, 0.5] \n")

# Define output paths
json_path = "Code/data/Scenes/channel_curve_2D.json"
output_path = "Code/bin/output/channel_curve_2D"

# Clean output directory
if clean_output:clean_files(output_path)


# Ensure directory exists
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# Write to JSON file
with open(json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data written to '{json_path}'")