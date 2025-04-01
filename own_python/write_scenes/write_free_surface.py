import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import * 


def main():
    #-----------------------------#
    #    SIMULATION PARAMETERS    #
    #-----------------------------#
    
    r = 0.005                   # Particle radius
    U_0 = 2                     # Initial velocity
    t_end = 15                  # Simulation end time
    timeStepSize = 0.001        # Time step size
    FPS = 40                    # Frames per second for export
    clean_output = True         # Clean output directory
    
    rho_0 = 1000                # Reference density
    
    sim2D = True                # 2D simulation
    mapInvert = False           # Domain is outside the unit box
    attr = "velocity;angular velocity;p / rho^2;density"  # Exported attributes
    
    xsph_fluid = 0.2            # XSPH for fluid
    xsph_boundary = 0.0         # XSPH for boundaries
    viscosity_boundary = 0.0    # Boundary viscosity
    
    simulationMethod = 4        # DFSPH
    viscosityMethod = 6         # Weiler et al. 2018
    nu = 1e-6                   # Kinematic viscosity
    vorticityMethod = 1         # Micropolar model
    vorticity = 0.25            # Vorticity coefficient
    viscosityOmega = 0.1        # Angular viscosity
    inertiaInverse = 0.5        # Inverse inertia
    boundaryHandlingMethod = 2  # Volume maps
    elasticityMethod = 0        # No elasticity
    dragMethod = 2              # Gissler et al. 2017
    drag = 1.0                  # Drag coefficient
    
    #-------------------------------#
    #     GEOMETRIC CONFIGURATION   #
    #-------------------------------#
    
    Ly = 0.1                    # Half-height of walls
    Lz = 1.0                    # Depth
    Lx_1 = 3.0                  # Length of first rectangle
    Lx_2 = 3.0                  # Length of second rectangle
    
    nb_elem = 60                # Number of segments for parabola (increased for better precision)
    parabola_start = 3.0        # Start position of parabola (matching Lx_1)
    parabola_end = 4.0          # End position of parabola
    
    trans_first_rect = [Lx_1/2, -Ly/2, 0]                                          
    trans_second_rect = [parabola_end + Lx_2/2, -Ly/2, 0]
    
    RigidBodies = [
        # First rectangle (left flat section)
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
            "mapResolution": [30, 20, 20],
            "samplingMode": 1
        },
        # Second rectangle (right flat section)
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
        }
    ]
    
    x = np.linspace(parabola_start, parabola_end, nb_elem+1)
    z = parabole(np.linspace(7, 13, nb_elem+1)) 
    
    segments = []
    
    # Each segment needs 4 points (bottom-left, top-left, top-right, bottom-right)
    for i in range(len(x)-1):
        # Calculate angle of the segment
        angle = np.arctan2(z[i+1] - z[i], x[i+1] - x[i])
        
        # Width of the segment
        width = np.sqrt((x[i+1] - x[i])**2 + (z[i+1] - z[i])**2)
        
        # Center point of the segment
        center_x = (x[i] + x[i+1]) / 2
        center_z = (z[i] + z[i+1]) / 2
        
        segments.append({
            "center": [center_x, center_z],
            "width": width,
            "angle": angle
        })
    
    overlap_factor = 1.01  # 1% overlap to ensure no gaps
    
    for i, segment in enumerate(segments):
        center_x, center_z = segment["center"]
        width = segment["width"] * overlap_factor  # Add overlap
        angle = segment["angle"]
        
        RigidBodies.append({
            "geometryFile": "../models/UnitBox.obj",
            "translation": [center_x, center_z - Ly/2, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": angle,
            "scale": [width, Ly, Lz],
            "color": [0.1, 0.4, 0.6, 1.0],
            "isDynamic": False,
            "isWall": False,
            "mapInvert": mapInvert,
            "mapThickness": 0.0,
            "mapResolution": [max(10, int(width/r)), 10, 10],  # Resolution based on width
            "samplingMode": 1
        })
    
    # Left connection piece
    RigidBodies.append({
        "geometryFile": "../models/UnitBox.obj",
        "translation": [parabola_start - 0.01, z[0]/2 - Ly/2, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [0.02, z[0] + Ly, Lz],
        "color": [0.1, 0.4, 0.6, 1.0],
        "isDynamic": False,
        "isWall": False,
        "mapInvert": mapInvert,
        "mapThickness": 0.0,
        "mapResolution": [10, 20, 20],
        "samplingMode": 1
    })
    
    # Right connection piece
    RigidBodies.append({
        "geometryFile": "../models/UnitBox.obj",
        "translation": [parabola_end + 0.01, z[-1]/2 - Ly/2, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [0.02, z[-1] + Ly, Lz],
        "color": [0.1, 0.4, 0.6, 1.0],
        "isDynamic": False,
        "isWall": False,
        "mapInvert": mapInvert,
        "mapThickness": 0.0,
        "mapResolution": [10, 20, 20],
        "samplingMode": 1
    })
    
    #---------------------------#
    #    FLUID CONFIGURATION    #
    #---------------------------#
    
    fluid_x11, fluid_y11 = Lx_1/4, 0.05
    fluid_x21, fluid_y21 = Lx_1 - 2*r, 0.4
    
    box_min, box_max = calculate_scene_bounds(RigidBodies, margin=5*r)
    
    emit_H = 0.41                # Emitter height
    emit_L = (2*r)*2             # Emitter width
    
    emitter_config = {
        "physical_width": emit_L,             
        "physical_height": emit_H,          
        "width": int(emit_L / (2 * r)),      
        "height": int(emit_H / (2 * r)),   
        "translation": [0.5, emit_H/2 + 4*r, 0],           
        "velocity": U_0,                           
        "type": 0,                                   
        "end_time": 4                               
    }
    
    final_box_min, final_box_max = calculate_scene_bounds(RigidBodies, margin=10*r)
    
    # Add some padding to the bounds 
    padding = [(final_box_max[i] - final_box_min[i]) * 1 for i in range(3)]
    emitter_box_min = [final_box_min[i] - padding[i] for i in range(3)]
    emitter_box_max = [final_box_max[i] + padding[i] for i in range(3)]
    
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
                "maxEmitterParticles": 100000,
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
        
        "RigidBodies": RigidBodies,   
    }

    '''"FluidBlocks": [
            {
                "denseMode": 0,
                "start": [fluid_x11, fluid_y11, -0.5],
                "end": [fluid_x21, fluid_y21, 0.5],
                "scale": [1, 1, 1]
            }
        ],'''
    
    # 3.7 Write JSON file
    json_path = "SPlisHSPlasH/data/Scenes/free_surface_2D.json"
    output_path = "SPlisHSPlasH/bin/output/free_surface_2D"
    
    # Clean output directory
    if clean_output:
        clean_files(output_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Write JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"\nData written to '{json_path}'")
    

if __name__ == "__main__":
    main()