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
    
    r = 0.005*np.sqrt(2)             # Particle radius
    U_0 = 0.36                     # Initial velocity
    t_end = 300                 # Simulation end time
    timeStepSize = 0.001        # Time step size
    FPS = 40                    # Frames per second for export
    clean_output = True         # Clean output directory
    g = 9.81
    
    rho_0 = 1000                # Reference density
    
    sim2D = True                # 2D simulation
    mapInvert = False           # Domain is outside the unit box
    attr = "pressure acceleration;velocity;angular velocity;p / rho^2;density"  # Exported attributes
    
    xsph_fluid = 0.15           # XSPH for fluid
    xsph_boundary = 0.5         # XSPH for boundaries
    viscosity_boundary = 0.2    # Boundary viscosity
    
    simulationMethod = 4        # DFSPH
    viscosityMethod = 6         # Weiler et al. 2018
    nu = 1e-6                   # Kinematic viscosity
    vorticityMethod = 1        # Micropolar model
    vorticity = 0.15            # Vorticity coefficient
    viscosityOmega = 0.1        # Angular viscosity
    inertiaInverse = 0.5        # Inverse inertia
    boundaryHandlingMethod = 2  # Volume maps
    elasticityMethod = 0        # No elasticity
    dragMethod = 2              # Gissler et al. 2017
    drag = 1.0                  # Drag coefficient
    
    #-------------------------------#
    #     GEOMETRIC CONFIGURATION   #
    #-------------------------------#
    
    # Horizontal blocs
    Ly = 0.1                   
    Lz = 1.0                   
    Lx_1 = 1.5                
    Lx_2 = 3               

    # Parabola obstacle
    nb_elem = 20               
    parabola_start = Lx_1       
    parabola_end = Lx_1 + 4      

    # Second obstacle 
    Lx_obstacle = 0.02
    Ly_obstacle = 0.12
    Lz_obstacle = 0.8
    dist_x_obstacle = 1.5
    

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
    z = parabole(np.linspace(8, 12, nb_elem+1)) 
    
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
    dx = 1
    RigidBodies.append({
        "geometryFile": "../models/UnitBox.obj",
        "translation": [parabola_start - dx, z[0]/2 - Ly/2, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [2*dx, z[0] + Ly, Lz],
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

    trans_obstacle = [parabola_end + Lx_obstacle/2 + dist_x_obstacle, z[-1]/2 + Ly_obstacle/2 , 0]
    
    # Add obstacle at the end
    RigidBodies.append({
        "geometryFile": "../models/UnitBox.obj",
        "translation": trans_obstacle,
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [Lx_obstacle, Ly_obstacle, Lz_obstacle],
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
    
    emit_H = 0.5                # Emitter height
    emit_L = (2*r)*2             # Emitter width
    
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
    
    final_box_min, final_box_max = calculate_scene_bounds(RigidBodies, margin=10*r)
    
    # Add some padding to the bounds 
    padding = [(final_box_max[i] - final_box_min[i]) * 1 for i in range(3)]
    emitter_box_min = [final_box_min[i] - padding[i] for i in range(3)]
    emitter_box_max = [final_box_max[i] + padding[i] for i in range(3)]

    box_min = np.array(emitter_box_min)
    box_max = np.array(emitter_box_max)

    center_box = list((box_max + box_min)/2)
    size_box = list(np.abs(box_max - box_min))

    #trans_anim = [parabola_end + 0.01 + Lx_obstacle/2 + 3*(2*r), z[-1]/2 + 4*Ly/2, 0]
    trans_anim = [parabola_end + 0.01 + Lx_obstacle/2 + 3*(2*r), z[-1]/2 + 4*Ly/2, 0]
    
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


    '''        "AnimationFields": [

            {
                "particleField": "p / rho^2",
                "translation": center_box,
                "rotationAxis": [0, 0, 1],
                "rotationAngle": 0.0,
                "scale": size_box,
                "shapeType": 0,
                "expression_x": "",
                "expression_y": f"{g}*y/1000",
                "expression_z": ""
            }

            
]'''

    ''' {
                "particleField": "velocity",
                "translation": center_box,
                "rotationAxis": [0, 0, 1],
                "rotationAngle": 0.0,
                "scale": size_box,
                "shapeType": 0,
                "expression_x": f"{U_0}",
                "expression_y": "",
                "expression_z": ""
            },'''
    
    # Write JSON file
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