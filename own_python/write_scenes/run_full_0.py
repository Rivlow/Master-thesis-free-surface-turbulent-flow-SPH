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
    
    r = 0.01         # Particle radius
    particle = 2*r
    U_0 = 0.36                   # Initial velocity
    t_end = 200                 # Simulation end time
    timeStepSize = 0.001        # Time step size
    FPS = 10                    # Frames per second for export
    clean_output = True         # Clean output directory
    g = 9.81
    
    rho_0 = 1000                # Reference density
    
    sim2D = False                # 2D simulation
    mapInvert = False           # Domain is outside the unit box
    attr = "pressure acceleration;velocity;angular velocity;p / rho^2;density"  # Exported attributes
    
    xsph_fluid = 0.07           # XSPH for fluid
    xsph_boundary = 0.0         # XSPH for boundaries
    viscosity_boundary = 0.00    # Boundary viscosity
    
    simulationMethod = 4        # DFSPH
    viscosityMethod = 0         # Weiler et al. 2018
    nu = 0                  # Kinematic viscosity
    vorticityMethod = 1        # Micropolar model
    vorticity = 0.05            # Vorticity coefficient
    viscosityOmega = 0.03        # Angular viscosity
    inertiaInverse = 2        # Inverse inertia
    boundaryHandlingMethod = 2  # Volume maps
    elasticityMethod = 0        # No elasticity
    dragMethod = 2              # Gissler et al. 2017
    drag = 1.0                  # Drag coefficient

    maxEmitterParticles = 10000000
    
    #-------------------------------#
    #     GEOMETRIC CONFIGURATION   #
    #-------------------------------#
    
    # External domain
    Ly_dom = 1                                    
    Lx_dom = 2   
    Lz_dom = 1.0 
    
    # Left-right obstruction
    Lz_obs = 0.2*Lz_dom  
    Ly_obs = Ly_dom
    Lx_obs = Ly_obs

    # Emitter     
    Lx_emit = particle         
    Ly_emit = Ly_dom
    Lz_emit = Lz_dom

    Q_init = (U_0*Lz_emit*Ly_emit)
    
    # Animation field
    Lx_anim = 4*particle
    Ly_anim = Ly_dom
    Lz_anim = Lz_dom
    
    
    #=============#
    # Translation #
    #=============#
    
    trans_dom = [Lx_dom/2, Ly_dom/2, 0]  
    trans_emit = [Lx_emit/2, Ly_emit/2, 0]
    
                                            
    trans_obs_left = [Lx_dom/2, Ly_dom/2, -Lz_dom/2]
    trans_obs_right = [Lx_dom/2, Ly_dom/2, Lz_dom/2]



    trans_anim = [trans_ground_2[0] + Lx_2/2 - anim_x/2 , anim_y/2, 0]
    scale_anim = [anim_x, anim_y, Lz]

    

    
    
    RigidBodies = [
        # First rectangle (left flat section)
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": trans_ground_1,
            "rotationAxis": [1, 0, 0],
            "rotationAngle": 0,
            "scale": [Lx_1, Ly, Lz],
            "color": [0.1, 0.4, 0.6, 1.0], 
            "isDynamic": False,
            "isWall": False,
            "mapInvert": mapInvert,
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 10],
            "samplingMode": 1
        },
        # Second rectangle (right flat section)
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": trans_ground_2,
            "rotationAxis": [1, 0, 0],
            "rotationAngle": 0,
            "scale": [Lx_2, Ly, Lz],
            "color": [0.1, 0.4, 0.6, 1.0], 
            "isDynamic": False,
            "isWall": False,
            "mapInvert": mapInvert,
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 10],
            "samplingMode": 1
        }
    ]
    
    x = np.linspace(parabola_start, parabola_end, nb_elem+1)
    z = parabole(np.linspace(8, 12, nb_elem+1)) 

    x_min = 0
    y_min = -2*emit_H
    x_max = parabola_end + Lx_2 + anim_x 
    y_max = -y_min
    z_min = -y_max
    z_max = y_max
    
    block_test_i = [x_min, y_min, z_min]
    block_test_f = [x_max,y_max, z_max]
    
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
    
    overlap_factor = 1.0  # 1% overlap to ensure no gaps
    
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
            "mapResolution": [40, 40, 10],  
            "samplingMode": 1
        })
    
    emitter_config = {
        "physical_width": emit_L,             
        "physical_height": emit_H,          
        "width": int(emit_L / (2 * r)),      
        "height": int(emit_H / (2 * r)),   
        "translation": trans_emit,           
        "velocity": U_0,                           
        "type": 0,                                   
        "end_time": 4                               
    }
    
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
            "maxIterationsV": 1000,
            "maxErrorV": 0.005,
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
                "maxEmitterParticles": maxEmitterParticles,
                "emitterReuseParticles": True,
                "emitterBoxMin": [x_min, y_min, z_min],
                "emitterBoxMax": [x_max,y_max, z_max]
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

         "AnimationFields": [

            {
                "particleField": "velocity",
                "translation": trans_anim,
                "rotationAxis": [0, 0, 1],
                "rotationAngle": 0.0,
                "scale": scale_anim,
                "shapeType": 0,
                "expression_x": f"{U_0}*y/0.33",
                "expression_y": "",
                "expression_z": ""
            }
        ]
    }
    
    # Write JSON file
    json_path = "SPlisHSPlasH/data/Scenes/free_surface_pressure_2D.json"
    output_path = "SPlisHSPlasH/bin/output/free_surface_pressure_2D"
    
    # Clean output directory
    if clean_output:
        clean_files(output_path)
        print('Output folder cleaned')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Write JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"\nData written to '{json_path}'")
    

if __name__ == "__main__":
    main()