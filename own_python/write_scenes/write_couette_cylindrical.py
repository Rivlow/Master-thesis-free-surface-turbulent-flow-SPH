import json
import os
import numpy as np
import math
import os
import sys

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import * 

def main():
    #-----------------------------#
    #    SIMULATION PARAMETERS    #
    #-----------------------------#
    
    r = 0.02              # Particle radius
    t_end = 100.0           # Simulation end time
    timeStepSize = 0.001   # Time step size
    FPS = 25               # Frames per second for export
    clean_output = True    # Clean output directory
    
    rho_0 = 1000           # Reference density
    particle = 2*r

    angular_velo = 20
    
    sim2D = False           # 2D simulation
    attr = "velocity;acceleration;angular velocity;vorticity;density"  # Exported attributes
    
    simulationMethod = 4   # DFSPH
    xsph_fluid = 0.09           # XSPH for fluid
    xsph_boundary = 0.01        # XSPH for boundaries
    viscosity_boundary = 0.1    # Boundary viscosity

    viscosityMethod = 6    # Weiler et al. 2018
    nu = 1e-6              # Kinematic viscosity

    vorticityMethod = 1    # Micropolar model
    vorticity = 0.04        # Vorticity coefficient
    viscosityOmega = 0.1   # Angular viscosity
    inertiaInverse = 0.5   # Inverse inertia

    boundaryHandlingMethod = 2  # Volume maps
    
    #-------------------------------#
    #     COUETTE FLOW PARAMETERS   #
    #-------------------------------#
    
    # Cylindrical container parameters
    R_inner = 0.8          # Inner cylinder radius
    R_outer = 2.0          # Outer cylinder radius
    height = 0.7           # Height of cylinders
    
    # Rotation parameters
    omega_inner = 0.0     # Angular velocity of inner cylinder (rad/s)
    omega_outer = 0.0      # Angular velocity of outer cylinder (rad/s)
    
    # Calculate domain bounds
    domain_size = 2.5 * R_outer
    box_min = [-domain_size, -domain_size, -height/2]
    box_max = [domain_size, domain_size, height/2]
    
    #---------------------------#
    #     RIGID BODIES SETUP    #
    #---------------------------#
    
    RigidBodies = [
        # Outer cylinder (static)
        {
            "id": 0,
            "geometryFile": "../models/cylinder.obj",
            "translation": [0, 0, 0],
            "rotationAxis": [1, 0, 0],
            "rotationAngle": np.pi/2,
            "scale": [R_outer, height, R_outer],
            "color": [0.1, 0.4, 0.6, 0.2], 
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,  # True because fluid is inside
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 20],
            "samplingMode": 1
        },
        # Inner cylinder (rotating)
        {
            "id": 1,
            "geometryFile": "../models/cylinder.obj",
            "translation": [0, 0, 0],
            "rotationAxis": [1, 0, 0],
            "rotationAngle": np.pi/2,
            "scale": [R_inner, height, R_inner],
            "color": [0.6, 0.2, 0.2, 0.2], 
            "isDynamic": True,  # Set to True so it can be rotated
            "isWall": False,
            "mapInvert": False,  # False because fluid is outside
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 20],
            "samplingMode": 1
        }
    ]
    
    #---------------------------#
    #    FLUID CONFIGURATION    #
    #---------------------------#
    
    # Generate blocks along the mean radius
    R_mean = 0.5*(R_inner + R_outer) 
    thickness = (R_outer - R_inner)  # Slightly smaller to avoid boundary issues
    block_size = thickness  # Size of each block
    
    # Number of blocks to place around the circle
    num_blocks = 20  # Every 10 degrees
    
    FluidBlocks = []

    # Create fluid blocks positioned in a circle at the mean radius
    for i in range(num_blocks):
        angle = i * (2 * np.pi / num_blocks)
        # Position of the block center
        x = R_mean * np.cos(angle)
        y = R_mean * np.sin(angle)

       
        
        # Align the block with the tangent direction
        FluidBlocks.append({
            "id": "Fluid",
            "denseMode": 0,  # More dense sampling
            "start": [-height/3, -height/3, -height/2 + particle ],
            "end": [height/3, height/3, height/2 - particle ],
            "translation": [x, y, 0],
            "rotationAxis": [0, 0, 1],
        })
    
    #---------------------------#
    #    MOTOR JOINT SETUP     #
    #---------------------------#
    
    # Use a motor to rotate the inner cylinder
    TargetVelocityMotorHingeJoints = [
        {
            "bodyID1": 1,  # Inner cylinder
            "bodyID2": 0,  # Outer cylinder (as reference frame)
            "position": [0, 0, 0],  # At the center
            "axis": [0, 0, 1],      # Rotation around z-axis
            "repeatSequence": 1,
            "targetSequence": [0,angular_velo,1,angular_velo]
        }
    ]
    
    #-------------------------#
    #    SCENE CONFIGURATION  #
    #-------------------------#
    
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
            "gravitation": [0, 0, -9.81],  # No gravity for this case
            "cflMethod": 1,
            "cflFactor": 0.5,
            "cflMaxTimeStepSize": 0.005,
            "maxIterations": 100,
            "maxError": 0.01,
            "maxIterationsV": 100,
            "maxErrorV": 0.01,
            "stiffness": 50000,
            "exponent": 7,
            "velocityUpdateMethod": 0,
            "enableDivergenceSolver": True,
            "boundaryHandlingMethod": boundaryHandlingMethod
        },
    
        "Materials": [
            {
                "id": "Fluid",
                "xsphBoundary": xsph_boundary, 
                "xsph": xsph_fluid,   
                "viscosityBoundary": viscosity_boundary,
                "viscosityMethod": viscosityMethod,
                "viscosity": nu,
                "vorticityMethod": vorticityMethod,
                "vorticity": vorticity, 
                "viscosityOmega": viscosityOmega,
                "inertiaInverse": inertiaInverse,
                "colorMapType": 1
            }
        ],

        "FluidBlocks": FluidBlocks,
        "RigidBodies": RigidBodies,
        "TargetVelocityMotorHingeJoints": TargetVelocityMotorHingeJoints
    }
    
    # Write JSON file
    json_path = "SPlisHSPlasH/data/Scenes/couette_flow_3D.json"
    output_path = "SPlisHSPlasH/bin/output/couette_flow_3D"
    
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