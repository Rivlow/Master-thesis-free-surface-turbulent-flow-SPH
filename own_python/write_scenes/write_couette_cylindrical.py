import json
import os
import numpy as np
import math

def main():
    #-----------------------------#
    #    SIMULATION PARAMETERS    #
    #-----------------------------#
    
    r = 0.005               # Particle radius
    t_end = 10.0            # Simulation end time
    timeStepSize = 0.001    # Time step size
    FPS = 60                # Frames per second for export
    clean_output = True     # Clean output directory
    
    rho_0 = 1000            # Reference density
    
    sim2D = True            # 2D simulation
    mapInvert = False       # Domain is outside the unit box
    attr = "velocity;acceleration;angular velocity;vorticity;density"  # Exported attributes
    
    simulationMethod = 4    # DFSPH
    viscosityMethod = 6     # Weiler et al. 2018
    nu = 0.01               # Kinematic viscosity
    vorticityMethod = 1     # Micropolar model
    vorticity = 0.2         # Vorticity coefficient
    viscosityOmega = 0.1    # Angular viscosity
    inertiaInverse = 0.5    # Inverse inertia
    boundaryHandlingMethod = 2  # Volume maps
    
    #-------------------------------#
    #     COUETTE FLOW PARAMETERS   #
    #-------------------------------#
    
    # Cylindrical container parameters
    R_inner = 0.05          # Inner cylinder radius
    R_outer = 0.1           # Outer cylinder radius
    height = 0.2            # Height of cylinders
    
    # Rotation parameters
    omega_inner = 10.0      # Angular velocity of inner cylinder (rad/s)
    omega_outer = 0.0       # Angular velocity of outer cylinder (rad/s)
    
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
            "geometryFile": "../models/UnitCylinder.obj",
            "translation": [0, 0, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0,
            "scale": [R_outer, R_outer, height/2],
            "color": [0.1, 0.4, 0.6, 1.0], 
            "isDynamic": False,
            "isWall": False,
            "mapInvert": True,  # True because fluid is inside
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 20],
            "samplingMode": 1
        },
        # Inner cylinder (rotating)
        {
            "id": 1,
            "geometryFile": "../models/UnitCylinder.obj",
            "translation": [0, 0, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0,
            "scale": [R_inner, R_inner, height/2],
            "color": [0.6, 0.2, 0.2, 1.0], 
            "isDynamic": False,  # Not simulated by PBD
            "isWall": False,
            "mapInvert": False,  # False because fluid is outside
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 20],
            "samplingMode": 1
        },
        # Bottom cap
        {
            "id": 2,
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0, 0, -height/2 - 0.01],
            "rotationAxis": [1, 0, 0],
            "rotationAngle": 0,
            "scale": [R_outer+0.01, R_outer+0.01, 0.01],
            "color": [0.1, 0.4, 0.6, 1.0], 
            "isDynamic": False,
            "isWall": False,
            "mapInvert": False,
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 10],
            "samplingMode": 1
        },
        # Top cap
        {
            "id": 3,
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0, 0, height/2 + 0.01],
            "rotationAxis": [1, 0, 0],
            "rotationAngle": 0,
            "scale": [R_outer+0.01, R_outer+0.01, 0.01],
            "color": [0.1, 0.4, 0.6, 1.0], 
            "isDynamic": False,
            "isWall": False,
            "mapInvert": False,
            "mapThickness": 0.0,
            "mapResolution": [40, 40, 10],
            "samplingMode": 1
        }
    ]
    
    #---------------------------#
    #    FLUID CONFIGURATION    #
    #---------------------------#
    
    # Generate blocks along the mean radius
    R_mean = (R_inner + R_outer) / 2
    thickness = (R_outer - R_inner) * 0.8  # Slightly smaller to avoid boundary issues
    block_size = thickness * 0.5  # Size of each block
    
    # Number of blocks to place around the circle
    num_blocks = 36  # Every 10 degrees
    
    FluidBlocks = []
    
    # Create fluid blocks positioned in a circle at the mean radius
    for i in range(num_blocks):
        angle = i * (2 * np.pi / num_blocks)
        # Position of the block center
        x = R_mean * np.cos(angle)
        y = R_mean * np.sin(angle)
        
        # Align the block with the tangent direction
        FluidBlocks.append({
            "denseMode": 1,  # More dense sampling
            "start": [-block_size/2, -thickness/2, -height/2 + 3*r],
            "end": [block_size/2, thickness/2, height/2 - 3*r],
            "translation": [x, y, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": angle
        })

    
    #---------------------------#
    #   ANIMATION FOR ROTATION  #
    #---------------------------#
    
    # For rotation of the inner cylinder
    AnimationFields = [
        {
            "particleField": "velocity",
            "translation": [0, 0, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0.0,
            "scale": [R_inner + r, R_inner + r, height],
            "shapeType": 1,  # Sphere shape, we'll use this as a cylinder
            "expression_x": f"-y*{omega_inner}",  # Tangential velocity in x
            "expression_y": f"x*{omega_inner}",   # Tangential velocity in y
            "expression_z": ""
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
            "gravitation": [0, 0, 0],  # No gravity for this case
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
                "viscosityMethod": viscosityMethod,
                "viscosity": nu,
                "vorticityMethod": vorticityMethod,
                "vorticity": vorticity, 
                "viscosityOmega": viscosityOmega,
                "inertiaInverse": inertiaInverse,
                "colorMapType": 1,
                "renderMinValue": 0,
                "renderMaxValue": omega_inner * R_inner * 1.2,  # Max expected velocity
            }
        ],
        
        "RigidBodies": RigidBodies,
        "FluidBlocks": FluidBlocks,
        "AnimationFields": AnimationFields
    }
    
    # Write JSON file
    json_path = "SPlisHSPlasH/data/Scenes/couette_flow_2D.json"
    output_path = "SPlisHSPlasH/bin/output/couette_flow_2D"
    
    # Clean output directory
    if clean_output and os.path.exists(output_path):
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Write JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"\nData written to '{json_path}'")
    print(f"Expected analytical velocity profile:")
    print(f"v_θ(r) = {omega_inner}*{R_inner}²/(r * ({R_outer}² - {R_inner}²)) * ({R_outer}² - r²)")
    print(f"Peak velocity at r = {R_inner}: {omega_inner * R_inner} m/s")
    
    # Calculate and print expected values for a few sample points
    sample_radii = np.linspace(R_inner, R_outer, 5)
    print("\nExpected velocity at various radii:")
    for r_i in sample_radii:
        v_theta = (omega_inner * R_inner**2) / (R_outer**2 - R_inner**2) * (R_outer**2/r_i - r_i)
        print(f"At r = {r_i:.3f} m: v_θ = {v_theta:.3f} m/s")

if __name__ == "__main__":
    main()