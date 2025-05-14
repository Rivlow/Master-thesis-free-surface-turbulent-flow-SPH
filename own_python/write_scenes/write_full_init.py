import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *

m = 1
mm = 1e-3*m
s = 1


def main():

	# Write JSON file
	json_path = "SPlisHSPlasH/data/Scenes/full_init.json"
	output_path = "SPlisHSPlasH/bin/output/full_init"

	#-----------------------------#
	#    SIMULATION PARAMETERS    #
	#-----------------------------#

	# Simulation time and step
	t_end = 50
	timeStepSize = 0.0001
	sim2D = False
	maxEmitterParticles = 10000000

	# Physical parameters
	r = 2*0.01*m                # Particle radius
	particle = 2*r            # Particle diameter
	U_0 = 0.5 *(m/s)            # Initial velocity
	Fr = isFluvial(False)
	g = 9.81                  # Gravity
	rho_0 = 1000              # Reference density

	#----Pressure solver----#
	simulationMethod = 4       # DFSPH
	maxIterations = 1000 	   # Density solver
	maxError = 0.05
	maxIterationsV = 1000 	   # Divergence solver
	maxErrorV = 0.05

	#----CFL conditiopn----#
	cflMethod = 2 # adapative dt + consider nb pressure solver iteration 
	cflFactor = 0.6
	cflMaxTimeStepSize = 0.0001
		
	
	#----Viscosity----#
	nu = 1e-6                     # Kinematic viscosity
	viscosityMethod = 6        # Weiler et al. 2018
	viscosity_boundary = 1  # Boundary viscosity

	#----XSPH----#
	xsph_fluid = 0.05
	xsph_boundary = 0.01

	#----Vorticity----
	vorticityMethod = 1        # Micropolar model
	vorticity = 0.05           # Vorticity coefficient
	viscosityOmega = 0.03      # Angular viscosity
	inertiaInverse = 2         # Inverse inertia

	#----Boundary interaction----#
	boundaryHandlingMethod = 2 # Volume maps
	elasticityMethod = 0       # No elasticity
	dragMethod = 2             # Gissler et al. 2017
	drag = 0.1               # Drag coefficient

	#----PBD---#
	clothSimulationMethod = 0 # None
	restitution = 0.0
	friction = 0.6
	contactTolerance = 4*particle
	contactStiffnessRigidBody = 50
	contactStiffnessParticleRigidBody = 0.5
	

	# Export settings
	attr = "pressure acceleration;velocity;angular velocity;p / rho^2;density;time;dt;mass"
	clean_output = True
	FPS = 20

	#-------------------------------#
	#     GEOMETRIC CONFIGURATION   #
	#-------------------------------#

	# Domain dimensions
	Ly_dom = 0.5               # Height
	Lx_dom = 10                 # Length
	Lz_dom = 2.0               # Width

	# Obstruction dimensions
	Lz_obs = 0.25*Lz_dom        # Width of obstruction
	Ly_obs = 3*Ly_dom            # Height of obstruction
	Lx_obs = Ly_obs            # Length of obstruction

	# Cylinder objects dimensions
	D = 0.035/2
	L = 0.4
	Lx_obj = 0.05
	Ly_obj = 2.3*0.6
	Lz_obj = 0.05


	# Emitter configuration
	Lx_emit = particle
	Ly_emit = 1.5*Ly_dom 
	Lz_emit = Lz_dom - 2*particle
	Q_init = U_0*Ly_emit*Lz_emit

	# Animation field configuration
	Lx_anim = 4*particle
	Ly_anim = Ly_dom
	Lz_anim = Lz_dom	

	#----Fluid blocks positions----#
	# (Before obstacles)
	water_height = Ly_dom
	fluid_1_min = [Lx_anim, 0, -Lz_dom/2 + particle]
	fluid_1_max = [Lx_dom/2 - Lx_obs/2 - particle, water_height, Lz_dom/2 - particle]

	# (Between obstacles)
	fluid_2_min = [Lx_dom/2 - Lx_obs/2 - particle, 0, -Lz_dom/2 + Lz_obs + particle]
	fluid_2_max = [Lx_dom/2 + Lx_obs/2, water_height, Lz_dom/2 - Lz_obs - particle]

	# (After obstacles)
	fluid_3_min = [Lx_dom/2 + Lx_obs/2 + particle, 0, -Lz_dom/2 + particle]
	fluid_3_max = [0.7*Lx_dom - particle, water_height, Lz_dom/2 - particle]

	#----Translation vectors----#
	trans_obs_left = [Lx_dom/2, 3*Ly_dom/2, -Lz_dom/2 + Lz_obs/2]
	trans_obs_right = [Lx_dom/2, 3*Ly_dom/2, Lz_dom/2 - Lz_obs/2]
	trans_emit = [Lx_emit/2, Ly_emit/2, 0]
	trans_anim = [Lx_dom - Lx_anim/2 - particle, Ly_anim/2, 0]

	# Cylinder positions
	center_fluid_1_x = (fluid_1_min[0] + fluid_1_max[0]) / 2
	center_fluid_1_z = (fluid_1_min[2] + fluid_1_max[2]) / 2
	height_above_fluid = 1.5 * Ly_dom  # Just above domain
	trans_obj_1 = [center_fluid_1_x - 0.4, height_above_fluid, center_fluid_1_z - 0.2]
	trans_obj_2 = [center_fluid_1_x + 0.4, height_above_fluid, center_fluid_1_z - 0.1]
	trans_obj_3 = [center_fluid_1_x + 0.2, height_above_fluid, center_fluid_1_z + 0.1]
	trans_obj_4 = [center_fluid_1_x - 0.2, height_above_fluid, center_fluid_1_z + 0.2]

	#-------------------------------#
	#       DEFINE COMPONENTS       #
	#-------------------------------#

	# 1. Rigid Bodies (Domain and obstacles)
	RigidBodies = [
		# Domain
		{
			"id": 0,
			"geometryFile": "../models/UnitBox.obj",
			"translation": [Lx_dom/2, 4*Ly_dom/2, 0],
			"scale": [Lx_dom, 4*Ly_dom, Lz_dom],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": 0,
			"collisionObjectType": 2,
			"collisionObjectScale": [Lx_dom, 4*Ly_dom, Lz_dom],
			"color": [0.1, 0.4, 0.6, 1.0],
			"isDynamic": False,
			"isWall": True,
			"mapInvert": True,
			"invertSDF": True,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40],
			"samplingMode": 1,
			"friction": 0
		},
		# Left obstacle
		{
			"id": 1,
			"geometryFile": "../models/UnitBox.obj",
			"translation": trans_obs_left,
			"scale": [Lx_obs, Ly_obs, Lz_obs],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": 0,
			"collisionObjectType": 2,
			"collisionObjectScale": [Lx_obs, Ly_obs, Lz_obs],
			"color": [0.1, 0.4, 0.6, 1.0],
			"isDynamic": False,
			"isWall": False,
			"mapInvert": False,
			"invertSDF": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40],
			"samplingMode": 1,
			"friction": 0
		},
		# Right obstacle
		{
			"id": 2,
			"geometryFile": "../models/UnitBox.obj",
			"translation": trans_obs_right,
			"scale": [Lx_obs, Ly_obs, Lz_obs],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": 0,
			"collisionObjectType": 2,
			"collisionObjectScale": [Lx_obs, Ly_obs, Lz_obs],
			"color": [0.1, 0.4, 0.6, 1.0],
			"isDynamic": False,
			"isWall": False,
			"mapInvert": False,
			"invertSDF": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40],
			"samplingMode": 1,
			"friction": 0
		},
		# First cylinder object
		{
			"id": 3,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_1,
			"scale": [Lx_obj, Ly_obj, Lz_obj],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": [Lx_obj, Ly_obj, Lz_obj],
			"color": [0.3, 0.5, 0.8, 1.0],
			"isDynamic": True,
			"density": 400,
			"velocity": [0, 0, 0],
			"restitution": 0.5,
			"friction": 0.7,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		},
		# Second cylinder object
		{
			"id": 4,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_2,
			"scale": [Lx_obj, Ly_obj, Lz_obj],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": [Lx_obj, Ly_obj, Lz_obj],
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": 400,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		},
		# Thrid cylinder object
		{
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_3,
			"scale": [Lx_obj, Ly_obj, Lz_obj],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": [Lx_obj, Ly_obj, Lz_obj],
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": 400,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		},
		# Fourth cylinder object
		{
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_4,
			"scale": [Lx_obj, Ly_obj, Lz_obj],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": [Lx_obj, Ly_obj, Lz_obj],
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": 400,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		}
	]

	# 2. Fluid emitters
	Emitters = [
		{
			"width": int(Lz_emit / (2 * r)),
			"height": int(Ly_emit / (2 * r)),
			"translation": trans_emit,
			"velocity": U_0,
			"type": 0,
			"rotationAxis": [0, 0, 1],
			"rotationAngle": 0.0,
			"emitEndTime": t_end
		}
	]

	# 3. Animation Fields
	AnimationFields = [
		{
			"particleField": "velocity",
			"translation": trans_anim,
			"scale": [Lx_anim, Ly_anim, Lz_anim],
			"rotationAxis": [0, 0, 1],
			"rotationAngle": 0.0,
			"shapeType": 0,
			"expression_x": f"{Fr*np.sqrt(g)}*sqrt(y)",
			"expression_y": "",
			"expression_z": ""
		}
	]

	# 4. Fluid Blocks
	FluidBlocks = [
		{
			"denseMode": 0,
			"start": fluid_1_min,
			"end": fluid_1_max,
			"initialVelocity": [U_0, 0.0, 0.0]
		},
		{
			"denseMode": 0,
			"start": fluid_2_min,
			"end": fluid_2_max,
			"initialVelocity": [U_0, 0.0, 0.0]
		},
		{
			"denseMode": 0,
			"start": fluid_3_min,
			"end": fluid_3_max,
			"initialVelocity": [U_0, 0.0, 0.0]
		}
	]

	# 5. Materials properties
	Materials = [
		{
			"id": "Fluid",
			# Viscosity parameters
			"viscosityMethod": viscosityMethod,
			"viscosity": nu,
			"viscosityBoundary": viscosity_boundary,
			
			# XSPH parameters
			"xsph": xsph_fluid,
			"xsphBoundary": xsph_boundary,
			
			# Vorticity parameters
			"vorticityMethod": vorticityMethod,
			"vorticity": vorticity,
			"viscosityOmega": viscosityOmega,
			"inertiaInverse": inertiaInverse,
			
			# Drag parameters
			"dragMethod": dragMethod,
			"drag": drag,
			
			# Visualization
			"colorMapType": 1,
			
			# Emitter parameters
			"maxEmitterParticles": maxEmitterParticles,
			"emitterReuseParticles": True,
			"emitterBoxMin": [0, 0, -Lz_dom/2],
			"emitterBoxMax": [Lx_dom-2*particle, 4*Ly_dom, Lz_dom/2]
		}
	]

	# 6. Configuration parameters
	Configuration = {
		# Basic simulation parameters
		"timeStepSize": timeStepSize,
		"stopAt": t_end,
		"sim2D": sim2D,
		"particleRadius": r,
		"density0": rho_0,
		
		# Physics models
		"simulationMethod": simulationMethod,
		"gravitation": [0, -9.81, 0],
		"boundaryHandlingMethod": boundaryHandlingMethod,
		
		# Numerical parameters
		"cflMethod": cflMethod,
		"cflFactor": cflFactor,
		"cflMaxTimeStepSize": cflMaxTimeStepSize,
		"maxIterations": maxIterations,
		"maxError": maxError,
		"maxIterationsV": maxIterationsV,
		"maxErrorV": maxErrorV,
		"enableDivergenceSolver": True,
		
		# Visualization and output
		"particleAttributes": attr,
		"numberOfStepsPerRenderUpdate": 4,
		"enableVTKExport": True,
		"enableRigidBodyVTKExport": True,
		"dataExportFPS": FPS
	}

	# 7. Simulation parameters
	Simulation = {

		"maxIter": 50,
		"maxIterVel": 50,
		"velocityUpdateMethod": 1,
		
		# Contact handling
		"contactTolerance": contactTolerance,
		"contactStiffnessRigidBody": contactStiffnessRigidBody, # body-body coupling
		"contactStiffnessParticleRigidBody": contactStiffnessParticleRigidBody, # fluid-body coupling 
		
		# Solid parameters
		"solid_stiffness": 60.0, # rigid bodies
		"solid_poissonRatio": 0.2,
		"solid_normalizeStretch": 0,
		"solid_normalizeShear": 0
	}

	data = {
		"Configuration": Configuration,
		"Simulation": Simulation,
		"Materials": Materials,
		"RigidBodies": RigidBodies,
		"Emitters": Emitters,
		"FluidBlocks": FluidBlocks,
		"AnimationFields": AnimationFields
	}

	# Clean output directory
	if clean_output:
		clean_files(output_path)
		print('Output folder cleaned')

	# Ensure directory exists
	os.makedirs(os.path.dirname(json_path), exist_ok=True)

	# Write JSON file
	write_summary_file(data, output_path)
	with open(json_path, 'w') as json_file:
		json.dump(data, json_file, indent=4)

	print(f"\nData written to '{json_path}'")


if __name__ == "__main__":
	main()