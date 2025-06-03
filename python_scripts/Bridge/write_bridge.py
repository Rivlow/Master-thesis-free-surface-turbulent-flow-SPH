import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

sys.path.append((os.getcwd()))
from python_scripts.Tools_scenes import *

m = 1
mm = 1e-3
s = 1


def main():

	# Write JSON file
	json_path = "SPlisHSPlasH/data/Scenes/bridge.json"
	output_path = "SPlisHSPlasH/bin/output/bridge"

	#-----------------------------#
	#    SIMULATION PARAMETERS    #
	#-----------------------------#

	# Simulation time and step
	t_end = 25*s
	timeStepSize = 0.0005*s
	sim2D = False
	maxEmitterParticles = 10000000

	# Physical parameters
	r = 5 * (mm)                # Particle radius
	particle = 2*r            # Particle diameter
	U_0 = 0.15/2 *(m/s)            # Initial velocity
	Fr = isFluvial(False)
	g = 9.81                  # Gravity
	rho_0 = 1000              # Reference density
	wood_density = 800

	#------Pressure solver------#
	simulationMethod = 4       # DFSPH
	maxIterations = 2000 	   # Density solver
	maxError = 0.05
	maxIterationsV = 1000 	   # Divergence solver
	maxErrorV = 0.05

	#------CFL conditiopn------#
	cflMethod = 2 # adapative dt + consider nb pressure solver iteration 
	cflFactor = 0.6
	cflMaxTimeStepSize = 0.01
	
	#------Viscosity------#
	nu = 1e-6                     # Kinematic viscosity
	viscosityMethod = 6        # Weiler et al. 2018
	viscoMaxIter = 500
	viscoMaxError = 0.05
	viscosity_boundary = 1  # Boundary viscosity

	#------XSPH------#
	xsph_fluid = 0.08
	xsph_boundary = 0.01

	#------Vorticity------#
	vorticityMethod = 1        # Micropolar model
	vorticity = 0.04           # Vorticity coefficient
	viscosityOmega = 0.08      # Angular viscosity
	inertiaInverse = 1         # Inverse inertia

	#------Boundary interaction------#
	boundaryHandlingMethod = 2 # Volume maps
	elasticityMethod = 0       # No elasticity
	dragMethod = 2             # Gissler et al. 2017
	drag = 0.5                 # Drag coefficient

	#------PBD------#
	clothSimulationMethod = 0 # None
	velocityUpdateMethod = 1
	maxIter = 100
	maxIterVel = 100
	
	restitution = 0.0
	friction = 0.8
	DampingCoeff = 0.1

	contactTolerance = 1*particle
	contactStiffnessRigidBody = 100
	contactStiffnessParticleRigidBody = 0.1

	#-------Export settings------#
	attr = "pressure acceleration;velocity;angular velocity;p / rho^2;density;time;dt;mass"
	clean_output = True
	FPS = 5

	#-------------------------------#
	#     GEOMETRIC CONFIGURATION   #
	#-------------------------------#

	# Channel (domain) dimensions
	Ly_dom = 0.5*m               # Height
	Lx_dom = 3.5*m                # Length
	Lz_dom = 1.0*m              # Width

	water_height = 1*100*mm
	y_init_wood = 1.4*water_height
	x_entrance = 1.5*m

	#----Bridge dimensions----#
	# Roof   
	Lx_roof = 500*mm  
	Ly_roof = 64*mm    
	Lz_roof = 986*mm             

	# Foot of the bridge
	Lx_foot = 500*mm
	Ly_foot = 211*mm
	Lz_foot = 64*mm

	offset	 = (Lz_roof- Lz_foot)/2

	# Emitter configuration
	Lx_emit = particle
	Ly_emit = 2*water_height
	Lz_emit = Lz_dom - 2*particle

	Q_init = U_0*Ly_emit*Lz_emit
	U_out = 0.95*(Q_init/(Lz_dom*Ly_foot))

	# Animation field configuration
	Lx_anim = 10*particle
	Ly_anim = Ly_dom
	Lz_anim = Lz_dom	

	#----Translation vectors----#
	trans_roof = [x_entrance + Lx_roof/2, Ly_foot + Ly_roof/2, 0]
	trans_mid_foot = [x_entrance + Lx_foot/2, Ly_foot/2, 0]
	trans_left_foot = [x_entrance + Lx_foot/2, Ly_foot/2, -Lz_foot/2 -offset/2]
	trans_right_foot = [x_entrance + Lx_foot/2, Ly_foot/2, +Lz_foot/2 +offset/2]

	trans_emit = [4*particle + Lx_emit/2, Ly_emit/2, 0]
	trans_anim = [Lx_dom - Lx_anim/2, Ly_anim/2, 0]




	#-------------------------------#
	#       DEFINE COMPONENTS       #
	#-------------------------------#

	RigidBodies = []
	#•Domain
	RigidBodies.append({
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
	})

	
	RigidBodies = create_bridge(RigidBodies,
								Lx_dom, Ly_dom, Lz_dom,
								Lx_roof, Ly_roof, Lz_roof, trans_roof,
								Lx_foot, Ly_foot, Lz_foot, trans_left_foot, trans_mid_foot, trans_right_foot)			

	# Add cylinder obstacles (representing wood logs)
	wood_distribution = [
		#{"L": 0.100, "D": 0.05, "count": 62},  # 61.80%
		#{"L": 0.200, "D": 0.010, "count": 18},  # 18.26%
		#{"L": 0.300, "D": 0.015, "count": 10},  # 9.55%
		{"L": 0.400, "D": 0.020, "count": 5},   # 5.06%
		{"L": 0.500, "D": 0.025, "count": 3},   # 2.81%
		{"L": 0.600, "D": 0.030, "count": 2},   # 1.69%
		{"L": 0.400, "D": 0.035, "count": 3}    # 0.84%
	]

	xyz_fluid_inlet_start = [8*particle, 0, -Lz_dom/2 + 2*particle]
	xyz_fluid_inlet_end = [x_entrance, water_height, Lz_dom/2 - 2*particle]

	xyz_fluid_outlet_start = [x_entrance + Lx_roof, 0, -Lz_dom/2 + 2*particle]
	xyz_fluid_outlet_end = [Lx_dom, water_height, Lz_dom/2 - 2*particle]

	x_mid = 0.5*(xyz_fluid_inlet_end[0] + xyz_fluid_inlet_start[0] )
	z_mid = 0.5*(xyz_fluid_inlet_end[2] + xyz_fluid_inlet_start[2] )

	trans_obj_1 = [x_mid - 0.4, y_init_wood, z_mid - 0.2]
	trans_obj_2 = [x_mid + 0.4, y_init_wood, z_mid - 0.1]
	trans_obj_3 = [x_mid + 0.2, y_init_wood, z_mid + 0.1]
	trans_obj_4 = [x_mid - 0.2, y_init_wood, z_mid + 0.2]
	trans_obj_5 = [x_mid + 0.1, y_init_wood, z_mid + 0.2]
	trans_obj_6 = [x_mid - 0.1, y_init_wood, z_mid]
	trans_obj_7 = [x_mid, y_init_wood, z_mid - 0.2]

	xyz_min_bound = [0, 0, -Lz_dom/2]
	xyz_max_bound= [Lx_dom - 5*particle, Ly_dom, Lz_dom/2]


	D = 0.035
	L = 0.4
	dim_wood = [D, L, D] # D, L, D
	# First cylinder object
	RigidBodies.append(
	{"id": 3,
		"geometryFile": "../models/cylinder.obj",
		"translation": trans_obj_1,
		"scale": dim_wood,
		"rotationAxis": [1, 0, 0],
		"rotationAngle": np.pi/2,
		"collisionObjectType": 3,
		"collisionObjectScale": dim_wood,
		"color": [0.3, 0.5, 0.8, 1.0],
		"isDynamic": True,
		"density": 700,
		"velocity": [0, 0, 0],
		"restitution": 0.06,
		"friction": 0.2,
		"mapInvert": False,
		"mapThickness": 0.0,
		"mapResolution": [80, 80, 80]
	})

	
	# Second cylinder object
	RigidBodies.append({
			"id": 4,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_2,
			"scale": dim_wood,
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": dim_wood,
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": wood_density,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		})
	
	# Thrid cylinder object
	RigidBodies.append({
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_3,
			"scale": dim_wood,
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": dim_wood,
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": wood_density,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		})
	# Fourth cylinder object
	RigidBodies.append({
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_4,
			"scale": dim_wood,
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": dim_wood,
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": wood_density,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		})
	
	# Fifth cylinder object
	RigidBodies.append({
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_5,
			"scale": dim_wood,
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": dim_wood,
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": wood_density,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		})
	
	# Sixth cylinder object
	RigidBodies.append({
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_6,
			"scale": dim_wood,
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": dim_wood,
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": wood_density,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		})
	
	# Seventh cylinder object
	RigidBodies.append({
			"id": 5,
			"geometryFile": "../models/cylinder.obj",
			"translation": trans_obj_7,
			"scale": dim_wood,
			"rotationAxis": [1, 0, 0],
			"rotationAngle": np.pi/2,
			"collisionObjectType": 3,
			"collisionObjectScale": dim_wood,
			"color": [0.2, 0.6, 0.4, 1.0],
			"isDynamic": True,
			"density": wood_density,
			"velocity": [0, 0, 0],
			"restitution": restitution,
			"friction": friction,
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 40]
		})

	'''
	RigidBodies = create_wood_distribution(RigidBodies, wood_distribution, next_id,
										   Lx_emit, Ly_emit, Lz_emit, trans_emit,
										   wood_density, restitution, friction,
										   placement_area_x_min, placement_area_x_max,
										   placement_area_z_min, placement_area_z_max,
										   placement_area_y)
	'''

	FluidBlocks = [
		# Before bridge
		{
			"denseMode": 0,
			"start": xyz_fluid_inlet_start,
			"end": xyz_fluid_inlet_end,
			"initialVelocity": [U_0, 0.0, 0.0]
		},
			# After bridge
		{
			"denseMode": 0,
			"start": xyz_fluid_outlet_start,
			"end": xyz_fluid_outlet_end,
			"initialVelocity": [U_0, 0.0, 0.0]
		},# Inside bridge (inner left)
		{
			"denseMode": 0,
			"start": [x_entrance, 0, - offset/2 + particle],
			"end": [x_entrance+ Lx_roof, water_height, -Lz_foot/2 - particle],
			"initialVelocity": [U_0, 0.0, 0.0]
		},
		# Inside bridge (inner right)
		{
			"denseMode": 0,
			"start": [x_entrance-particle, 0, Lz_foot/2 + 2*particle],
			"end": [x_entrance + Lx_roof + particle, water_height, offset/2 - particle],
			"initialVelocity": [U_0, 0.0, 0.0]
		},
		# Inside bridge (outer left)
		{
			"denseMode": 0,
			"start": [x_entrance - particle, 0, -Lz_dom/2 + particle],
			"end": [x_entrance + Lx_roof + particle, water_height, -offset/2 - Lz_foot - particle],
			"initialVelocity": [U_0, 0.0, 0.0]
		},
		# Inside bridge (outer right)
		{
			"denseMode": 0,
			"start": [x_entrance - particle, 0, offset/2 + Lz_foot + particle],
			"end": [x_entrance + Lx_roof + particle, water_height, Lz_dom/2- particle],
			"initialVelocity": [U_0, 0.0, 0.0]
		}
	]


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

	AnimationFields = [
		{
			"particleField": "velocity",
			"translation": trans_anim,
			"scale": [Lx_anim, Ly_anim, Lz_anim],
			"rotationAxis": [0, 0, 1],
			"rotationAngle": 0.0,
			"shapeType": 0,
			"expression_x": f"{U_out}",
			"expression_y": "",
			"expression_z": ""
		}
	]

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
			"emitterBoxMin": xyz_min_bound,
			"emitterBoxMax": xyz_max_bound
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
		"viscoMaxIter": viscoMaxIter,
		"viscoMaxError": viscoMaxError,
		
		# Visualization and output
		"particleAttributes": attr,
		"numberOfStepsPerRenderUpdate": 4,
		"enableVTKExport": True,
		"enableRigidBodyVTKExport": True,
		"dataExportFPS": FPS
	}

	# 7. Simulation parameters
	Simulation = {

		"maxIter": maxIter,
		"maxIterVel": maxIterVel,
		"velocityUpdateMethod": velocityUpdateMethod,
		
		# Contact handling
		"contactTolerance": contactTolerance,
		"contactStiffnessRigidBody": contactStiffnessRigidBody, # body-body coupling
		"contactStiffnessParticleRigidBody": contactStiffnessParticleRigidBody, # fluid-body coupling 

		"damping": DampingCoeff,
		
		# Solid parameters
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
		"AnimationFields": AnimationFields,
		"FluidBlocks":FluidBlocks
	}

	'''# Inside bridge (inner left)
		{
			"denseMode": 0,
			"start": [x_entrance, 0, -Lz_foot - offset/2 + 2*particle],
			"end": [x_entrance+ Lx_roof, water_height, -Lz_foot/2 - particle],
			"initialVelocity": [0, 0.0, 0.0]
		},
		# Inside bridge (inner right)
		{
			"denseMode": 0,
			"start": [x_entrance, 0, Lz_foot/2 + particle],
			"end": [x_entrance+ Lx_roof, water_height, Lz_foot + offset/2 - 2*particle],
			"initialVelocity": [0, 0.0, 0.0]
		},
		# Inside bridge (outer left)
		{
			"denseMode": 0,
			"start": [x_entrance, 0, -Lz_dom/2 + particle],
			"end": [x_entrance+ Lx_roof, water_height, -offset/2 - Lz_foot - particle],
			"initialVelocity": [0, 0.0, 0.0]
		},
		# Inside bridge (outer right)
		{
			"denseMode": 0,
			"start": [x_entrance, 0, offset/2 + Lz_foot + particle],
			"end": [x_entrance+ Lx_roof, water_height, Lz_dom/2- particle],
			"initialVelocity": [0, 0.0, 0.0]
		},'''
	
	# Clean output directory
	if clean_output:
		clean_files(output_path)
		print('Output folder cleaned')

	# Ensure directory exists
	os.makedirs(os.path.dirname(json_path), exist_ok=True)
	write_summary_file(data, output_path)

	# Write JSON file
	with open(json_path, 'w') as json_file:
		json.dump(data, json_file, indent=4)

	print(f"\nData written to '{json_path}'")


if __name__ == "__main__":
	main()