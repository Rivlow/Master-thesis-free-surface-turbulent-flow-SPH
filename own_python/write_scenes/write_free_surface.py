import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import * 

kg = 1
m = 1
mm = 1e-3*m
s = 1


def main():

	# Write JSON file
	write = True
	json_path = "SPlisHSPlasH/data/Scenes/free_surface.json"
	summary_path = "my_output/local/free_surface/r_10mm/summary.txt"
	output_path = "SPlisHSPlasH/bin/output/free_surface"

	#-----------------------------#
	#	SIMULATION PARAMETERS	#
	#-----------------------------#
	
	# Simulation time and step
	t_end = 400*s
	timeStepSize = 0.001*s
	sim2D = True
	maxEmitterParticles = 10000000
	

	# Physical parameters
	r = 10 * (mm)               
	U_0 = 0.36 * (m/s)         
	g = 9.81 * (m/s**2)
	rho_0 = 1000 * (kg/m**3)

	particle = 2*r 
	
	# Export settings
	clean_output = True
	attr = "pressure acceleration;velocity;angular velocity;p / rho^2;density;time;dt;mass"  # Exported attributes
	FPS = 1
	
	#------Pressure solver------#
	simulationMethod = 4      # DFSPH
	maxIterations = 100      # Density solver
	maxError = 0.05
	maxIterationsV = 100     # Divergence solver
	maxErrorV = 0.05

	#------CFL conditiopn------#
	cflMethod = 2            # adapative dt + consider nb pressure solver iteration 
	cflFactor = 0.6
	cflMaxTimeStepSize = 0.01
	
	#------Viscosity------#
	nu = 0                    # Kinematic viscosity
	viscosityMethod = 0       # No viscosity
	viscosity_boundary = 0.0  # Boundary viscosity

	#------XSPH------#
	xsph_fluid = 0.04
	xsph_boundary = 0.0

	#------Vorticity------#
	vorticityMethod = 1       # Micropolar model
	vorticity = 0.02          # Vorticity coefficient
	viscosityOmega = 0.1     # Angular viscosity
	inertiaInverse = 1        # Inverse inertia

	#------Boundary interaction------#
	boundaryHandlingMethod = 2 # Volume maps
	elasticityMethod = 0       # No elasticity
	dragMethod = 2             # Gissler et al. 2017
	drag = 1.0                 # Drag coefficient


	#-------------------------------#
	#	 GEOMETRIC CONFIGURATION   #
	#-------------------------------#
	
	# Horizontal blocs
	Ly = 0.1
	Lz = 1.0
	Lx_1 = 8
	Lx_2 = 13

	# Parabola obstacle
	nb_elem = 10
	parabola_start = Lx_1
	parabola_end = parabola_start + 4

	# Emitter 
	Ly_emit = 0.5 * (m)               # Emitter height
	Lx_emit = particle          # Emitter width

	# Translations
	trans_emit = [-Lx_emit/2 + particle, Ly_emit/2 + particle, 0]
	trans_ground_1 = [Lx_1/2, -Ly/2, 0]
	trans_ground_2 = [parabola_end + Lx_2/2, -Ly/2, 0]

	# Evaluate correct flow rate
	y_pos = calculate_emitter_particle_positions(trans_emit, int(Lx_emit / (2 * r)), int(Ly_emit / (2 * r)), r)
	Q_init = U_0*(np.max(y_pos))
	
	# Animation field
	Ly_anim = 1
	Lx_anim = 25*(particle) 
	trans_anim = [trans_ground_2[0] + Lx_2/2 - Lx_anim/2, Ly_anim/2, 0]
	scale_anim = [Lx_anim, Ly_anim, Lz]

	# Domain bounds
	x_min = -Lx_emit/2 - 2*r
	y_min = -2*Ly_emit
	x_max = parabola_end + Lx_2 + Lx_anim 
	y_max = -y_min
	z_min = -y_max
	z_max = y_max
	
	#-------------------------------#
	#       DEFINE COMPONENTS       #
	#-------------------------------#
	
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
			"mapInvert": False,
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
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 10],
			"samplingMode": 1
		}
	]
	
	# Add parabolic segments
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
			"mapInvert": False,
			"mapThickness": 0.0,
			"mapResolution": [40, 40, 10],
			"samplingMode": 1
		})

	Emitters = [
		{
			"width": int(Lx_emit / (2 * r)),
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
			"scale": scale_anim,
			"rotationAxis": [0, 0, 1],
			"rotationAngle": 0.0,
			"shapeType": 0,
			"expression_x": f"{Q_init}/0.33",
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
			"emitterBoxMin": [x_min, y_min, z_min],
			"emitterBoxMax": [x_max, y_max, z_max]
		}
	]

	# Configuration parameters
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
		"dataExportFPS": FPS,
	}


	# Assemble final data structure
	data = {
		"Configuration": Configuration,
		"Materials": Materials,
		"RigidBodies": RigidBodies,
		"Emitters": Emitters,
		"AnimationFields": AnimationFields
	}

	data_save = {
		"Configuration": Configuration,
		"Materials": Materials,
	}
	
	if write:
		# Clean output directory
		if clean_output:
			clean_files(output_path)
			print('Output folder cleaned')
		
		write_summary(summary_path, data_save)

		# Ensure directory exists
		os.makedirs(os.path.dirname(json_path), exist_ok=True)
		
		# Write JSON file
		with open(json_path, 'w') as json_file:
			json.dump(data, json_file, indent=4)
		
		print(f"\nData written to '{json_path}'")
		
	

if __name__ == "__main__":
	main()