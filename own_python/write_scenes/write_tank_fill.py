import json
import os
import numpy as np

def main():
	# Write JSON file
	json_path = "SPlisHSPlasH/data/Scenes/hydrostatic_2d.json"
	output_path = "SPlisHSPlasH/bin/output/hydrostatic_2d"

	#-----------------------------#
	#    SIMULATION PARAMETERS    #
	#-----------------------------#

	# Simulation time and step
	t_end = 5
	timeStepSize = 0.001
	sim2D = True  # Simulation 2D

	# Physical parameters
	r = 2*0.005                 # Rayon des particules (plus grand pour avoir moins de particules)
	g = 9.81                  # Gravité
	rho_0 = 1000              # Densité de référence

	# Domaine dimensions
	Lx_dom = 2.0              # Longueur (axe X)
	Ly_dom = 2.0              # Hauteur (axe Y)
	Lz_dom = 0.1              # Épaisseur (un peu plus épaisse pour 2D)

	#----Pressure solver----#
	simulationMethod = 4       # DFSPH
	maxIterations = 100        # Density solver
	maxError = 0.01
	maxIterationsV = 100       # Divergence solver
	maxErrorV = 0.01

	#----CFL condition----#
	cflMethod = 2 
	cflFactor = 0.5
	cflMaxTimeStepSize = 0.005

	#----Viscosity----#
	nu = 1e-4                 # Viscosité cinématique
	viscosityMethod = 6        # Standard (plus simple)
	viscosity_boundary = 0.01   # Viscosité à la frontière

	#----XSPH----#
	xsph_fluid = 0.001          # Désactivé pour plus de stabilité
	xsph_boundary = 0.01

	# Export settings
	attr = "pressure;density"
	clean_output = True
	FPS = 30

	#-------------------------------#
	#       DEFINE COMPONENTS       #
	#-------------------------------#

	# 1. Rigid Body (Domain box only)
	RigidBodies = [
		# Domain
		{
			"id": 0,
			"geometryFile": "../models/UnitBox.obj",
			"translation": [Lx_dom/2, Ly_dom/2, 0],
			"scale": [Lx_dom, Ly_dom, Lz_dom],
			"rotationAxis": [1, 0, 0],
			"rotationAngle": 0,
			"collisionObjectType": 2,
			"collisionObjectScale": [Lx_dom, Ly_dom, Lz_dom],
			"color": [0.1, 0.4, 0.6, 0.5],
			"isDynamic": False,
			"isWall": True,
			"mapInvert": True,
			"invertSDF": True,
			"mapThickness": 0.0,
			"mapResolution": [30, 30, 20],
			"samplingMode": 0,
			"friction": 0
		}
	]

	# 2. Fluid Blocks - Calcul précis de la marge
	margin = 2.5 * r  # Marge pour éviter les particules trop près des bords
	print([margin, margin, -Lz_dom/4])
	print([Lx_dom - margin, Ly_dom - margin, Lz_dom/4])    

	FluidBlocks = [
		{
			"denseMode": 0,
			"start": [margin, margin, -Lz_dom/4],
			"end": [Lx_dom - margin, Ly_dom - margin, Lz_dom/4],
			"initialVelocity": [0.0, 0.0, 0.0]  # Particules au repos
		}
	]

	# 3. Materials properties
	Materials = [
		{
			"id": "Fluid",
			"density0": rho_0,
			
			# Viscosity parameters
			"viscosityMethod": viscosityMethod,
			"viscosity": nu,
			
			# XSPH parameters
			"xsph": xsph_fluid,
			"xsphBoundary": xsph_boundary,
			
			# Vorticity - désactivé
			"vorticityMethod": 0,
			
			# Drag - désactivé
			"dragMethod": 0,
			
			# Surface tension - désactivé
			"surfaceTensionMethod": 0,
			
			# Visualization
			"colorField": "pressure",
			"colorMapType": 1,
			"renderMinValue": 0,
			"renderMaxValue": 20000
		}
	]

	# 4. Configuration parameters
	Configuration = {
		# Basic simulation parameters
		"timeStepSize": timeStepSize,
		"stopAt": t_end,
		"sim2D": sim2D,
		"particleRadius": r,
		"density0": rho_0,
		
		# Physics models
		"simulationMethod": simulationMethod,
		"gravitation": [0, -g, 0],
		"boundaryHandlingMethod": 2,  # Volume maps
		
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
		"renderWalls": 3,  # Rendre la géométrie
		
		# Kernel function
		"kernel": 1,        # Wendland quintic C2 pour 2D
		"gradKernel": 1     # Wendland quintic C2 pour 2D
	}

	data = {
		"Configuration": Configuration,
		"Materials": Materials,
		"RigidBodies": RigidBodies,
		"FluidBlocks": FluidBlocks
	}

	# Clean output directory
	if clean_output:
		try:
			import shutil
			if os.path.exists(output_path):
				shutil.rmtree(output_path)
			os.makedirs(output_path, exist_ok=True)
			print('Output folder cleaned')
		except Exception as e:
			print(f'Error cleaning output folder: {e}')

	# Ensure directory exists
	os.makedirs(os.path.dirname(json_path), exist_ok=True)

	# Write JSON file
	with open(json_path, 'w') as json_file:
		json.dump(data, json_file, indent=4)

	print(f"\nData written to '{json_path}'")

if __name__ == "__main__":
	main()