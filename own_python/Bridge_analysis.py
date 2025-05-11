import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6, 16, 18
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Configure project path
sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *
from own_python.Transfer_data import *
from validation_test.turbulent_flow import *
from validation_test.free_surface_flow import *
from validation_test.Tools_valid import *
from validation_test.final import *

# Units
kg = 1
m = 1
mm = 1e-3*m
s = 1

g = 9.81 * (m/s**2)

def main():

	vtk_folder = "my_output/local/bridge/r_3_33mm"


	U_0 = 0.36 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)
	r = 4 * (mm)

	h = 4*r
	particle = 2*r

	# Load vtk files
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=900)
	vtk_file = all_vtk[-1]
	#plot_vtk(vtk_file, mask=None, attribute='velocity')
	
	positions, values, derivative = spatial_derivative(
		vtk_file=vtk_file,
		attribute='velocity',
		plane='xy',
		axis='x', along = [5, 45],
		thickness=50*particle,
		component=0  # if velocity is selected
	)


	plt.figure(figsize=(12, 8))

	plt.plot(positions, derivative, 'r-o')
	plt.axhline(y=0, color='k', linestyle='--')  # Ligne horizontale à y=0
	plt.title('Dérivée spatiale de la vitesse (du_x/dx)')
	plt.xlabel('Position x')
	plt.ylabel('du_x/dx')
	plt.grid(True)

	plt.tight_layout()
	plt.show()
	

	#plot_vtk(vtk_file, mask=mask_line, attribute='velocity', r=r)

	plt.show()






if __name__ == "__main__":
	main()