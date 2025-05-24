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

	vtk_folder = "my_output/local/bridge/r_4mm/h_200mm"
	savepath = 'Pictures/CH5_valid_test/bridge/'


	U_0 = 0.36/2 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)
	r = 4 * (mm)

	h = 4*r
	particle = 2*r

	# Load vtk files
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=700)
	last_vtk = all_vtk[-1]
	#plot_vtk(vtk_file, mask=None, attribute='velocity')
	
	mid_plane, mask_plane = project_surface(last_vtk.points, "z", -150*mm, thickness=2*particle)

	points, h_sph, u_sph, Fr_sph, Head_rel_sph, Head_abs_sph = extract_water_height(last_vtk, mask=mask_plane, plot=False, save=False)
	plot_Head(h_sph[:,0], Head_abs_sph, 0, 1, save=False, savepath=savepath)

	plotter = plot_vtk(last_vtk, mask=mask_plane, attribute='velocity')

	plt.show()






if __name__ == "__main__":
	main()