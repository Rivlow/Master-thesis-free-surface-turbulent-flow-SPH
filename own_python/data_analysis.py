import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 10, 16, 18
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=MEDIUM_SIZE)  
plt.rc('ytick', labelsize=MEDIUM_SIZE)   
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE)

# Configure project path
sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *
from validation_test.turbulent_flow import *
from validation_test.free_surface_flow import *
from own_python.Transfer_data import *



def main():

	vtk_folder="output/free_surface_2D/dist_x_1.5_Ly_0_12"

	U_0 = 5
	nu = 1e-6
	rho = 1e3
	r = 0.005
	h = 4*r
	particle = 2*r

	mm = 1e-3

	# Load vtk files    
	all_vtk = load_vtk_files(vtk_folder, print_files=False)


	#---------------------#
	# Turbulence analysis #
	#---------------------#

	'''
	#fig = visualize_particles(all_vtk[-1], velocity_component='x', show=False)
	steady_vtk = all_vtk[470:]
	last_vtk = steady_vtk[-1]

	x_min, y_min = 35, -1.615
	x_max, y_max = x_min + 5*particle, 1.615

	Q_init = U_0*np.abs(y_max - y_min)


	inside_mask, rectangle = find_particles_in_rectangle(last_vtk.points, x_min, y_min, x_max, y_max)

	projected_points, projected_attributes, vertical_line = project_particles(
			last_vtk, inside_mask, rectangle)
		
	visualize_results(last_vtk, inside_mask, projected_points, rectangle, vertical_line)



	u_all, rho_all, y_all = single_slice(steady_vtk, y_min, y_max, x_pos=35, slice_width=2*particle, plot=False)
	fit_results = fit_ghe_model(u_all, y_all) 

	x_start, x_end = 1, 45
	u_all, y_all = multiple_slices(steady_vtk[-1],
										x_start=x_start, x_end=x_end,
										num_slices=500,
										y_min=y_min, y_max=y_max,
										slice_width=2*particle,
										plot=False)

	x_span = np.linspace(x_start, x_end, len(u_all))
	integrate_slice(Q_init, x_span, u_all, y_all)
	'''
      
    #-----------------------#
	# Free surface analysis #
	#-----------------------#
      
	steady_vtk = all_vtk[780:]
	extract_water_height(steady_vtk[-1], save=False)

	
    
if __name__ == "__main__":
    main()