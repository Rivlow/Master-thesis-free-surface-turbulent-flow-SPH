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





def main():

	vtk_folder="my_output/local/free_surface/U_036/test"
	#vtk_folder = "my_output/turbulent_pipe/r_004"
	# Units

	kg = 1
	m = 1
	mm = 1e-3*m
	s = 1

	g = 9.81 * (m/s**2)
	U_0 = 0.36 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)
	r = 2.5 * (mm)
	h = 4*r

	particle = 2*r

	# Load vtk files
	all_vtk = load_vtk_files(vtk_folder, print_files=False)

	'''
	#---------------------#
	# Turbulence analysis #
	#---------------------#

	#fig = visualize_particles(all_vtk[-1], velocity_component='x', show=False)
	steady_vtk = all_vtk[485:]
	last_vtk = steady_vtk[-1]

	x_min, y_min = 35, -1.615
	x_max, y_max = x_min + 55*particle, 1.615

	Q_init = U_0*np.abs(y_max - y_min)


	#inside_mask, rectangle = find_particles_in_rectangle(last_vtk.points, x_min, y_min, x_max, y_max)
	#projected_points, projected_attributes, vertical_line = project_particles(last_vtk, inside_mask, rectangle)
	#visualize_results(last_vtk, inside_mask, projected_points, rectangle, vertical_line)

	single_data = vtk_data, attributes, dimensions, save_path,
							x, y_min, y_max,
							num_slices, slice_width, 
							remove=False, plot=False, save=False)
	#fit_ghe_model(u_all, y_all, y_min, plot=True, save=True)

	x_start, x_end = 1, 45
	u_all, y_all = multiple_slices(steady_vtk[-1],
										x_start=x_start, x_end=x_end,
										num_slices=100,
										y_min=y_min, y_max=y_max,
										slice_width=2*particle,
										plot=False,
										save=False)

	x_span = np.linspace(x_start, x_end, len(u_all))
	center_line(x_span, u_all, save=True)
	#integrate_slice(Q_init, x_span, u_all, y_all, save=False)

	plt.show()
	'''

	#-----------------------#
	# Free surface analysis #
	#-----------------------#

	# Dimensions
	Lx_1 = 4.5* (m)
	Ly = 0.02* (m)
	Ly_emit = 0.5 * (m)

	steady_vtk = all_vtk[100:]
	points, h_sph, u_sph = extract_water_height(steady_vtk[-1], plot=False, save=False)
	points[:,1] += 0.015
	h_sph[:,1] += 0.015

	xy_init = [15, 0]
	xy_final = [24, Ly_emit]
	attributes = ['velocity', 'density', 'p_/_rho^2']
	dimensions = {"velocity": r"[m/s]",
				  "density": r"[kg/$m^3$]",
				  "p_/_rho^2": r"[Pa]",
				  "temperature": r"[K]",
				  "viscosity": r"[Pa·s]"}
	
	multiple_data = multiple_slices_2D(steady_vtk[-1], attributes, dimensions, "free_surface_",
									   xy_init, xy_final,
									   num_slices=50, slice_width= 4*particle,
									   remove=False, plot=True, save=True)
	
	
	checkHydrostatic(multiple_data)

	xy_init = [0, 0]
	xy_final = [24, Ly_emit]
	Q_init_num = integrate_slice(multiple_data,
								 xy_init[0], xy_final[0],
								 Q_init=0.18, rho_0=1000,
								 plot=True,
								 save=True)
	

	x_th, z_th, h_th, Fr_th = compute_theoretical_water_height(Q_init_num)
	plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=True)
	

	plt.show()






if __name__ == "__main__":
	main()