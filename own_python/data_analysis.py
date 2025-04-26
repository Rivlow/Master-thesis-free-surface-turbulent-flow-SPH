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
from validation_test.turbulent_flow import *
from validation_test.free_surface_flow import *
from own_python.Transfer_data import *




def main():

	vtk_folder="my_output/free_surface/r_0005"
	#vtk_folder = "my_output/turbulent_pipe/r_004"
	# Units

	kg = 1
	mm = 1e-3
	m = 1
	s = 1

	g = 9.81*(m/s**2)

	U_0 = 0.36*(m/s)
	nu = 1e-6*(m**2/s)
	rho = 1e3*(kg/m**3)
	r = 5*mm
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

	#u_all, rho_all, y_all = single_slice(steady_vtk, y_min, y_max, x_pos=35, slice_width=13*particle, plot=True, save=True)
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
	Lx_1 = 4.5*m
	Ly = 0.02*m
	emit_H = 0.5

	steady_vtk = all_vtk[100:]
	points, h_sph, u_sph = extract_water_height(steady_vtk[-1], plot=False, save=False)

	x_th, z_th, h_th, Fr_th = compute_theoretical_water_height(U_0)

	points[:,1] += 0.015
	h_sph[:,1] += 0.015

	u_all, y_all, rho_all = multiple_slices(steady_vtk[-1],
											x_start=0, x_end=25,
											num_slices=50,
											y_min=0, y_max=emit_H,
											slice_width=2*particle,
											remove=False, plot=False, save=False)


	'''
	Q_init = U_0*(np.max(y_all[0]) - np.min(y_all[0]))
	x_span = np.linspace(0, 25, len(u_all))
	integrate_slice(Q_init, 1000, x_span, u_all, y_all, rho_all, save=False)
	
	# Compare actual flow rate (SPH vs Theory)
	x_span = np.arange(24, 24+4*particle, 2*particle)
	Q_m = compute_mass_flow_rate(x_span, steady_vtk[-1]) # emitter generates first particles which does not have good density
	Q_v = Q_m/1000

	print(f'\nTrue mass flow rate (inlet) = {Q_m} [kg/s]')
	print(f'True volume flow rate (inlet) = {Q_v} [m^2/s]')
	print(f'True outlet velocity (h_out = 0.33 [m]) = {Q_v/0.33} [m/s]\n')
	print(f'Pseudo volumic flow rate (inlet)= {U_0*emit_H} [m^2/s]')
	print(f'Pseudo outlet velocity (h_out = 0.33 [m]) = {U_0*emit_H/0.33} [m/s]')
	'''	

	plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False)

	plt.show()






if __name__ == "__main__":
	main()