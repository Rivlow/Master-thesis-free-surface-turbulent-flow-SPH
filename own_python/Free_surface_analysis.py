import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import time


# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 22, 26, 26
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
from Misc.kernels import *

# Units
kg = 1
m = 1
mm = 1e-3*m
s = 1

g = 9.81 * (m/s**2)

def main():

	U_0 = 0.36 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)

	r = 4 * (mm)
	particle = 2*r
	

	# Load vtk files
	vtk_folder = "my_output/local/free_surface/r_4mm/reduced_domain/"
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=0)
	last_vtk = all_vtk[-1]

	savepath = 'Pictures/CH5_valid_test/free_surface/'
	#configure_latex()
	
	# Dimensions
	Lx_1 = 4.5 * (m)
	Ly = 0.02 * (m)
	Ly_emit = 0.5 * (m)
	Lx_emit = particle 
	trans_emit = [-Lx_emit/2 + particle, Ly_emit/2 + particle, 0]
	y_pos = calculate_emitter_particle_positions(trans_emit, int(Lx_emit / (2 * r)), int(Ly_emit / (2 * r)), r)
	Q_init = np.max(y_pos)*U_0

	common_mult = {
		'vtk_file': last_vtk,
		'plane': 'xy',
		'axis': 'x',
		'along': [0, 24],
		'trans_val': [0.05, 0.3],
		'thickness': 10*particle,
	}

	common_single = {
		'vtk_files': all_vtk,
		'plane': 'xy',
		'axis': 'x',
		'fixed_coord': 5 * (m),
		'thickness': 2.5,
	}

	

	#-------------------#
	#   Main analysis   #
	#-------------------#

	points, h_sph, u_sph, Fr_sph, Head_rel_sph, Head_abs_sph = extract_water_height(all_vtk[-1], plot=False, save=False)
	
	points[:,1] += 0.015
	h_sph[:,1] += 0.015

	points[:,0] += 7
	h_sph[:,0] += 7
	x_th, z_th, h_th, Fr_th, H_inlet, H_outlet = compute_theoretical_water_height(Q_init, nb_points=len(h_sph[:,1]))

	#plot_Fr(h_sph[:,0], x_th, Fr_sph, Fr_th, save=True, savepath=savepath)
	#plot_Head(h_sph[:,0], Head_abs_sph, H_inlet, H_outlet, save=True, savepath=savepath)
	
	plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False, savepath=savepath)

	
	
	#---------------------------------------------#
	#   Hydrostatic/velocity check  (global form) #
	#---------------------------------------------#
	#u_mult = get_multiple_slices(**common_mult, attribute='velocity',component=0)
	p_rho2_mult = get_multiple_slices(**common_mult, attribute='p_/_rho^2')
	rho_mult = get_multiple_slices(**common_mult, attribute='density')
	#check_hydrostatic(p_rho2_mult, rho_mult, 0, 0.33, plot=True, save=False)

	#is_uniform(u_mult, save=False, savepath=savepath)
	#rho_slices  = get_multiple_slices(**common_mult, attribute='density')
	#Q_v, Q_m = compute_flow_rate(Q_init, 1000, u_slices, rho_slices, plot=True, save=True, savepath = savepath+'/free_surface_')
	#plot_vtk(all_vtk[-1], mask=None, attribute='angular_velocity', is_plt=True, save=False, savepath=savepath+'/vorticity')

	

	#--------------------------------------#
	#   Derivative analysis (local form)   #
	#--------------------------------------#

	'''
	bounds = {'x_min':0, 'y_min':0, 'x_max':25,'y_max':1, 'z_min':0,'z_max':0}
	min_bounds, max_bounds = get_global_bounds(all_vtk[-1], dimensions='3D', bounding=bounds)

	L_cell = 0.05*(4*particle)
	grid = create_grid(all_vtk[-1], [min_bounds, max_bounds], L_cell, plane='xy')
	W = extractKernel("cubic")
	#plot_grid_and_particles(all_vtk[-1], grid, min_bounds, max_bounds, L_cell, max_particles=5000)
	grid_rho = grid = compute_grid_values(grid, all_vtk[-1], 'velocity', 4*particle, W, component=0)
	axis = {'x':0, 'y':1, 'z':2}
	d_rho_dx  = spatial_derivative(grid_rho, axis['x'])
	
	plt.figure()
	plt.imshow(d_rho_dx.T, origin='lower', aspect='auto')
	'''


	plt.show()






if __name__ == "__main__":
	main()