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
from python_scripts.Tools_scenes import *
from python_scripts.Transfer_data import *
from python_scripts.Turbulent.Tools_turbulent import *
from python_scripts.Free_surface.Tools_free_surface import *
from python_scripts.Tools_global import *
from python_scripts.Misc.kernels import *

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
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=190)
	last_vtk = all_vtk[-1]

	savepath = 'Pictures/CH6_valid_test/free_surface/'
	configure_latex()
	
	# Dimensions
	Lx_1 = 4.5 * (m)
	Ly = 0.02 * (m)
	Ly_emit = 0.5 * (m)
	Lx_emit = particle 
	trans_emit = [-Lx_emit/2 + particle, Ly_emit/2 + particle, 0]
	y_pos = calculate_emitter_particle_positions(trans_emit, int(Lx_emit / (2 * r)), int(Ly_emit / (2 * r)), r)
	Q_init = np.max(y_pos)*U_0

	common_mult = {
		'vtk_files': all_vtk,
		'plane': 'xy',
		'axis': 'x',
		'along': [0, 8.5],
		'trans_val': None,
		'thickness': 4*particle,
	}

	#=============================================#
	#               Check assumptions             #
	#=============================================#
	
	u_slices = get_multiple_slices(**common_mult, attribute='velocity', component=0)
	v_slices = get_multiple_slices(**common_mult, attribute='velocity', component=1)
	
	rho_slices = get_multiple_slices(**common_mult, attribute='density')
	mass_slices = get_multiple_slices(**common_mult, attribute='mass')
	p_rho2_slices = get_multiple_slices(**common_mult, attribute='p_/_rho^2')
	

	#-------------- Steady state ---------------#
	#E_tot, time, dE, dt = compute_E_tot(all_vtk, mass_slices[-1], u_slices[-1], v_slices[-1], p_rho2_slices[-1], plot=True, save=False, savepath=savepath)


	#-------- Uniform velocity profile ---------#
	is_uniform(u_slices[-1], save=False, savepath=savepath)


	#----------- Hydrostatic pressure ----------#
	is_hydrostatic(p_rho2_slices, rho_slices, 0, 0.33, plot=True, save=False, savepath=savepath)


	#----------- Incompressibility ------------#
	Q_v, Q_m = compute_flow_rate(Q_init, 1000, u_slices[-1], rho_slices[-1], plot=True, save=False, savepath=savepath)
	plot_vtk(all_vtk[-1], mask=None, attribute='angular_velocity', is_plt=True, save=False, savepath=savepath+'/vorticity')

	
	bounds = {'x_min':0, 'y_min':0, 'x_max':25,'y_max':1, 'z_min':0,'z_max':0}
	min_bounds, max_bounds = get_global_bounds(all_vtk[-1], dimensions='3D', bounding=bounds)

	L_cell = 0.05*(4*particle)
	grid = create_grid(all_vtk[-1], [min_bounds, max_bounds], L_cell, plane='xy')


	W = cubic_kernel

	#plot_grid_and_particles(all_vtk[-1], grid, min_bounds, max_bounds, L_cell, max_particles=5000)
	grid_rho = grid = compute_grid_values(grid, all_vtk[-1], 'velocity', 4*particle, W, component=0)
	axis = {'x':0, 'y':1, 'z':2}
	d_rho_dx  = spatial_derivative(grid_rho, axis['x'])
	
	plt.figure()
	plt.imshow(d_rho_dx.T, origin='lower', aspect='auto')
	


	

	#===================#
	#   Main analysis   #
	#===================#

	points, h_sph, u_sph, Fr_sph, Head_rel_sph, Head_abs_sph = extract_water_height(all_vtk[-1], plot=False, save=False)
	
	points[:,1] += 2*particle
	h_sph[:,1] += 2*particle

	points[:,0] += 7
	h_sph[:,0] += 7
	x_th, z_th, h_th, Fr_th, H_inlet, H_outlet = compute_theoretical_water_height(Q_init, nb_points=5000)


	position = last_vtk.points
	velocity = last_vtk.point_data['velocity']

	x = position[:, 0]+7
	y = position[:, 1]
	z = position[:, 2]
	u = velocity[:, 0]  # Composante X de la vitesse
	v = velocity[:, 1]  # Composante Y de la vitesse
	w = velocity[:, 2]  # Composante Z de la vitesse


	z_b = lambda x: 0.2 - 0.05*((x-10)**2)
	obstacle_curve = {
		'type': 'curve',
		'x_range': (8, 12),
		'y_func': z_b,
		'region': 'below'
	}

	
	fig, ax = plot_streamlines(
		x, y, u, v, 
		obstacles=[obstacle_curve],
		nx=1500, ny=1500, density=20,
		streamline_length=20,  
		min_speed=0.005,        
		max_dist=1*particle,     
	)
	

	plot_quiver(x, y, u, v, save=False, savepath=savepath)

	
	
	plot_Fr(h_sph[:,0], x_th, Fr_sph, Fr_th, save=False, savepath=savepath)
	plot_Head(h_sph[:,0], Head_abs_sph, H_inlet, H_outlet, save=False, savepath=savepath)
	plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False, savepath=savepath)
	plot_vtk(all_vtk[200], mask=None, attribute='velocity', save=False, savepath=savepath+"step_4", is_plt=True)
	plot_conjugate_height(x_th, z_th, h_th, Fr_th, save=False)
	annotate_hydraulic_regions(x_th, z_th, h_th, Fr_th, save=False)
	compute_E_tot(all_vtk, mass_slices[-1], u_slices[-1], v_slices[-1], savepath=savepath)
	


	plt.show()






if __name__ == "__main__":
	main()