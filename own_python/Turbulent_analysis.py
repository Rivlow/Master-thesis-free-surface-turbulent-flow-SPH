import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 26, 26
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
mm = (1e-3) * (m)
s = 1
g = 9.81 * (m/s**2)

def main():

	U_0 = 5 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)
	D = 2* 1.97 * (m)

	r = 20 * (mm)
	particle = 2*r

	vtk_folder = "my_output/local/turbulent_pipe/r_20mm"
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=2150)
	last_vtk = all_vtk[-1]
	savepath = 'Pictures/CH5_valid_test/turbulent/'
	configure_latex()

	#plot_vtk(last_vtk, mask=None, attribute='velocity', is_plt=True, save=True, savepath=savepath+'/turb_velo')


	common_mult = {
		'vtk_file': last_vtk,
		'thickness': 5*particle,
	}

	common_single = {
		'vtk_files': all_vtk,
		'fixed_coord': 25 * (m),
		'thickness': 2*particle,

	}
	
	#-----------------------------------------------------------#
	# Get attribute over distance (assumption: unique timestep) #
	#-----------------------------------------------------------#
	#u_center  = get_multiple_slices(**common_mult, attribute='velocity', mean=True, trans_val=[-8*r, 8*r], along=[1, 45], plot=False, save=False, savepath=savepath+"/u_center")
	u_mult  = get_multiple_slices(**common_mult, attribute='velocity', along=[0, 46], plot=False, save=False, savepath=savepath+"/u")
	rho_x_mult  = get_multiple_slices(**common_mult, attribute='density', along=[0, 46], plot=False, save=False, savepath=savepath+"/rho")
	#rho_y_mult  = get_multiple_slices(**common_mult, attribute='density', axis='y', trans_val=[0, 46], plot=False, save=False, savepath=savepath+"/rho")
	Q_v, Q_m = compute_flow_rate(U_0*D, 1000, u_mult, rho_x_mult, plot=True, save=True, savepath=savepath)

	#-----------------------------------------------------------#
	#                  Get attribute over time                  #
	#-----------------------------------------------------------#
	#u_single = get_single_slice(**common_single, attribute='velocity', plot=False, save=False, savepath="/rho")
	#drhodx, _ = spatial_derivative(rho_x_mult, 'x', plot=True, save=True, savepath=savepath+"/rho_x")
	#drhody, _ = spatial_derivative(rho_y_mult, 'y', plot=True, save=False, savepath=savepath+"/rho_y")


	#_, _ = time_derivative(mass_single, plot=True, save=False, savepath=savepath+"/mass")
	#analyze_particle_distribution(last_vtk, 15, delta_x=5*particle, n_bins=80, plot=True, save=True, savepath=savepath)
	#plot_particles_with_selection_rectangle(all_vtk, plane='xy', axis='x', fixed_coord=15, thickness=5*particle, trans_val=[-1.6, 1.6], save=True, savepath=savepath)
	#fit_ghe_model(u_single, 1.615, plot=False, save=False)

	'''
	bounds = {'x_min':0, 'y_min':-3, 'x_max':45,'y_max':3, 'z_min':0,'z_max':0}
	min_bounds, max_bounds = get_global_bounds(all_vtk[-1], dimensions='3D', bounding=bounds)

	L_cell = 0.15*(4*particle)
	grid = create_grid(all_vtk[-1], [min_bounds, max_bounds], L_cell, plane='xy')
	W = extractKernel("cubic")
	#plot_grid_and_particles(all_vtk[-1], grid, min_bounds, max_bounds, L_cell, max_particles=5000)
	grid_rho = compute_grid_values(grid, all_vtk, 'density', 4*particle, W, component=0)
	drho_dx  = spatial_derivative(grid_rho, 'y')
	
	fig, ax = plot_matrix_with_colorbar(drho_dx, 0, 45, -1.97, 1.97, vmin=-2, vmax=2, cbar_label=r"d$\rho$/dy", save=False, savepath=savepath+"/drhody_averaged_")
	'''
	plt.show()






if __name__ == "__main__":
	main()