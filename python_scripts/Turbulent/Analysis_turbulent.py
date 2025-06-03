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
from python_scripts.Tools_scenes import *
from python_scripts.Transfer_data import *
from python_scripts.Turbulent.Tools_turbulent import *
from Free_surface.Tools_free_surface import *
from python_scripts.Tools_global import *
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
	D = 2* 1.65 * (m)

	r = 20 * (mm)
	particle = 2*r
	FPS = 25
	t_sim = np.arange(0, 30 + 1) * FPS


	vtk_folder = "my_output/local/turbulent_pipe/r_20mm"
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=t_sim[15], max_timestep=t_sim[30]) # 250 -> 750
	savepath = 'Pictures/CH6_valid_test/turbulent/'
	configure_latex()

	#plot_vtk(all_vtk[-1], mask=None, attribute='velocity', is_plt=True, save=False, savepath=savepath+'/turbulent_velo')
	#plot_u_ghe()


	
	
	common_mult = {
		'vtk_files': all_vtk,
		'thickness': 5*particle,
	}

	common_single = {
		'vtk_files': all_vtk,
		'fixed_coord': 25 * (m),
		'thickness': 5*particle,

	}
		
	#=============================================#
	#               Check assumptions             #
	#=============================================#

	#u_centre  = get_multiple_slices(**common_mult, attribute='velocity', mean=True, trans_val=[-8*r, 8*r], along=[0, 45], plot=False, save=False, savepath=savepath+"/u_center")
	#u_slices  = get_multiple_slices(**common_mult, attribute='velocity', along=[0, 45], plot=False, save=False, savepath=savepath+"/u")
	#rho_slices  = get_multiple_slices(**common_mult, attribute='density', along=[0, 45], plot=False, save=False, savepath=savepath+"/rho")
	#rho_y_mult  = get_multiple_slices(**common_mult, attribute='density', axis='y', trans_val=[0, 46], plot=False, save=False, savepath=savepath+"/rho")
	u_single = get_single_slice(**common_single, attribute='velocity', plot=False, save=False, savepath="/rho")
	#mass_single = get_single_slice(**common_single, attribute='mass', plot=False, save=False, savepath="/mass")
	#rho_single = get_single_slice(**common_single, attribute='density', plot=False, save=False, savepath="/rho")
	

	#--------- Particle distribution -----------#
	#analyze_particle_distribution(all_vtk[-1], 25, delta_x=5*particle, n_bins=80, plot=True, save=True, savepath=savepath)
	#plot_particles_with_selection_rectangle(all_vtk, plane='xy', axis='x', fixed_coord=15, thickness=5*particle, trans_val=[-1.6, 1.6], save=True, savepath=savepath)

	#-------------- Steady state ---------------#
	#_, _ = time_derivative(mass_single, plot=True, save=True, savepath=savepath+"/mass")

	#------------ Fully developed --------------#
	#centreline_velocity(u_centre, save=True)
	
	

	#----------- Incompressibility -------------#
	#Q_v, Q_m = compute_flow_rate(U_0*D, 1000, u_slices[-1], rho_slices[-1], plot=True, save=True, savepath=savepath)
	#drhodx, _ = spatial_derivative(rho_x_mult, 'x', plot=True, save=True, savepath=savepath+"/rho_x")
	#drhody, _ = spatial_derivative(rho_y_mult, 'y', plot=True, save=False, savepath=savepath+"/rho_y")

	'''
	bounds = {'x_min':0, 'y_min':-1.615, 'x_max':45,'y_max':1.615, 'z_min':0,'z_max':0}
	min_bounds, max_bounds = get_global_bounds(all_vtk[-1], dimensions='3D', bounding=bounds)

	L_cell = 0.25*(4*particle)
	grid = create_grid(all_vtk[-1], [min_bounds, max_bounds], L_cell, plane='xy')
	W = extractKernel("cubic")
	#plot_grid_and_particles(all_vtk[-1], grid, min_bounds, max_bounds, L_cell, max_particles=5000)
	grid_rho = compute_grid_values(grid, all_vtk, 'density', 4*particle, W, component=None)
	drho_dy = spatial_derivative(grid_rho, 'y')
	
	fig, ax = plot_matrix_with_colorbar(drho_dy, 0, 45, -1.615, 1.615, vmin=-2, vmax=2, cbar_label=r"$\partial \rho/\partial y$ [Kg/m$^3$/m]", save=True, savepath=savepath+"/drhody_")
	'''

	#===================#
	#   Main analysis   #
	#===================#

	
	delta_raw, gamma_raw = fit_ghe_model(u_single, D/2, method='raw', 
                                    plot=True, save=True)

	# Version données moyennées
	delta_binned, gamma_binned = fit_ghe_model(u_single, D/2, method='binned',
											dy=0.02, plot=True, save=True)

	
	plt.show()






if __name__ == "__main__":
	main()