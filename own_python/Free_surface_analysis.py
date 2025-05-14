import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import time


# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 122, 26, 26
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

def calculate_nb_elem(bounds, target_cell_size):
	"""
	Calcule le nombre d'éléments nécessaire pour obtenir des cellules de taille spécifiée.
	
	Args:
		bounds: Les limites de la grille [(x_min, y_min, z_min), (x_max, y_max, z_max)] ou dict
		target_cell_size: Taille de cellule souhaitée (e.g., 4*r)
		
	Returns:
		nb_elem: Nombre d'éléments recommandé
	"""
	# Traiter les différents formats de bounds
	if isinstance(bounds, dict):
		# Format dictionnaire
		if 'x_min' in bounds and 'x_max' in bounds:
			x_range = bounds['x_max'] - bounds['x_min']
		else:
			x_range = None
			
		if 'y_min' in bounds and 'y_max' in bounds:
			y_range = bounds['y_max'] - bounds['y_min']
		else:
			y_range = None
			
		if 'z_min' in bounds and 'z_max' in bounds:
			z_range = bounds['z_max'] - bounds['z_min']
		else:
			z_range = None
	else:
		# Format liste/tuple [(min_x, min_y, min_z), (max_x, max_y, max_z)]
		min_bounds, max_bounds = bounds
		
		if len(min_bounds) >= 1 and len(max_bounds) >= 1:
			x_range = max_bounds[0] - min_bounds[0]
		else:
			x_range = None
			
		if len(min_bounds) >= 2 and len(max_bounds) >= 2:
			y_range = max_bounds[1] - min_bounds[1]
		else:
			y_range = None
			
		if len(min_bounds) >= 3 and len(max_bounds) >= 3:
			z_range = max_bounds[2] - min_bounds[2]
		else:
			z_range = None
	
	# Calculer nb_elem pour chaque dimension
	nb_elems = []
	
	if x_range is not None:
		nb_elems.append(int(np.ceil(x_range / target_cell_size)))
	
	if y_range is not None:
		nb_elems.append(int(np.ceil(y_range / target_cell_size)))
	
	if z_range is not None:
		nb_elems.append(int(np.ceil(z_range / target_cell_size)))
	
	# Retourner le maximum (pour s'assurer que toutes les cellules sont au plus de taille target_cell_size)
	if nb_elems:
		return max(nb_elems)
	else:
		raise ValueError("Impossible de calculer nb_elem à partir des bornes fournies")

def main():

	U_0 = 0.36 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)

	r = 4 * (mm)
	particle = 2*r
	

	# Load vtk files
	vtk_folder = "my_output/local/free_surface/r_4mm"
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=200)
	last_vtk = all_vtk[-1]

	savepath = 'Pictures/CH5_valid_test/free_surface'
	#configure_latex()
	
	# Dimensions
	Lx_1 = 4.5 * (m)
	Ly = 0.02 * (m)
	Ly_emit = 0.5 * (m)
	Lx_emit = particle 
	trans_emit = [-Lx_emit/2 + particle, Ly_emit/2 + particle, 0]
	y_pos = calculate_emitter_particle_positions(trans_emit, int(Lx_emit / (2 * r)), int(Ly_emit / (2 * r)), r)
	Q_init = np.max(y_pos)*U_0

	points, h_sph, u_sph, Fr_sph, Head_rel_sph, Head_abs_sph = extract_water_height(all_vtk[-1], plot=False, save=False)
	points[:,1] += 0.015
	h_sph[:,1] += 0.015

	#plot_vtk(all_vtk[139], mask=None, attribute='velocity', is_plt=True, save=True, savepath=savepath+'/step4')

	common_mult = {
		'vtk_file': last_vtk,
		'plane': 'xy',
		'axis': 'x',
		'along': [15, 24],
		'thickness': 5*particle,
	}

	common_single = {
		'vtk_files': all_vtk,
		'plane': 'xy',
		'axis': 'x',
		'fixed_coord': 15 * (m),
		'thickness': 5*particle,
	}

	#u_mult = get_multiple_slices(**common_mult, attribute='velocity',component=0)
	#p_rho2_mult = get_multiple_slices(**common_mult, attribute='p_/_rho^2')
	#rho_mult = get_multiple_slices(**common_mult, attribute='density')
	

	bounds = {'x_min':0, 'y_min':0, 'x_max':25,'y_max':1}
	nb_elem = calculate_nb_elem(bounds, 4*r)//4
	min_bounds, max_bounds = get_global_bounds(all_vtk[-1], dimensions='2D', bounding=bounds)
	grid = create_grid(all_vtk[-1], [min_bounds, max_bounds], nb_elem=nb_elem, dimensions='2D', plane='xy')


	W = extractKernel('cubic')
	
	grid_rho = compute_grid_values(grid, all_vtk[-1], 'density', r, W)
	print('test')
	for rho in grid_rho:
		print(rho)
	drhodx = spatial_derivative(grid_rho, 'density', deriv_axis='x')
	drhody = spatial_derivative(grid_rho, 'density', deriv_axis='y')

	# U= (u,v,w)
	'''
	grid_velo = compute_grid_values(grid, all_vtk[-1], 'velocity', r, W)
	dUdx = spatial_derivative(grid_velo, 'velocity', deriv_axis='x')
	dUdy = spatial_derivative(grid_velo, 'velocity', deriv_axis='y')

	dudx = dUdx[:,:,0]
	dvdy = dUdy[:,:,1]
	dudy = dUdx[:,:,1]
	'''

	# 4. Visualiser les résultats

	#plt.figure()
	#plt.imshow(drhody+drhodx, origin='lower')
	#plt.title('div(u)')
	#plt.colorbar()

	#plt.subplot(1, 3, 3)
	#plt.imshow(drhody, origin='lower')
	#plt.title('drho/dy')
	#plt.colorbar()

	#plt.tight_layout()
	#plt.show()

	'''
	# 5. Calculer la divergence (∇·v = dvx/dx + dvy/dy)
	divergence = dudx + dvdy

	# 6. Visualiser la divergence
	plt.figure(figsize=(6, 5))
	plt.imshow(divergence, origin='lower', vmin = -10, vmax=50)
	plt.title('Divergence de la vitesse')
	plt.colorbar()
	plt.tight_layout()
	plt.show()
	'''

	'''
		#is_hydrostatic(p_rho2_mult, rho_mult, save=False, savepath=savepath+'/')
		#check_hydrostatic(p_rho2_mult, rho_mult, 0, 0.33, plot=True, save=False)


	#rho_slices  = get_multiple_slices(**common_mult, attribute='density')
	
	#Q_v, Q_m = compute_flow_rate(Q_init, 1000, u_slices, rho_slices, plot=True, save=True, savepath = savepath+'/free_surf_')
	
	#du_dx, pos = spatial_derivative(u_slices, plot=False, save=False, savepath = savepath+"/u")
	#drho_dt, times = time_derivative(att_single, plot=True, save=False, savepath = savepath+"/rho")

	#
	
	
	#x_th, z_th, h_th, Fr_th, H_inlet, H_outlet = compute_theoretical_water_height(Q_init)
	#plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False)
	#plot_Fr(h_sph[:,0], x_th, Fr_sph, Fr_th, save=True, savepath=savepath)
	#plot_Head(h_sph[:,0], Head_abs_sph, H_inlet, H_outlet, save=True, savepath=savepath)
	'''


	#plt.show()






if __name__ == "__main__":
	main()