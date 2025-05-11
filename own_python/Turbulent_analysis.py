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

	U_0 = 5 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)
	D = 2* 1.615 * (m)

	r = 4 * (mm)
	particle = 2*r

	vtk_folder = "my_output/local/turbulent_pipe/r_004"
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=400)
	vtk_file = all_vtk[-1]

	#plot_vtk(vtk_file, mask=None, attribute='velocity')
	common_mult = {
		'vtk_file': vtk_file,
		'plane': 'xy',
		'axis': 'x',
		'along': None,
		'thickness': 5*particle,
	}

	common_single = {
		'vtk_files': all_vtk,
		'plane': 'xy',
		'axis': 'x',
		'fixed_coord': 15 * (m),
		'thickness': 5*particle,
	}
	
	# It is assumed outputs at a single timestep
	u_slices  = get_multiple_slices(
		**common_mult,
		attribute='velocity',
		component=0  
	)

	rho_slices  = get_multiple_slices(
		**common_mult,
		attribute='density'
	)

	rho_single = get_single_slice(
		**common_single, 
		attribute='density')
	

	
	Q_v, Q_m = compute_flow_rate(U_0*D, 1000, 
							  u_slices, rho_slices, 
							  plot=True, save=False, savepath = 'Pictures/CH5_valid_test/turbulent')

	du_dx, pos = spatial_derivative(u_slices)
	drho_dt, times = time_derivative(rho_single)

	
	plt.figure(figsize=(12, 8))

	plt.plot(pos, du_dx, 'r-o')
	plt.axhline(y=0, color='k', linestyle='--')  # Ligne horizontale à y=0
	plt.title('Dérivée spatiale de la vitesse')
	plt.xlabel('Position x')
	plt.ylabel('du_x/dx')
	plt.grid(True)

	plt.tight_layout()

	plt.figure(figsize=(12, 8))

	plt.plot(times, drho_dt, 'r-o')
	plt.axhline(y=0, color='k', linestyle='--')  # Ligne horizontale à y=0
	plt.title('Dérivée temporelle de rho ')
	plt.xlabel('temps t')
	plt.ylabel('du_rho/ddt')
	plt.grid(True)

	plt.tight_layout()
	
	plt.show()
	

	#inside_mask, rectangle = find_particles_in_rectangle(last_vtk.points, x_min, y_min, x_max, y_max)
	#projected_points, projected_attributes, vertical_line = project_particles(last_vtk, inside_mask, rectangle)
	#visualize_results(last_vtk, inside_mask, projected_points, rectangle, vertical_line)
	'''
	single_data = vtk_data, attributes, dimensions, save_path,
							x, y_min, y_max,
							num_slices, slice_width, 
							remove=False, plot=False, save=False)
	'''
	#fit_ghe_model(u_all, y_all, y_min, plot=True, save=True)

	'''
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
	'''
	#integrate_slice(Q_init, x_span, u_all, y_all, save=False)

	plt.show()






if __name__ == "__main__":
	main()