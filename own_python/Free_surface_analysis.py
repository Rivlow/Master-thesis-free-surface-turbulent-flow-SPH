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

	U_0 = 0.36 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)

	r = 4 * (mm)
	particle = 2*r

	# Load vtk files
	#vtk_folder = "my_output/local/free_surface/r_10mm/corrected"
	vtk_folder = "my_output/local/free_surface/test_time"
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=0)
	vtk_file = all_vtk[-1]
	
	# Dimensions
	Lx_1 = 4.5 * (m)
	Ly = 0.02 * (m)
	Ly_emit = 0.5 * (m)
	Lx_emit = particle 

	points, h_sph, u_sph = extract_water_height(all_vtk[-1], plot=False, save=False)
	points[:,1] += 0.015
	h_sph[:,1] += 0.015

	xy_init = [0, 0.05]
	xy_final = [24, Ly_emit]
	attributes = ['velocity', 'density', 'p_/_rho^2']
	dimensions = {"velocity": r"[m/s]",
				  "density": r"[kg/$m^3$]",
				  "p_/_rho^2": r"[Pa]"}
	
	common_args = {
		'vtk_file': vtk_file,
		'plane': 'xy',
		'axis': 'x',
		'along': None,
		'thickness': 10*particle,
	}
	u_slices  = get_multiple_slices(
		**common_args,
		attribute='velocity',
		component=0  
	)

	rho_slices  = get_multiple_slices(
		**common_args,
		attribute='density'
	)

	Q_v, Q_m = compute_flow_rate(U_0*Ly_emit, 1000, 
							  u_slices, rho_slices, 
							  plot=False, save=False, savepath = 'Pictures/CH5_valid_test/free_surcace')
	
	
	'''
	multiple_data = multiple_slices_2D(all_vtk[-1], attributes, dimensions, "free_surface_",
									   xy_init, xy_final,
									   num_slices=100, slice_width= 4*particle,
									   remove=False, plot=False, save=False)
	
	#dt = 0.0002054976939689368 * (s) # r = 5 mm
	dt = 0.0004027805698569864 * (s) # r = 10mm
	check_hydrostatic(multiple_data, dt,
					  0, 0.33, 
					  plot=False, save=False)
	
	trans_emit = [-Lx_emit/2 + particle, Ly_emit/2 + particle, 0]
	y_pos = calculate_emitter_particle_positions(trans_emit, int(Lx_emit / (2 * r)), int(Ly_emit / (2 * r)), r)
	Q_init = U_0*(np.max(y_pos))
	Q_init_num = integrate_slice(multiple_data,
								 x_start=0, x_end=24,
								 r=r, 
								 Q_init=Q_init, rho_0=1000,
								 coef=0.8,
								 plot=True, save=False)
	
	
	x_th, z_th, h_th, Fr_th = compute_theoretical_water_height(0.18)
	#plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False)
	#annotate_hydraulic_regions(x_th, z_th, h_th, Fr_th, save=True)
	#plot_conjugate_height(x_th, z_th, h_th, Fr_th, save=True)
	'''

	plt.show()






if __name__ == "__main__":
	main()