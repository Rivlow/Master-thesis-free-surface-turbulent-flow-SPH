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

	vtk_folder="output/turbulent_pipe/r_004"

	# Units
	kg = 1
	mm = 1e-3
	m = 1
	s = 1
      
	g = 9.81*(m/s**2)
      
	U_0 = 5*(m/s)
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
										num_slices=250,
										y_min=y_min, y_max=y_max,
										slice_width=2*particle,
										plot=False, 
                                        save=False)

	x_span = np.linspace(x_start, x_end, len(u_all))
	integrate_slice(Q_init, x_span, u_all, y_all)
      
	plt.show()
	
      
	'''
	#-----------------------#
	# Free surface analysis #
	#-----------------------#

	Lx_1 = 4.5*m
	Ly = 0.02*m
	dist_x_obstacle = 4.5*m # 1.5*m (pour le ""bon""")

	steady_vtk = all_vtk[3900:]
	points, h_sph, u_sph = extract_water_height(steady_vtk[-1], plot=False, save=False)
	x_th, z_th, h_th, Fr_th = theoretical_water_height(U_0)

	
	offset_x = dist_x_obstacle 
	h_sph[:, 0] +=  5.4 - 0.083 - 1.352
	points[:, 0] += 5.4 - 0.083 - 1.352
	h_sph[:, 1] -= Ly - 0.01 
	points[:, 1] -= Ly - 0.01

	fig, ax1 = plt.subplots(figsize=(10, 6))

	ax1.scatter(points[:, 0], points[:, 1], s=5, color='blue', label='Particles')
	ax1.scatter(h_sph[:, 0], h_sph[:, 1], s=20, color='orange', label='SPH free surface')

	ax1.plot(x_th, z_th, 'k-', label='Topography')
	ax1.fill_between(x_th, z_th, z_th + h_th, color='lightblue', alpha=0.5)
	ax1.plot(x_th, h_th + z_th, color='darkblue', label='Theoretical free surface')

	#ax1.set_xlim(7, 18)
	ax1.set_xlabel('Distance x [m]')
	ax1.set_ylabel('Height [m]')
	ax1.legend(loc='best')
	ax1.grid(True)

	plt.tight_layout()
      
	#plt.savefig("Pictures/CH5_valid_test/water_height_last.pdf")
	plt.show()
      
	

	
    
if __name__ == "__main__":
    main()