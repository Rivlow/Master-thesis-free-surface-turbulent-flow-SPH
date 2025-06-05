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
from python_scripts.Tools_scenes import *
from python_scripts.Transfer_data import *
from python_scripts.Free_surface.Tools_free_surface import *
from python_scripts.Tools_global import *



def create_bridge_obstacles():
	"""
	Crée les obstacles du pont dans le plan X-Z pour plot_streamlines.
	Retourne une liste d'obstacles au format compatible avec plot_streamlines.
	"""
	mm = 1e-3
	
	Lx_roof = 500 * mm
	Ly_roof = 64 * mm
	Lz_roof = 986 * mm
	Lx_foot = 500 * mm
	Ly_foot = 211 * mm
	Lz_foot = 64 * mm
	x_entrance = 1.5
	
	offset = (Lz_roof - Lz_foot) / 2
	
	obstacles = []
	
	# Left pier 
	obstacles.append({
		'type': 'rectangle',
		'x_min': x_entrance,
		'x_max': x_entrance + Lx_foot,
		'y_min': 0,
		'y_max': Ly_foot,
		'z_min': -Lz_foot/2 - offset/2 - Lz_foot/2,
		'z_max': -Lz_foot/2 - offset/2 + Lz_foot/2
	})
	
	# Central pier
	obstacles.append({
		'type': 'rectangle',
		'x_min': x_entrance,
		'x_max': x_entrance + Lx_foot,
		'y_min': 0,
		'y_max': Ly_foot,
		'z_min': -Lz_foot/2,
		'z_max': Lz_foot/2
	})
	
	# Right pier (
	obstacles.append({
		'type': 'rectangle',
		'x_min': x_entrance,
		'x_max': x_entrance + Lx_foot,
		'y_min': 0,
		'y_max': Ly_foot,
		'z_min': Lz_foot/2 + offset/2 - Lz_foot/2,
		'z_max': Lz_foot/2 + offset/2 + Lz_foot/2
	})
	

	# Roof 
	obstacles.append({
		'type': 'rectangle',
		'x_min': x_entrance,
		'x_max': x_entrance + Lx_roof,
		'y_min': Ly_foot,
		'y_max': Ly_foot+Ly_roof,
		'z_min': -Lz_roof/2,
		'z_max': Lz_roof/2
	})
	
	
	return obstacles


# Units
kg = 1
m = 1
mm = 1e-3*m
s = 1

g = 9.81 * (m/s**2)

def main():

	vtk_folder = "my_output/local/bridge/r_5mm/free_surface"
	#vtk_folder = "my_output/local/bridge/r_5mm/pressurized"
	savepath = 'Pictures/CH8_final_simulation/'
	configure_latex()


	U_0 = 0.15/2 * (m/s)
	nu = 1e-6 * (m**2/s)
	rho = 1e3 * (kg/m**3)
	r = 5 * (mm)

	h = 4*r
	particle = 2*r

	# Load vtk files
	all_vtk = load_vtk_files(vtk_folder, print_files=False, min_timestep=0)
	last_vtk = all_vtk[-1]
	#plot_vtk(last_vtk, mask=None, attribute='velocity', is_plt=True)

	#===================#
	#   Main analysis   #
	#===================#

	obstacles = create_bridge_obstacles()

	_, mask_XZ = project_surface(last_vtk.points, "y", 120*mm, thickness=4*particle)
	_, mask_XY_1 = project_surface(last_vtk.points, "z", 150*mm, thickness=10*particle, bounds=None)
	_, mask_XY_2 = project_surface(last_vtk.points, "z", -150*mm, thickness=10*particle, bounds=None)

	pos_1, h_1, u_top, Fr_1, Head_rel, Head_abs = extract_water_height(last_vtk, mask=mask_XY_1, plot=False, save=False)
	pos_2, h_2, u_top, Fr_2, Head_rel, Head_abs = extract_water_height(last_vtk, mask=mask_XY_2, plot=False, save=False)

	#plot_water_height(particle, points=[pos_1, pos_2], h_sph=[h_1, h_2], label=['Slice at z = 150 [mm]', 'Slice at z = -150 [mm]'], obstacles=obstacles, save=True, savepath=savepath+'free_surf_two_')
	#plot_Fr([h_1[:,0], h_2[:,0]], [Fr_1, Fr_2], label=['Slice at z = 150 [mm]', 'Slice at z = -150 [mm]'], save=True, savepath=savepath+'free_surf_')
	#mean_head(all_vtk, particle, save=True, savepath=savepath)
	#plot_Head(x=h[:,0], H=Head_abs, H_inlet=None, H_outlet=None, save=False, savepath=savepath+'free_surf_')
	
	
	position = last_vtk.points[mask_XZ]
	velocity = last_vtk.point_data['velocity'][mask_XZ]
	
	x, y, z = position[:, 0], position[:, 1], position[:, 2]
	u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]

	#plot_quiver(x, y, u, v, save=False, savepath=savepath+'free_surf_')
	
	
	
	plot_streamlines(
		x, z, u, w,
		plane="XZ", apply_filter=True, min_speed_ratio=0.0,
		obstacles=obstacles,
		margin=0.05,
		nx=15000, ny=15000, density=1,
		save=True,
		savepath=savepath+'pressurized'
	)

	



	plt.show()







if __name__ == "__main__":
	main()