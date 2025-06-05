import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.colors import Normalize
import numba as nb
import matplotlib.patches as patches
from scipy.spatial import cKDTree
from scipy.interpolate import griddata



# Local imports
sys.path.append(os.getcwd())
from python_scripts.Tools_scenes import *
from python_scripts.Tools_global import *


def configure_latex():
	"""Configure matplotlib to use LaTeX for rendering text."""
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')


# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 26, 26
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)



def get_name(slice):

	if slice['name'] == 'density':
		return r'$\rho$', r'[Kg/m$^3$]'
	elif slice['name'] == 'velocity':
		return 'u', 'm/s'
	elif slice['name'] == 'mass':
		return 'm', 'Kg'
	elif slice['name'] == 'p_/_rho^2':
		return 'm', 'Kg'
	
def closest_value(list, target):
	return min(list, key=lambda x: abs(x - target))

def plot_matrix_with_colorbar(matrix, x_min, x_max, y_min, y_max, num_ticks=5,  
							vmin=None, vmax=None, cbar_label="Valeur", 
							x_label="X [m]", y_label="Y [m]", grid=True,  
							save=False, savepath=None): 
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	# Utiliser extent pour définir les coordonnées physiques
	extent = [x_min, x_max, y_min, y_max]
	im = ax.imshow(matrix.T, origin='lower', aspect='auto', 
				extent=extent, vmin=vmin, vmax=vmax)
	
	# Maintenant les ticks correspondent aux coordonnées physiques
	ax.set_xticks(np.linspace(x_min, x_max, num_ticks))
	ax.set_xlabel(x_label)
	
	ax.set_ylabel(y_label)
	#ax.set_ylim()
	ax.set_yticks(np.linspace(y_min, y_max, num_ticks))
	
	cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, location='top')
	cbar.set_label(cbar_label)
	
	if grid:
		plt.grid(True, alpha=0.3, linestyle='--')
	
	plt.tight_layout()
	
	if save and savepath is not None:
		plt.savefig(f'{savepath}imshow.pdf', dpi=30, bbox_inches='tight')
	
	return fig, ax


def project_line(points, 
				plane, axis, fixed_coord, 
				bounds=None, thickness=None):
	
	axis_dict = {'x': 0, 'y': 1, 'z': 2}
	plane_axes = [axis for axis in plane]
	plane_indices = [axis_dict[axis] for axis in plane_axes]
	
	# Get projected axis and other axis
	proj_idx = axis_dict[axis]
	other_axis = plane.replace(axis, '')
	other_idx = axis_dict[other_axis]
	
	# If bounds is none, take whole domain
	if bounds is None:
		min_pos = np.min(points[:, plane_indices], axis=0)
		max_pos = np.max(points[:, plane_indices], axis=0)

		bounds = (min_pos[0], min_pos[1], max_pos[0], max_pos[1])
	
	# Appply first mask (2D -> 2D)
	bounds_mask = np.ones(len(points), dtype=bool)
	
	if plane == 'xy':
		xmin, ymin, xmax, ymax = bounds
		bounds_mask &= (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
		bounds_mask &= (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
	elif plane == 'xz':
		xmin, zmin, xmax, zmax = bounds
		bounds_mask &= (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
		bounds_mask &= (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
	elif plane == 'yz':
		ymin, zmin, ymax, zmax = bounds
		bounds_mask &= (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
		bounds_mask &= (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
	
	filtered_points = points[bounds_mask]
	
	# Apply second mask (2D -> 1D)
	line_mask = np.abs(filtered_points[:, proj_idx] - fixed_coord) <= thickness
	
	projected_points = filtered_points.copy()
	projected_points[:, proj_idx] = fixed_coord
	
	# Trouver les indices originaux des points sélectionnés
	final_mask = np.zeros(len(points), dtype=bool)
	filtered_indices = np.where(bounds_mask)[0]
	line_indices = filtered_indices[line_mask]
	final_mask[line_indices] = True

	# Sort points along desired axis
	sort_indices = np.argsort(projected_points[line_mask][:, other_idx])
	sorted_points = projected_points[line_mask][sort_indices]
	
	sorted_indices = line_indices[sort_indices]
	sorted_mask = np.zeros(len(points), dtype=bool)
	sorted_mask[sorted_indices] = True
	
	return sorted_points, sorted_mask


def project_surface(points, 
					axis, fixed_coord, 
					bounds=None, thickness=0.1):
	
	axis_dict = {'x': 0, 'y': 1, 'z': 2}
	
	# If bounds is none, take whole domain
	if bounds is None:
		min_pos = np.min(points, axis=0)
		max_pos = np.max(points, axis=0)
		xmin, ymin, zmin = min_pos
		xmax, ymax, zmax = max_pos
	else:
		xmin, ymin, zmin, xmax, ymax, zmax = bounds

	# Apply first mask (3D -> 3D)
	bounds_mask = np.ones(len(points), dtype=bool)
	bounds_mask &= (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
	bounds_mask &= (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
	bounds_mask &= (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
	filtered_points = points[bounds_mask]
	
	# Apply second mask (3D -> 2D)
	plane_mask = np.abs(filtered_points[:, axis_dict[axis]] - fixed_coord) <= thickness
	
	projected_points = filtered_points.copy()
	projected_points[:, axis_dict[axis]] = fixed_coord
	
	final_mask = np.zeros(len(points), dtype=bool)
	filtered_indices = np.where(bounds_mask)[0]
	plane_indices = filtered_indices[plane_mask]
	final_mask[plane_indices] = True
	
	return projected_points[plane_mask], final_mask


def get_single_slice(vtk_files, attribute, 
					plane="xy", axis='x', fixed_coord=None, thickness=0.1, component=0, trans_val=None, mean=False,
					plot=False, save=False, savepath=None):

	axis_dict = {'x': 0, 'y': 1, 'z': 2}
	plane_axes = [a for a in plane]
	target_idx = axis_dict[axis]
	
	# Determine perpendicular axis
	if plane == 'xy':
		other_axis = 'y' if axis == 'x' else 'x'
	elif plane == 'xz':
		other_axis = 'z' if axis == 'x' else 'x'
	elif plane == 'yz':
		other_axis = 'z' if axis == 'y' else 'y'
	
	other_axis_idx = axis_dict[other_axis]
	min_coords = np.min(vtk_files[0].points, axis=0)
	max_coords = np.max(vtk_files[0].points, axis=0)
	
	if fixed_coord is None:
		target_min = min_coords[target_idx]
		target_max = max_coords[target_idx]
	else:
		target_min = fixed_coord - thickness
		target_max = fixed_coord + thickness

	# Iterate over all vtk files
	slice_result = []
	for idx, vtk_file in enumerate(vtk_files):

		points = vtk_file.points
		
		# Create slice depending on plane and projected axis			
		if plane == 'xy':
			if axis == 'x':
				if trans_val is None:
					bounds = (target_min, min_coords[1], target_max, max_coords[1])
				else:
					bounds = (target_min, trans_val[0], target_max, trans_val[1])
			else:  # axis == 'y'
				if trans_val is None:
					bounds = (min_coords[0], target_min, max_coords[0], target_max)
				else:
					bounds = (trans_val[0], target_min, trans_val[1], target_max)
		elif plane == 'xz':
			if axis == 'x':
				if trans_val is None:
					bounds = (target_min, min_coords[2], target_max, max_coords[2])
				else:
					bounds = (target_min, trans_val[0], target_max, trans_val[1])
			else:  # axis == 'z'
				if trans_val is None:
					bounds = (min_coords[0], target_min, max_coords[0], target_max)
				else:
					bounds = (trans_val[0], target_min, trans_val[1], target_max)

		elif plane == 'yz':
			if axis == 'y':
				if trans_val is None:
					bounds = (target_min, min_coords[2], target_max, max_coords[2])
				else:
					bounds = (target_min, trans_val[0], target_max, trans_val[1])
			else:  # axis == 'z'
				if trans_val is None:
					bounds = (min_coords[1], target_min, max_coords[1], target_max)
				else:
					bounds = (trans_val[0], target_min, trans_val[1], target_max)
		
		_, mask = project_line(
			points=points,
			plane=plane,
			axis=axis,
			fixed_coord=fixed_coord,
			bounds=bounds,
			thickness=thickness
		)

		if np.any(mask):

			slice_points = points[mask]
			attr_values = vtk_file.point_data[attribute][mask]
			time_values = vtk_file.point_data['time'][mask]
			other_axis_coords = slice_points[:, other_axis_idx]

			# Sort by increasing order for transversal axis
			sort_idx = np.argsort(other_axis_coords)
			other_axis_coords = other_axis_coords[sort_idx]
			attr_values = attr_values[sort_idx]
			
			if component is not None and attr_values.ndim > 1:
				attr_values = attr_values[:, component]

			if mean:
				attr_values = np.mean(attr_values)
			
			slice_result.append({
				'position': fixed_coord,
				'values': attr_values,
				'time': time_values,
				'coordinates': other_axis_coords,
				'name': attribute
			})
	
	if plot:
		plt.figure()
		if mean:
			store_val = []
			time = []
			name, dim = get_name(slice_result[0])
			
			for slice in slice_result:
				store_val.append(np.mean(slice['values']))
				time.append(slice['time'][0])

			time = np.asarray(time)
			store_val = np.asarray(store_val)

			plt.plot(0.8*time, 0.8*store_val/slice_result[0]['values'], 'b-o')
			
			plt.ylabel(fr'{name}(t)/m$_0$ [-]')
			plt.xlabel(f'Time [t]')
			plt.grid()
			#plt.legend()
			plt.tight_layout()

			if save and savepath is not None:
				plt.savefig(f'{savepath}_mean_single.pdf', dpi=30)
		else:
			name, dim = get_name(slice_result[0])
			for slice in slice_result:
				plt.scatter(slice['coordinates'], slice['values'], s=5)
				plt.ylabel(f'{name}(t) [{dim}]')
				plt.xlabel(f'Time [t]')
				plt.grid()
			plt.tight_layout()


			if save and savepath is not None:
				plt.savefig(f'{savepath}_single.pdf', dpi=30)

	return slice_result

def find_bounds(pos, thickness, trans_val, min_coords, max_coords, plane, axis):
	if plane == 'xy':
		if axis == 'x':
			if trans_val is None:
				bounds = (pos - thickness, min_coords[1], pos + thickness, max_coords[1])
			else:
				bounds = (pos - thickness, trans_val[0], pos + thickness, trans_val[1])
		else:  # axis == 'y'
			if trans_val is None:
				bounds = (min_coords[0], pos - thickness, max_coords[0], pos + thickness)
			else:
				bounds = (trans_val[0], pos - thickness, trans_val[1], pos + thickness)
	elif plane == 'xz':
		if axis == 'x':
			if trans_val is None:
				bounds = (pos - thickness, min_coords[2], pos + thickness, max_coords[2])
			else:
				bounds = (pos - thickness, trans_val[0], pos + thickness, trans_val[1])
		else:  # axis == 'z'
			if trans_val is None:
				bounds = (min_coords[0], pos - thickness, max_coords[0], pos + thickness)
			else:
				bounds = (trans_val[0], pos - thickness, trans_val[1], pos + thickness)

	elif plane == 'yz':
		if axis == 'y':
			if trans_val is None:
				bounds = (pos - thickness, min_coords[2], pos + thickness, max_coords[2])
			else:
				bounds = (pos - thickness, trans_val[0], pos + thickness, trans_val[1])
		else:  # axis == 'z'
			if trans_val is None:
				bounds = (min_coords[1], pos - thickness, max_coords[1], pos + thickness)
			else:
				bounds = (trans_val[0], pos - thickness, trans_val[1], pos + thickness)

	return bounds


def get_multiple_slices(vtk_files, attribute, mean=False, global_mask=None,
						plane='xy', axis='x', along=None, thickness=0.1, trans_val=None, component=0, 
						plot=False, save=False, savepath=None):

	axis_dict = {'x': 0, 'y': 1, 'z': 2}
	plane_axes = [a for a in plane]
	target_idx = axis_dict[axis]
	
	# Determine perpendicular axis
	if plane == 'xy':
		other_axis = 'y' if axis == 'x' else 'x'
	elif plane == 'xz':
		other_axis = 'z' if axis == 'x' else 'x'
	elif plane == 'yz':
		other_axis = 'z' if axis == 'y' else 'y'
	
	other_axis_idx = axis_dict[other_axis]
	
	target_min = along[0]
	target_max = along[1]
	
	positions = np.arange(target_min + thickness, target_max - thickness, 2*thickness)
	
	all_slice = []
	
	for vtk in vtk_files:

		if global_mask is not None:
			# Apply global mask to vtk points and attributes
			points = vtk.points[global_mask]
			attribute_values = vtk.point_data[attribute][global_mask]

		points = vtk.points
		min_coords = np.min(points, axis=0)
		max_coords = np.max(points, axis=0)

		slice_result = []
		for pos in positions:
		
			bounds = find_bounds(pos, thickness, trans_val, min_coords, max_coords, plane, axis)	
			
			_, mask = project_line(
				points=points,
				plane=plane,
				axis=axis,
				fixed_coord=pos,
				bounds=bounds,
				thickness=thickness
			)
			
			if np.any(mask):

				slice_points = points[mask]
				attr_values = vtk.point_data[attribute][mask]
				other_axis_coords = slice_points[:, other_axis_idx]

				# Sort by increasing order for transversal axis
				sort_idx = np.argsort(other_axis_coords)
				other_axis_coords = other_axis_coords[sort_idx]
				attr_values = attr_values[sort_idx]
				
				if component is not None and attr_values.ndim > 1:
					attr_values = attr_values[:, component]

				if mean:
					attr_values = np.mean(attr_values)
				
				slice_result.append({
					'position': pos,
					'values': attr_values,
					'coordinates': other_axis_coords,
					'name': attribute,
					'dt':vtk['dt'][0],
					'time':vtk['time'][0]
				})

	
		all_slice.append(slice_result)
			
	return all_slice

def compute_E_tot(vtk_files, mass_slices, u_slices, v_slices, g=9.81, plot=False, save=False, savepath=None):

	# E_tot = E_cin + E_pot + E_in = m*v^2/2 + m*g*h + P*m/rho (friction ignored)
	E_tot = [np.sum(vtk.point_data['mass'] * (0.5 * np.sum(vtk.point_data['velocity']**2, axis=1) + g * vtk.points[:, 1])) for vtk in vtk_files]
	time = [vtk.point_data['time'][0] for vtk in vtk_files]

	dt = np.diff(time)
	dE = np.diff(E_tot)

	last_vtk = vtk_files[-1]
	E_tot_slice = []
	span = []
	# Local E_tot (per slice of the domain) 
	for mass, u, v in zip(mass_slices, u_slices, v_slices):
		E_tot_slice.append(np.sum(mass['values'] * (0.5 * (u['values']**2 + v['values']**2) + g * mass['coordinates'])))
		span.append(7+u['position'])

	plt.figure(figsize=(12,6))
	plt.bar(span, E_tot_slice, alpha=0.7, color='royalblue', width=span[1]-span[0])
	plt.xlabel('Position x [m]')
	plt.ylabel(r'Total energy $E_{tot}$ [J]')

	plt.xlim(7,15)
	plt.xticks(np.arange(7, 15+1, 1))
	plt.ylim(0, 80,)
	plt.yticks(np.arange(0, 80+10, 10))
	plt.grid(True, linestyle='--', alpha=0.4)

	plt.tight_layout()
	plt.savefig(f'{savepath}E_tot_slice.pdf', bbox_inches="tight", dpi=30)



	if plot:

		# Plot E(t)
		fig, ax = plt.subplots(figsize=(12, 6))
		ax.plot(time, E_tot, 'b-o')
		ax.set_xlabel('Time t [s]')
		ax.set_ylabel(r'Total energy $E_{tot}$ [J]')

		'''
		ax.set_xlim((0, 80))
		ax.set_xticks(np.arange(0, 81, 10))
		ax.set_ylim(0, 6000)
		ax.set_yticks(np.arange(0, 6000+1000, 1000))
		'''
		ax.grid(True, linestyle='--', alpha=0.5)
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}E_tot.pdf', bbox_inches="tight", dpi=30)
	

		# Plot dE(t)/dt
		fig, ax = plt.subplots(figsize=(12, 6))
		plt.plot(time[:-1], dE, 'b-o')
		
		ax.set_xlabel('Time t [s]')
		ax.set_ylabel(r'Generated power $dE_{tot}/dt$ [W]')
		'''
		ax.set_xlim((0, 80))
		ax.set_xticks(np.arange(0, 81, 10))
		ax.set_ylim(0, 500)
		ax.set_yticks(np.arange(-50, 500+50, 100))
		'''
		ax.grid(True, linestyle='--', alpha=0.5)

		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}dE_tot.pdf', bbox_inches="tight", dpi=30)

	return E_tot, time, dE, dt



	

def spatial_derivative__(slices, chosen_axis, plot=False, save=False, savepath=None):

	axis_dict = {'x':0, 'y':1, 'z':2}
	axis = axis_dict[chosen_axis]

	mean_vals = []
	pos = []

	for slice in slices:
		mean_vals.append(np.mean(slice['values']))
		pos.append(slice['position'])

	mean_vals = np.array(mean_vals)
	positions = np.array(pos)
	
	# Finite difference method
	derivative = np.zeros_like(mean_vals)
	if len(positions) > 2:
		derivative[1:-1] = (mean_vals[2:] - mean_vals[:-2]) / (positions[2:] - positions[:-2]) # centrered inside domain
		derivative[0] = (mean_vals[1] - mean_vals[0]) / (positions[1] - positions[0]) # forward at entrance
		derivative[-1] = (mean_vals[-1] - mean_vals[-2]) / (positions[-1] - positions[-2]) # backward at exit

	if plot:

		name, dim = get_name(slices[0])
		plt.figure()
		plt.plot(positions, derivative, 'b-o')
		plt.axhline(y=0, color='k', linestyle='--') 
		plt.xlabel(f'Position {axis} [m]')
		plt.ylabel(r'$\frac{d}{dx}'+f'${name} {dim}/m')
		plt.grid(True)
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}_dx.pdf', dpi=30)
		
	return derivative, positions

def time_derivative(slices, plot=False, save=False, savepath=None):

	mean_vals = []
	time = []

	for slice in slices:
		mean_vals.append(np.mean(slice['values']))
		time.append(slice['time'][0])

	mean_vals = np.array(mean_vals)
	times = np.array(time)

	lower_percentile = np.percentile(mean_vals, 0)
	upper_percentile = np.percentile(mean_vals, 98)

	# Créer un masque pour garder seulement les valeurs dans la plage
	mask = (mean_vals >= lower_percentile) & (mean_vals <= upper_percentile)

	# Appliquer le masque aux deux arrays
	mean_vals = mean_vals[mask]
	times = times[mask]
	
	# Forward finite difference method
	derivative = np.zeros_like(mean_vals)
	derivative[1:] = (mean_vals[1:] - mean_vals[:-1])/(times[1:] - times[:-1]) # derivative[1:] = (mean_vals[:-1] - mean_vals[:1])/(times[:-1] - times[:1]) to compare to init

	if plot:

		name, dim = get_name(slices[0])

		plt.figure()
		plt.plot(times, mean_vals/mean_vals[-1], 'b-o', alpha=0.8)

		plt.xlabel('Time t [s]')
		plt.xlim(0,15)
		plt.xticks(np.arange(0,15+2.5,2.5))
		
		plt.ylabel(fr'{name}(t)/$m_f$ [{dim}]')
		plt.ylim(0.8, 1)
		plt.yticks(np.arange(0.8,1+0.1,0.1))

		plt.grid(True, alpha=0.3, ls='--')
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}_over_time.pdf', dpi=30, bbox_inches='tight')


		plt.figure()
		plt.plot(times[1:], derivative[1:], 'b-o', alpha=0.8)
		plt.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='Zero line') 


		plt.xlabel('Time t [s]')
		plt.xlim(0,15)
		plt.xticks(np.arange(0,15+2.5,2.5))
		
		plt.ylabel(fr'$\partial${name}/$\partial$t [{dim}/s]')
		plt.ylim(-50, 550)
		plt.yticks(np.arange(-50,550+100,100))

		
		plt.grid(True, alpha=0.3, ls='--')
		plt.legend()
		plt.tight_layout()

		if save and savepath is not None:
			pass
			#plt.savefig(f'{savepath}_dt.pdf', dpi=30, bbox_inches='tight')
		
	return derivative, times
	
def plot_vtk(vtk_file, mask=None, attribute='velocity', save=False, savepath=None, is_plt=False):

	if mask is not None:
		points = vtk_file.points[mask]
		attribute_values = vtk_file.point_data[attribute][mask]
	else:
		points = vtk_file.points
		attribute_values = vtk_file.point_data[attribute]
	
	# Vérifier si les attributs sont vectoriels et calculer la magnitude si nécessaire
	if len(attribute_values.shape) > 1 and attribute_values.shape[1] > 1:
		magnitude = np.sqrt(np.sum(attribute_values**2, axis=1))
	else:
		magnitude = attribute_values
	
	# Choisir le type de visualisation
	if not is_plt:
		# Utiliser PyVista (code original amélioré)
		point_cloud = pv.PolyData(points)
		
		if len(attribute_values.shape) > 1 and attribute_values.shape[1] > 1:
			# Si c'est un attribut vectoriel, ajouter aussi la magnitude
			point_cloud.point_data[attribute] = attribute_values
			point_cloud.point_data["magnitude"] = magnitude
			scalars = "magnitude"
		else:
			point_cloud.point_data[attribute] = attribute_values
			scalars = attribute
		
		plotter = pv.Plotter()
		plotter.add_mesh(
			point_cloud, 
			render_points_as_spheres=True, 
			scalar_bar_args={"title": f"{attribute}"},
			cmap="viridis", 
			scalars=scalars
		)
		
		# Ajouter une grille et configurer la vue 2D
		plotter.show_grid()
		plotter.view_xy()  # Vue selon le plan XY
		plotter.enable_parallel_projection()
		
		if save and savepath is not None:
			plotter.screenshot(savepath)

		plotter.show()
		
		return plotter  # Retourner le plotter pour permettre show() plus tard
	
	else:
		# Utiliser Matplotlib
		x = points[:, 0] 
		y = points[:, 1]
		
		# Créer la figure et l'axe
		fig, ax = plt.subplots(figsize=(12,6))
		
		# Tracer les points avec une coloration selon l'attribut
		norm = Normalize()
		scatter = ax.scatter(x, y, c=magnitude, cmap='viridis', s=10, alpha=0.8, vmin=0, vmax=8)
		
		# Ajouter une barre de couleur
		cbar = plt.colorbar(scatter, ax=ax)
		cbar.set_label('Velocity magnitude [m/s]')

		#z = parabole(np.linspace(8, 12, 100))
		#plt.fill_between(np.linspace(8, 12, 100), np.zeros_like(z), z, color='grey')
		#plt.plot(np.linspace(8, 12, 100), z, color='black', linewidth=5)
		
		# Ajouter une grille
		ax.grid(True, linestyle='--', alpha=0.3)
		
		# Configurer les axes
		ax.set_xlabel('Distance x [m]')
		ax.set_xlim(0,45)
		ax.set_xticks(np.linspace(0, 45, 9))
		

		ax.set_ylabel('Diameter y [m]')
		ax.set_ylim(-1.615, 1.615)
		ax.set_yticks(np.linspace(-1.615, 1.615, 6))
		

		ax.set_aspect('auto')
		
		if save and savepath is not None:
			plt.savefig(f'{savepath}.pdf', dpi=15, bbox_inches='tight')
		
		plt.tight_layout()
		return fig, ax  # Retourner figure et axes pour personnalisation ultérieure

def compute_flow_rate(Q_v_th, rho0, 
					u_slices, rho_slices, 
					plot=False, save=False, savepath=None):

	Q_v, Q_m, span = [], [], []

	# Integrate Q_v and Q_m along transversal axis
	for u, rho in zip(u_slices, rho_slices):

		u_vals = np.asarray(u['values'])
		rho_vals = np.asarray(rho['values'])
		y_vals = np.asarray(u['coordinates'])

		span.append(u['position'])
		Q_v.append(np.trapezoid(u_vals, x=y_vals))
		Q_m.append(np.trapezoid(u_vals*rho_vals, x=y_vals))

	Q_v = np.asarray(Q_v)
	Q_m = np.asarray(Q_m)
	span = np.asarray(span)

	# Calculate error
	Q_m_th = Q_v_th*rho0
	Q_v_mean = np.mean(Q_v)
	Q_m_mean = np.mean(Q_m)
	error_Q_v = (Q_v_th - Q_v_mean) / Q_v_th
	error_Q_m = (Q_m_th - Q_m_mean) / Q_m_th

	print(f'Error on flow rate: {100*error_Q_v}%')
	print(f'Error on masss flow: {100*error_Q_v}%')

	if plot:

		# Flow rate
		plt.figure()
		plt.bar(span, Q_v, alpha=0.7, color='royalblue', width=span[1] - span[0])
		plt.hlines(Q_v_th, span[0], span[-1], color='blue', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_v_th:.2f} [m$^2$/s]')
		plt.hlines(Q_v_mean, span[0], span[-1], color='navy', linestyle='-', linewidth=2,
				label=fr'Mean Q: {Q_v_mean:.2f} [m$^2$/s]')
		
		plt.xlabel('Position x [m]')
		#plt.xlim(0, 45)
		#plt.xticks(np.arange(0, 45+5, 5))

		plt.ylabel(r'Flow rate [m$^2$/s]')
		#plt.ylim(0, 22)
		#plt.yticks(np.linspace(0, 22, 5))


		plt.grid(True, alpha=0.3, ls="--")
		plt.legend(loc='center')
		plt.tight_layout()
		if save and savepath is not None:
			plt.savefig(f'{savepath}flow_rate.pdf', dpi=30, bbox_inches='tight')

		# Mass flow
		plt.figure()
		plt.bar(span, Q_m, alpha=0.7, color='green', width=span[1] - span[0])
		plt.hlines(Q_m_th, span[0], span[-1], color='blue', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_m_th:.2f} [Kg/s]')
		plt.hlines(Q_m_mean, span[0], span[-1], color='navy', linestyle='-', linewidth=2,
				label=fr'Mean Q: {Q_m_mean:.2f} [Kg/s]')
		
		plt.xlabel(r'Position x [m]')
		plt.xlim(0, 45)
		plt.xticks(np.arange(0, 45+5, 5))

		plt.ylabel(r'Mass flow [Kg/s]')
		plt.ylim(0, 22000)
		plt.yticks(np.linspace(0, 22000, 5))
		plt.grid(True, alpha=0.3, ls="--")
		plt.legend(loc='center')
		plt.tight_layout()
		if save and savepath is not None:
			plt.savefig(f'{savepath}mass_flow.pdf', dpi=30, bbox_inches='tight')


	return Q_v, Q_m


def analyze_particle_distribution(vtk_data, x_slice, delta_x=0.1, n_bins=50, plot=False, save=False, savepath=None):
	"""
	Analyze the particle distribution in a slice of the domain.
	
	Args:
		vtk_data: VTK data containing particle information
		x_slice (float): x-position of the slice
		delta_x (float): Half-width of the slice
		n_bins (int): Number of bins for histogram
		plot (bool): If True, generates a plot
	
	Returns:
		tuple: (Slice points, (high density regions, low density regions, bin centers))
	"""
	points = vtk_data.points
	x_pos = points[:, 0]
	# Slice between [x - dx/2, x + dx/2]
	mask = (x_pos >= x_slice - delta_x/2) & (x_pos <= x_slice + delta_x/2)
	slice_points = points[mask]


	if len(slice_points) == 0:
		print(f"No particles found at x = {x_slice} +- {delta_x/2}")
		return None, (None, None, None)
	
	# Y domain (for bins)
	y_pos = slice_points[:, 1]
	y_min, y_max = np.min(y_pos), np.max(y_pos)
	y_range = y_max - y_min
	bins = np.linspace(y_min - 0.05*y_range, y_max + 0.05*y_range, n_bins)
	
	# Histogram
	counts, bin_edges = np.histogram(y_pos, bins=bins)
	bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	
	# Find potential low/high density regions
	bin_width = bin_edges[1] - bin_edges[0]
	density_per_bin = counts / bin_width
	mean_density = np.mean(density_per_bin)
	high_density_threshold = 1.5 * mean_density
	low_density_threshold = 0.5 * mean_density
	
	high_density_regions = bin_centers[density_per_bin > high_density_threshold]
	low_density_regions = bin_centers[density_per_bin < low_density_threshold]
	
	if plot:
		configure_latex()
		fig, ax = plt.subplots(figsize=(12,6))
		ax.bar(bin_centers, counts, width=bin_width*0.9, alpha=0.7, color='blue')
		
		ax.set_xlabel('Diameter y [m]')
		ax.set_xlim(-2, 2)
		ax.set_xticks(np.arange(-2, 2+0.5, 0.5))
		ax.set_ylim(0, 8)
		ax.set_yticks(np.arange(0, 10, 2))
		ax.set_ylabel('Number of particles [-]')
		ax.grid(True, alpha=0.4, ls='--')
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}/particle_distribution.pdf', dpi=30)
	
	return slice_points, (high_density_regions, low_density_regions, bin_centers)


def plot_particles_with_selection_rectangle(vtk_files, plane='xy', axis='x', 
										fixed_coord=None, thickness=0.1, 
										trans_val=None, save=False, savepath=None):
	
	# Utiliser le dernier fichier VTK
	vtk_file = vtk_files[-1]
	points = vtk_file.points
	
	# Dictionnaire des axes pour l'indexation
	axis_dict = {'x': 0, 'y': 1, 'z': 2}
	target_idx = axis_dict[axis]
	
	# Obtenir les coordonnées min et max
	min_coords = np.min(points, axis=0)
	max_coords = np.max(points, axis=0)
	
	# Définir la coordonnée fixe si non fournie
	if fixed_coord is None:
		fixed_coord = (min_coords[target_idx] + max_coords[target_idx]) / 2
	
	# Définir les limites pour la tranche, suivant la logique de get_single_slice
	if plane == 'xy':
		plot_idx1, plot_idx2 = 0, 1
		xlabel, ylabel = 'X [m]', 'Y [m]'
		if axis == 'x':
			if trans_val is None:
				bounds = (fixed_coord - thickness, min_coords[1], fixed_coord + thickness, max_coords[1])
			else:
				bounds = (fixed_coord - thickness, trans_val[0], fixed_coord + thickness, trans_val[1])
		else:  # axis == 'y'
			if trans_val is None:
				bounds = (min_coords[0], fixed_coord - thickness, max_coords[0], fixed_coord + thickness)
			else:
				bounds = (trans_val[0], fixed_coord - thickness, trans_val[1], fixed_coord + thickness)
	elif plane == 'xz':
		plot_idx1, plot_idx2 = 0, 2
		xlabel, ylabel = 'X [m]', 'Z [m]'
		if axis == 'x':
			if trans_val is None:
				bounds = (fixed_coord - thickness, min_coords[2], fixed_coord + thickness, max_coords[2])
			else:
				bounds = (fixed_coord - thickness, trans_val[0], fixed_coord + thickness, trans_val[1])
		else:  # axis == 'z'
			if trans_val is None:
				bounds = (min_coords[0], fixed_coord - thickness, max_coords[0], fixed_coord + thickness)
			else:
				bounds = (trans_val[0], fixed_coord - thickness, trans_val[1], fixed_coord + thickness)
	elif plane == 'yz':
		plot_idx1, plot_idx2 = 1, 2
		xlabel, ylabel = 'Y [m]', 'Z [m]'
		if axis == 'y':
			if trans_val is None:
				bounds = (fixed_coord - thickness, min_coords[2], fixed_coord + thickness, max_coords[2])
			else:
				bounds = (fixed_coord - thickness, trans_val[0], fixed_coord + thickness, trans_val[1])
		else:  # axis == 'z'
			if trans_val is None:
				bounds = (min_coords[1], fixed_coord - thickness, max_coords[1], fixed_coord + thickness)
			else:
				bounds = (trans_val[0], fixed_coord - thickness, trans_val[1], fixed_coord + thickness)
	
	# Récupérer le masque pour les points dans la tranche en utilisant project_line
	_, mask = project_line(
		points=points,
		plane=plane,
		axis=axis,
		fixed_coord=fixed_coord,
		bounds=bounds,
		thickness=thickness
	)
	
	fig, ax = plt.subplots(figsize=(12, 6))
	
	
	ax.scatter(points[~mask, plot_idx1], points[~mask, plot_idx2], 
			c='blue', s=5, alpha=0.5, label='Not selected')
	
	ax.scatter(points[mask, plot_idx1], points[mask, plot_idx2], 
			c='red', s=5, alpha=0.7, label='Selected')
	
	xmin, ymin, xmax, ymax = bounds
	
	rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
						linewidth=2, edgecolor='green', facecolor='none',
						label='Slice bounds')
	ax.add_patch(rect)
	
	ax.set_xlabel(r'${}$'.format(xlabel))
	ax.set_ylabel(r'${}$'.format(ylabel))
	ax.grid(True, alpha=0.4, ls='--')
	ax.legend(loc='upper right')
	
	# Définir les limites pour montrer tout le domaine avec un peu de marge
	ax.set_xlim(14.5, 15.5)
	ax.set_xticks(np.arange(14.5, 15.5+0.25, 0.25))
	ax.set_ylim(-1.65, 1.65)
	ax.set_yticks(np.linspace(-1.65, 1.65, 5))
	
	plt.tight_layout()
	
	# Enregistrer la figure si demandé
	if save and savepath is not None:
		plt.savefig(f'{savepath}/framed_particles.pdf', dpi=30, bbox_inches='tight')
	
	plt.show()
	
	return fig

def get_global_bounds(vtk_files, dimensions='3D', bounding=None):

	# Initialize arrays for min and max coordinates
	if dimensions == '3D':
		x_loc_max, y_loc_max, z_loc_max = [], [], []
		x_loc_min, y_loc_min, z_loc_min = [], [], []
	else:
		x_loc_max, y_loc_max = [], []
		x_loc_min, y_loc_min = [], []
	
	if not isinstance(vtk_files, list):
		vtk_files = [vtk_files]
		
	# Si aucune contrainte n'est fournie, initialiser une structure vide
	if bounding is None:
		bounding = {}

	# Find min/max coordinates across all vtk files
	for vtk in vtk_files:
		if dimensions == '3D':
			x_loc_max.append(np.max(vtk.points, axis=0)[0])
			y_loc_max.append(np.max(vtk.points, axis=0)[1])
			z_loc_max.append(np.max(vtk.points, axis=0)[2])
			
			x_loc_min.append(np.min(vtk.points, axis=0)[0])
			y_loc_min.append(np.min(vtk.points, axis=0)[1])
			z_loc_min.append(np.min(vtk.points, axis=0)[2])
		else:  # 2D mode
			x_loc_max.append(np.max(vtk.points, axis=0)[0])
			y_loc_max.append(np.max(vtk.points, axis=0)[1])
			
			x_loc_min.append(np.min(vtk.points, axis=0)[0])
			y_loc_min.append(np.min(vtk.points, axis=0)[1])
	
	# Global min/max
	x_glob_max, y_glob_max = np.max(x_loc_max), np.max(y_loc_max)
	x_glob_min, y_glob_min = np.min(x_loc_min), np.min(y_loc_min)
	
	# Appliquer les contraintes de la bounding box, si spécifiées
	if 'x_min' in bounding and bounding['x_min'] is not None:
		x_glob_min = max(x_glob_min, bounding['x_min'])
	if 'x_max' in bounding and bounding['x_max'] is not None:
		x_glob_max = min(x_glob_max, bounding['x_max'])
	if 'y_min' in bounding and bounding['y_min'] is not None:
		y_glob_min = max(y_glob_min, bounding['y_min'])
	if 'y_max' in bounding and bounding['y_max'] is not None:
		y_glob_max = min(y_glob_max, bounding['y_max'])
	
	if dimensions == '3D':
		z_glob_max = np.max(z_loc_max)
		z_glob_min = np.min(z_loc_min)
		
		# Appliquer les contraintes 3D si spécifiées
		if 'z_min' in bounding and bounding['z_min'] is not None:
			z_glob_min = max(z_glob_min, bounding['z_min'])
		if 'z_max' in bounding and bounding['z_max'] is not None:
			z_glob_max = min(z_glob_max, bounding['z_max'])
		
		return [x_glob_min, y_glob_min, z_glob_min], [x_glob_max, y_glob_max, z_glob_max]
	
	return [x_glob_min, y_glob_min], [x_glob_max, y_glob_max]


def create_grid(vtk_file, bounds, L_cell, dimensions='2D', plane='xy', r=None):
	
	# Check 2D
	if dimensions == '2D' and plane not in ['xy', 'xz', 'yz']:
		raise ValueError("PLane must be 'xy', 'xz', 'yz' (2D)")
	
	# Check format [(x_min, y_min, z_min), (x_max, y_max, z_max)]
	elif isinstance(bounds, list) and len(bounds) == 2:

		min_bounds, max_bounds = bounds

		if dimensions == '3D' and (len(min_bounds) < 3 or len(max_bounds) < 3):
			raise ValueError("3D: bounds must contain x, y and z")
		
		if dimensions == '2D' and len(min_bounds) == 2 and len(max_bounds) == 2:
			# 2D -> 3D bounding to use KDTree later
			min_temp, max_temp = [0, 0, 0], [0, 0, 0]
			
			if plane == 'xy':
				min_temp[0], min_temp[1] = min_bounds
				max_temp[0], max_temp[1] = max_bounds
			elif plane == 'xz':
				min_temp[0], min_temp[2] = min_bounds
				max_temp[0], max_temp[2] = max_bounds
			elif plane == 'yz':
				min_temp[1], min_temp[2] = min_bounds
				max_temp[1], max_temp[2] = max_bounds
			
			min_bounds, max_bounds = min_temp, max_temp
	else:
		raise ValueError("Format bounds warning. Use[(x_min, y_min, z_min), (x_max, y_max, z_max)]")
	
	# Compute nb_elem in each dir
	DOM_x = max_bounds[0] - min_bounds[0]
	DOM_y = max_bounds[1] - min_bounds[1]
	DOM_z = max_bounds[2] - min_bounds[2]
	
	nb_elem_x = int(np.ceil(DOM_x / L_cell))
	nb_elem_y = int(np.ceil(DOM_y / L_cell))
	nb_elem_z = int(np.ceil(DOM_z / L_cell))
	
	# Adjust DOM_size if offset (if nb_elem_i* L_cell > DOM_i)
	max_bounds[0] = min_bounds[0] + nb_elem_x * L_cell
	max_bounds[1] = min_bounds[1] + nb_elem_y * L_cell
	max_bounds[2] = min_bounds[2] + nb_elem_z * L_cell
	
	nb_elems = (nb_elem_x, nb_elem_y, nb_elem_z)
	print(f"Number of cells in (x,y,z) dir: {nb_elems}")
	
	x_span = np.linspace(min_bounds[0], max_bounds[0], nb_elem_x+1)
	y_span = np.linspace(min_bounds[1], max_bounds[1], nb_elem_y+1)
	z_span = np.linspace(min_bounds[2], max_bounds[2], nb_elem_z+1)

	x_min, y_min, z_min = x_span[0], y_span[0], z_span[0]
	x_max, y_max, z_max = x_span[-1], y_span[-1], z_span[-1]
	dx = L_cell
	dy = L_cell
	dz = L_cell
	
	if dimensions == '3D':
		grid_cells = {cell_idx: {
			'particles': [],
		} for cell_idx in np.ndindex((nb_elems[0], nb_elems[1], nb_elems[2]))}
	else:  # 2D
		grid_cells = {cell_idx: {
			'particles': [],
		} for cell_idx in np.ndindex((nb_elems[0], nb_elems[1]))}
	
	# Process all particles at once
	total_particles = 0
	skipped_particles = 0
	
	points = np.array(vtk_file.points)
	total_particles = len(points)
	
	# Filter points within bounds
	if dimensions == '3D':
		mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & 
				(points[:, 1] >= y_min) & (points[:, 1] <= y_max) & 
				(points[:, 2] >= z_min) & (points[:, 2] <= z_max))
	else:  # 2D
		if plane == 'xy':
			mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & 
					(points[:, 1] >= y_min) & (points[:, 1] <= y_max))
		elif plane == 'xz':
			mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & 
					(points[:, 2] >= z_min) & (points[:, 2] <= z_max))
		elif plane == 'yz':
			mask = ((points[:, 1] >= y_min) & (points[:, 1] <= y_max) & 
					(points[:, 2] >= z_min) & (points[:, 2] <= z_max))
	
	valid_points = points[mask]
	valid_indices = np.where(mask)[0]
	skipped_particles = len(points) - len(valid_points)
	
	# Calculate cell indices for all valid points
	if dimensions == '3D':
		i = np.minimum(((valid_points[:, 0] - x_min) / dx).astype(int), nb_elems[0] - 1)
		j = np.minimum(((valid_points[:, 1] - y_min) / dy).astype(int), nb_elems[1] - 1)
		k = np.minimum(((valid_points[:, 2] - z_min) / dz).astype(int), nb_elems[2] - 1)
		
		for idx, (i_val, j_val, k_val) in enumerate(zip(i, j, k)):
			grid_cells[(i_val, j_val, k_val)]['particles'].append(valid_indices[idx])
	
	else:  # 2D
		if plane == 'xy':
			i = np.minimum(((valid_points[:, 0] - x_min) / dx).astype(int), nb_elems[0] - 1)
			j = np.minimum(((valid_points[:, 1] - y_min) / dy).astype(int), nb_elems[1] - 1)
		elif plane == 'xz':
			i = np.minimum(((valid_points[:, 0] - x_min) / dx).astype(int), nb_elems[0] - 1)
			j = np.minimum(((valid_points[:, 2] - z_min) / dz).astype(int), nb_elems[2] - 1)
		elif plane == 'yz':
			i = np.minimum(((valid_points[:, 1] - y_min) / dy).astype(int), nb_elems[1] - 1)
			j = np.minimum(((valid_points[:, 2] - z_min) / dz).astype(int), nb_elems[2] - 1)
		
		for idx, (i_val, j_val) in enumerate(zip(i, j)):
			grid_cells[(i_val, j_val)]['particles'].append(valid_indices[idx])
	
	print(f"Total number of particles: {total_particles}")
	print(f"Skipped particles (out of bounds): {skipped_particles} ({skipped_particles/total_particles*100:.2f}%)")
	
	# Create grid object with all properties
	grid = {
		'nb_elems': nb_elems, 'cells': grid_cells, 
		'dimensions': dimensions,
		'total_particles': total_particles,
		'included_particles': total_particles - skipped_particles,
		'skipped_particles': skipped_particles,
		'L_cell':L_cell
	}
	
	if dimensions == '3D':
		xc = x_span[:-1] + L_cell/2
		yc = y_span[:-1] + L_cell/2
		zc = z_span[:-1] + L_cell/2
		X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')
		cell_centers = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
		shape = grid['nb_elems']

	else:  # 2D
		grid['plane'] = plane
		if plane == 'xy':
			xc = x_span[:-1] + L_cell/2
			yc = y_span[:-1] + L_cell/2
			X, Y = np.meshgrid(xc, yc, indexing='ij')
			cell_centers = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z_min)]
			shape = grid['nb_elems'][:2]
		elif plane == 'xz':
			xc = x_span[:-1] + L_cell/2
			zc = z_span[:-1] + L_cell/2
			X, Z = np.meshgrid(xc, zc, indexing='ij')
			cell_centers = np.c_[X.ravel(), np.full(X.size, y_min), Z.ravel()]
			shape = (grid['nb_elems'][0], grid['nb_elems'][2])
		elif plane == 'yz':
			yc = y_span[:-1] + L_cell/2
			zc = z_span[:-1] + L_cell/2
			Y, Z = np.meshgrid(yc, zc, indexing='ij')
			cell_centers = np.c_[np.full(Y.size, x_min), Y.ravel(), Z.ravel()]
			shape = (grid['nb_elems'][1], grid['nb_elems'][2])

	grid['cell_centers'] = cell_centers
	grid['cell_shape'] = shape

	return grid

import itertools

def compute_grid_values(grid, vtk_files, attribute_name, h, W, component=None):
	import numpy as np
	from scipy.spatial import cKDTree
	import itertools

	# Gestion des fichiers VTK (liste ou fichier unique)
	if not isinstance(vtk_files, list):
		vtk_files = [vtk_files]

	# Initialisation des accumulateurs
	if 'time_data' not in grid:
		grid['time_data'] = {
			'values_sum': None,
			'count': np.zeros(grid['cell_shape'], dtype=int)
		}

	for vtk_file in vtk_files:
		# Calcul des valeurs instantanées
		result = compute_instant_values(grid, vtk_file, attribute_name, h, W, component)
		
		# Mise à jour des accumulateurs
		valid_mask = ~np.isnan(result)
		
		# Initialisation de la somme selon la forme des données
		if grid['time_data']['values_sum'] is None:
			grid['time_data']['values_sum'] = np.zeros_like(result)
		
		# Remplacement des NaN par 0 pour l'accumulation
		result_to_add = np.where(valid_mask, result, 0.0)
		grid['time_data']['values_sum'] += result_to_add
		
		# Mise à jour du compteur pour les cellules valides
		if result.ndim > len(grid['cell_shape']):  # Cas vectoriel
			grid['time_data']['count'] += np.any(valid_mask, axis=-1)
		else:  # Cas scalaire
			grid['time_data']['count'] += valid_mask.astype(int)

	# Calcul de la moyenne temporelle
	if grid['time_data']['values_sum'] is not None:
		divisor = grid['time_data']['count']
		# Gestion de la forme pour les vecteurs
		if grid['time_data']['values_sum'].ndim > len(grid['cell_shape']):
			divisor = divisor[..., np.newaxis]
		
		# Calcul avec protection division par zéro
		grid['cell_values'] = np.divide(
			grid['time_data']['values_sum'], 
			divisor, 
			where=divisor > 0,
			out=np.full_like(grid['time_data']['values_sum'], np.nan))
	else:
		grid['cell_values'] = np.full(grid['cell_shape'], np.nan)

	return grid

def compute_instant_values(grid, vtk_file, attribute_name, h, W, component=None):
	dimensions = grid['dimensions']
	plane = grid.get('plane', None)
	dim = 3 if dimensions == '3D' else 2
	shape = grid['cell_shape']
	
	# Collecte des particules de la grille
	particle_indices = list(itertools.chain.from_iterable(
		cell['particles'] for cell in grid['cells'].values()
	))
	if not particle_indices:
		grid['cell_values'] = np.full(shape, np.nan)
		return grid
	
	all_positions = vtk_file.points[particle_indices]
	all_attributes = vtk_file.point_data[attribute_name][particle_indices]
	
	if component is not None:
		if isinstance(all_attributes, np.ndarray) and all_attributes.ndim > 1:
			all_attributes = all_attributes[:, component]
		else:
			all_attributes = all_attributes
	
	# Find neighbours of cells centers (KDTree)
	tree = cKDTree(all_positions)
	cell_centers = grid['cell_centers']
	indices = tree.query_ball_point(cell_centers, h) # 
	
	# Flatten neighbours idx
	cell_idx = []
	particle_idx = []
	for i, neighbors in enumerate(indices):
		cell_idx.extend([i] * len(neighbors))
		particle_idx.extend(neighbors)
	if not particle_idx:
		grid['cell_values'] = np.full(shape, np.nan)
		return grid
	
	cell_idx = np.array(cell_idx)
	particle_idx = np.array(particle_idx)
	
	# Pre compute valid neighbouring particles distance 
	displacements = cell_centers[cell_idx] - all_positions[particle_idx]
	dists = np.linalg.norm(displacements, axis=1)
	valid_mask = dists <= h
	cell_idx_valid = cell_idx[valid_mask]
	particle_idx_valid = particle_idx[valid_mask]
	dists_valid = dists[valid_mask]
	
	if len(dists_valid) == 0:
		grid['cell_values'] = np.full(shape, np.nan)
		return grid
	
	# SPH interpolation
	weights_valid = W(dists_valid, h)
	sum_weights = np.bincount(cell_idx_valid, weights=weights_valid, minlength=len(cell_centers))
	valid_cells = sum_weights >= 1e-12
	
	is_vector = all_attributes.ndim > 1 and component is None
	if is_vector:
		D = all_attributes.shape[1]
		weighted_sums = np.zeros((len(cell_centers), D))
		valid_attrs = all_attributes[particle_idx_valid]
		weighted_attrs = valid_attrs * weights_valid[:, np.newaxis]
		np.add.at(weighted_sums, cell_idx_valid, weighted_attrs)
		averages = np.full((len(cell_centers), D), np.nan)
		averages[valid_cells] = weighted_sums[valid_cells] / sum_weights[valid_cells, np.newaxis]
		result = averages.reshape(shape + (D,))
	else:
		weighted_sums = np.bincount(cell_idx_valid, weights=all_attributes[particle_idx_valid] * weights_valid, minlength=len(cell_centers))
		averages = np.full(len(cell_centers), np.nan)
		averages[valid_cells] = weighted_sums[valid_cells] / sum_weights[valid_cells]
		result = averages.reshape(shape)
	
	
	return result

def spatial_derivative(grid, attribute_name, deriv_axis='x', component=None):
	
	dimensions = grid['dimensions']
	cell_values = grid['cell_values']
	dx, dy, dz = grid['L_cell'], grid['L_cell'], grid['L_cell']
	
	if dimensions == '2D':
		if deriv_axis == 'y' and grid.get('plane') == 'xz':
			raise ValueError(f"The derivative axis 'y' is not in the plane '{grid['plane']}'")
		elif deriv_axis == 'z' and grid.get('plane') == 'xy':
			raise ValueError(f"The derivative axis 'z' is not in the plane '{grid['plane']}'")
		elif deriv_axis == 'x' and grid.get('plane') == 'yz':
			raise ValueError(f"The derivative axis 'x' is not in the plane '{grid['plane']}'")
		
		
	if dimensions == '2D':
		if grid['plane'] == 'xy':
			nx, ny = grid['nb_elems'][0], grid['nb_elems'][1]
		elif grid['plane'] == 'xz':
			nx, nz = grid['nb_elems'][0], grid['nb_elems'][2]
		elif grid['plane'] == 'yz':
			ny, nz = grid['nb_elems'][1], grid['nb_elems'][2]
	else:  # 3D
		nx, ny, nz = grid['nb_elems'][0], grid['nb_elems'][1], grid['nb_elems'][2]
	

	derivative = np.zeros_like(cell_values)
	if deriv_axis == 'x':
		if dimensions == '2D':
			if grid['plane'] == 'xy':
				
				# Schéma centré d'ordre 4 pour les points intérieurs
				for i in range(2, nx-2):
					derivative[i, :] = (-cell_values[i+2, :] + 8*cell_values[i+1, :] - 
										8*cell_values[i-1, :] + cell_values[i-2, :]) / (12*dx)
				
				# Pour les points proches des bords
				if nx > 3:
					derivative[1, :] = (cell_values[2, :] - cell_values[0, :]) / (2*dx)
					derivative[nx-2, :] = (cell_values[nx-1, :] - cell_values[nx-3, :]) / (2*dx)
					derivative[0, :] = (-3*cell_values[0, :] + 4*cell_values[1, :] - cell_values[2, :]) / (2*dx)
					derivative[nx-1, :] = (3*cell_values[nx-1, :] - 4*cell_values[nx-2, :] + cell_values[nx-3, :]) / (2*dx)
			
			elif grid['plane'] == 'xz':
				
				for i in range(2, nx-2):
					derivative[i, :] = (-cell_values[i+2, :] + 8*cell_values[i+1, :] - 
										8*cell_values[i-1, :] + cell_values[i-2, :]) / (12*dx)
				
				if nx > 3:
					derivative[1, :] = (cell_values[2, :] - cell_values[0, :]) / (2*dx)
					derivative[nx-2, :] = (cell_values[nx-1, :] - cell_values[nx-3, :]) / (2*dx)
					derivative[0, :] = (-3*cell_values[0, :] + 4*cell_values[1, :] - cell_values[2, :]) / (2*dx)
					derivative[nx-1, :] = (3*cell_values[nx-1, :] - 4*cell_values[nx-2, :] + cell_values[nx-3, :]) / (2*dx)
		
		elif dimensions == '3D':
			
			for i in range(2, nx-2):
				derivative[i, :, :] = (-cell_values[i+2, :, :] + 8*cell_values[i+1, :, :] - 
									8*cell_values[i-1, :, :] + cell_values[i-2, :, :]) / (12*dx)
			
			if nx > 3:
				derivative[1, :, :] = (cell_values[2, :, :] - cell_values[0, :, :]) / (2*dx)
				derivative[nx-2, :, :] = (cell_values[nx-1, :, :] - cell_values[nx-3, :, :]) / (2*dx)
				derivative[0, :, :] = (-3*cell_values[0, :, :] + 4*cell_values[1, :, :] - 
									cell_values[2, :, :]) / (2*dx)
				derivative[nx-1, :, :] = (3*cell_values[nx-1, :, :] - 4*cell_values[nx-2, :, :] + 
										cell_values[nx-3, :, :]) / (2*dx)
	
	elif deriv_axis == 'y':
		if dimensions == '2D':
			if grid['plane'] == 'xy':
				
				for j in range(2, ny-2):
					derivative[:, j] = (-cell_values[:, j+2] + 8*cell_values[:, j+1] - 
									8*cell_values[:, j-1] + cell_values[:, j-2]) / (12*dy)
				
				if ny > 3:
					derivative[:, 1] = (cell_values[:, 2] - cell_values[:, 0]) / (2*dy)
					derivative[:, ny-2] = (cell_values[:, ny-1] - cell_values[:, ny-3]) / (2*dy)
					derivative[:, 0] = (-3*cell_values[:, 0] + 4*cell_values[:, 1] - 
									cell_values[:, 2]) / (2*dy)
					derivative[:, ny-1] = (3*cell_values[:, ny-1] - 4*cell_values[:, ny-2] + 
										cell_values[:, ny-3]) / (2*dy)
			
			elif grid['plane'] == 'yz':
				
				for j in range(2, ny-2):
					derivative[j, :] = (-cell_values[j+2, :] + 8*cell_values[j+1, :] - 
									8*cell_values[j-1, :] + cell_values[j-2, :]) / (12*dy)
				
				if ny > 3:
					derivative[1, :] = (cell_values[2, :] - cell_values[0, :]) / (2*dy)
					derivative[ny-2, :] = (cell_values[ny-1, :] - cell_values[ny-3, :]) / (2*dy)
					derivative[0, :] = (-3*cell_values[0, :] + 4*cell_values[1, :] - 
									cell_values[2, :]) / (2*dy)
					derivative[ny-1, :] = (3*cell_values[ny-1, :] - 4*cell_values[ny-2, :] + 
										cell_values[ny-3, :]) / (2*dy)
		
		elif dimensions == '3D':
			
			
			for j in range(2, ny-2):
				derivative[:, j, :] = (-cell_values[:, j+2, :] + 8*cell_values[:, j+1, :] - 
									8*cell_values[:, j-1, :] + cell_values[:, j-2, :]) / (12*dy)
			
			if ny > 3:
				derivative[:, 1, :] = (cell_values[:, 2, :] - cell_values[:, 0, :]) / (2*dy)
				derivative[:, ny-2, :] = (cell_values[:, ny-1, :] - cell_values[:, ny-3, :]) / (2*dy)
				derivative[:, 0, :] = (-3*cell_values[:, 0, :] + 4*cell_values[:, 1, :] - 
									cell_values[:, 2, :]) / (2*dy)
				derivative[:, ny-1, :] = (3*cell_values[:, ny-1, :] - 4*cell_values[:, ny-2, :] + 
										cell_values[:, ny-3, :]) / (2*dy)
	
	elif deriv_axis == 'z':
		if dimensions == '2D':
			if grid['plane'] == 'xz':
				
				for k in range(2, nz-2):
					derivative[:, k] = (-cell_values[:, k+2] + 8*cell_values[:, k+1] - 
									8*cell_values[:, k-1] + cell_values[:, k-2]) / (12*dz)
				
				if nz > 3:
					derivative[:, 1] = (cell_values[:, 2] - cell_values[:, 0]) / (2*dz)
					derivative[:, nz-2] = (cell_values[:, nz-1] - cell_values[:, nz-3]) / (2*dz)
					derivative[:, 0] = (-3*cell_values[:, 0] + 4*cell_values[:, 1] - 
									cell_values[:, 2]) / (2*dz)
					derivative[:, nz-1] = (3*cell_values[:, nz-1] - 4*cell_values[:, nz-2] + 
										cell_values[:, nz-3]) / (2*dz)
		
			elif grid['plane'] == 'yz':
			
				for k in range(2, nz-2):
					derivative[:, k] = (-cell_values[:, k+2] + 8*cell_values[:, k+1] - 
									8*cell_values[:, k-1] + cell_values[:, k-2]) / (12*dz)
				
				if nz > 3:
					derivative[:, 1] = (cell_values[:, 2] - cell_values[:, 0]) / (2*dz)
					derivative[:, nz-2] = (cell_values[:, nz-1] - cell_values[:, nz-3]) / (2*dz)
					derivative[:, 0] = (-3*cell_values[:, 0] + 4*cell_values[:, 1] - 
									cell_values[:, 2]) / (2*dz)
					derivative[:, nz-1] = (3*cell_values[:, nz-1] - 4*cell_values[:, nz-2] + 
										cell_values[:, nz-3]) / (2*dz)
		
		elif dimensions == '3D':
			
			for k in range(2, nz-2):
				derivative[:, :, k] = (-cell_values[:, :, k+2] + 8*cell_values[:, :, k+1] - 
									8*cell_values[:, :, k-1] + cell_values[:, :, k-2]) / (12*dz)
			
			if nz > 3:
				derivative[:, :, 1] = (cell_values[:, :, 2] - cell_values[:, :, 0]) / (2*dz)
				derivative[:, :, nz-2] = (cell_values[:, :, nz-1] - cell_values[:, :, nz-3]) / (2*dz)
				derivative[:, :, 0] = (-3*cell_values[:, :, 0] + 4*cell_values[:, :, 1] - 
									cell_values[:, :, 2]) / (2*dz)
				derivative[:, :, nz-1] = (3*cell_values[:, :, nz-1] - 4*cell_values[:, :, nz-2] + 
										cell_values[:, :, nz-3]) / (2*dz)
	
	else:
		raise ValueError(f"Derivative axis '{deriv_axis}' non valid. Use 'x', 'y' or 'z'.")
	
	comp_str = f"_{component}" if component is not None else ""
	deriv_key = f"d{attribute_name}{comp_str}_d{deriv_axis}"
	
	grid[deriv_key] = derivative
	grid[f"{deriv_key}_info"] = {
		'attribute': attribute_name,
		'component': component,
		'axis': deriv_axis,
	}
	
	return derivative

def plot_grid_and_particles(vtk_file, grid, min_bounds, max_bounds, L_cell, max_particles=5000):
	# Création de la figure
	fig, ax = plt.subplots(figsize=(12, 10))


	x_min, y_min = min_bounds[0], min_bounds[1]
	dx, dy = L_cell, L_cell
	x_idx, y_idx = 0, 1  # Indices pour extraire les coordonnées Y et Z
	nx, ny = grid['nb_elems'][0], grid['nb_elems'][1]

	
	# 1. Afficher les particules (limitées à max_particles pour la performance)
	points = vtk_file.points
	if len(points) > max_particles:
		# Échantillonner aléatoirement pour ne pas surcharger le graphique
		indices = np.random.choice(len(points), max_particles, replace=False)
		points_to_plot = np.array(points)[indices]
	else:
		points_to_plot = np.array(points)
	

	points_in_box = points_to_plot
	
	# Tracer les particules
	ax.scatter(points_in_box[:, x_idx], points_in_box[:, y_idx], s=3, color='blue', alpha=0.5, label='Particules')
	
	# 2. Dessiner la grille
	# Dessiner les lignes verticales
	for i in range(nx + 1):
		ax.plot([x_min + i * dx, x_min + i * dx],
				[y_min, y_min + ny * dy],
				'k-', linewidth=0.5, alpha=0.3)
	
	# Dessiner les lignes horizontales
	for j in range(ny + 1):
		ax.plot([x_min, x_min + nx * dx],
				[y_min + j * dy, y_min + j * dy],
				'k-', linewidth=0.5, alpha=0.3)
	
	# 3. Afficher les indices des cellules
	# Ajuster le nombre de cellules à étiqueter pour éviter l'encombrement
	step_i = max(1, nx // 20)
	step_j = max(1, ny // 20)
	
	for i in range(0, nx, step_i):
		for j in range(0, ny, step_j):
			cell_center_x = x_min + (i + 0.5) * dx
			cell_center_y = y_min + (j + 0.5) * dy
			
			# Ajouter l'étiquette (i,j)
			ax.text(cell_center_x, cell_center_y, f"({i},{j})",
				fontsize=8, ha='center', va='center',
				bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
	
	# 4. Mise en forme du graphique
	ax.set_xlim(min_bounds[0], max_bounds[0])
	ax.set_ylim(min_bounds[1], max_bounds[1])
	
	ax.set_title('Particules VTK et grille de calcul')
	
	# Ajouter une boîte englobante
	rect = patches.Rectangle((min_bounds[0], min_bounds[1]),
						max_bounds[0]-min_bounds[0],
						max_bounds[1]-min_bounds[1],
						linewidth=2, edgecolor='r', facecolor='none')
	ax.add_patch(rect)
	
	plt.tight_layout()
	
	return fig, ax



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import matplotlib.patches as patches
from matplotlib.collections import LineCollection



def plot_streamlines(x1, x2, u1, u2, 
					plane='XY', apply_filter=True,
					nx=400, ny=400, margin=0.1, obstacles=None, 
					density=5,
					min_speed_ratio=0.01,
					safety_margin=0.02,
					water_surface_x=None,
					water_surface_h=None,
					save=False, savepath=None):
	
	# 1. Calcul des bornes du domaine
	x1_min, x1_max = x1.min(), x1.max()
	x2_min, x2_max = x2.min(), x2.max()
	
	dx1 = (x1_max - x1_min) * margin
	dx2 = (x2_max - x2_min) * margin
	
	# Création de la grille régulière
	step_x1 = (x1_max - x1_min + 2*dx1) / (nx - 1)
	step_x2 = (x2_max - x2_min + 2*dx2) / (ny - 1)
	
	x1_grid = (x1_min - dx1) + np.arange(nx) * step_x1
	x2_grid = (x2_min - dx2) + np.arange(ny) * step_x2
	
	X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='xy')
	points = np.column_stack((x1, x2))
	xi = np.column_stack((X1.ravel(), X2.ravel()))

	all_points = points.copy()
	all_u1 = u1.copy()
	all_u2 = u2.copy()
	
	if apply_filter:
		# 2. Ajout de points fictifs sur les obstacles pour respecter les conditions aux limites
		
		
		if obstacles:
			# Déterminer les clés selon le plan
			if plane == 'XY':
				rect_keys = ('x_min', 'x_max', 'y_min', 'y_max')
				curve_range_key = 'x_range'
				curve_func_key = 'y_func'
			elif plane == 'XZ':
				rect_keys = ('x_min', 'x_max', 'z_min', 'z_max')
				curve_range_key = 'x_range'
				curve_func_key = 'z_func'
			elif plane == 'YZ':
				rect_keys = ('y_min', 'y_max', 'z_min', 'z_max')
				curve_range_key = 'y_range'
				curve_func_key = 'z_func'
			
			for obs in obstacles:
				if obs['type'] == 'rectangle':
					# Ajout de points sur les bords supérieur et inférieur
					x_edge = np.linspace(obs[rect_keys[0]], obs[rect_keys[1]], 50)
					
					# Bord supérieur
					y_top = np.full_like(x_edge, obs[rect_keys[3]])
					all_points = np.vstack([all_points, np.column_stack((x_edge, y_top))])
					all_u1 = np.concatenate([all_u1, np.zeros_like(x_edge)])
					all_u2 = np.concatenate([all_u2, np.zeros_like(x_edge)])
					
					# Bord inférieur
					y_bottom = np.full_like(x_edge, obs[rect_keys[2]])
					all_points = np.vstack([all_points, np.column_stack((x_edge, y_bottom))])
					all_u1 = np.concatenate([all_u1, np.zeros_like(x_edge)])
					all_u2 = np.concatenate([all_u2, np.zeros_like(x_edge)])
					
				elif obs['type'] == 'curve':
					# Ajout de points sur la courbe
					x_vals = np.linspace(obs[curve_range_key][0], obs[curve_range_key][1], 100)
					y_vals = obs[curve_func_key](x_vals)
					all_points = np.vstack([all_points, np.column_stack((x_vals, y_vals))])
					all_u1 = np.concatenate([all_u1, np.zeros_like(x_vals)])
					all_u2 = np.concatenate([all_u2, np.zeros_like(x_vals)])
		
	# 3. Interpolation des vitesses sur la grille
	U = griddata(all_points, all_u1, xi, method='linear').reshape(X1.shape)
	V = griddata(all_points, all_u2, xi, method='linear').reshape(X1.shape)
	speed = np.sqrt(U**2 + V**2)
	
	if apply_filter:
		# 4. Création du masque de surface d'eau (approche exacte)
		water_mask = np.ones(X1.shape, dtype=bool)
		surface_cell_mask = np.zeros(X1.shape, dtype=bool)
		
		if water_surface_x is not None and water_surface_h is not None:
			# Nettoyage des données
			valid_mask = ~np.isnan(water_surface_h) & ~np.isnan(water_surface_x)
			water_x_clean = water_surface_x[valid_mask]
			water_h_clean = water_surface_h[valid_mask]
			
			# Créer une grille de référence
			x_min, x_max = x1_grid.min(), x1_grid.max()
			z_min, z_max = x2_grid.min(), x2_grid.max()
			
			# Calculer la taille des cellules
			dx = (x_max - x_min) / (nx - 1)
			dz = (z_max - z_min) / (ny - 1)
			
			# Identifier les cellules contenant les points de surface
			for x, h in zip(water_x_clean, water_h_clean):
				# Trouver l'indice x
				j = int((x - x_min) / dx)
				
				# Trouver l'indice z
				i = int((h - z_min) / dz)
				
				# Vérifier les limites
				if 0 <= j < nx and 0 <= i < ny:
					surface_cell_mask[i, j] = True
					water_mask[:, j] = (X2[:, j] <= h + safety_margin)
		
		# 5. Création du masque des obstacles
		mask = np.ones(X1.shape, dtype=bool)
		if obstacles:
			for obs in obstacles:
				if obs['type'] == 'rectangle':
					in_obs = ((X1 >= obs[rect_keys[0]]) & (X1 <= obs[rect_keys[1]]) & 
							(X2 >= obs[rect_keys[2]]) & (X2 <= obs[rect_keys[3]]))
					mask &= ~in_obs
				
				elif obs['type'] == 'curve':
					x_min, x_max = obs[curve_range_key]
					in_x_range = (X1 >= x_min) & (X1 <= x_max)
					y_curve = obs[curve_func_key](X1)
					if obs.get('region', 'below') == 'below':
						in_obs = in_x_range & (X2 < y_curve)
					else:
						in_obs = in_x_range & (X2 > y_curve)
					mask &= ~in_obs
		
		# 6. Masquage des zones à faible vitesse
		if min_speed_ratio > 0:
			speed_min = np.nanmin(speed)
			speed_max = np.nanmax(speed)
			min_speed = speed_min + min_speed_ratio * (speed_max - speed_min)
			zero_speed_mask = speed < min_speed
			
			# Combinaison des masques
			combined_mask = mask & ~zero_speed_mask & water_mask

		else:
			combined_mask = mask & water_mask
		
		# Application du masque combiné
		U[~combined_mask] = np.nan
		V[~combined_mask] = np.nan
		speed[~combined_mask] = np.nan
	
		# Masquage explicite des cellules au-dessus de la surface
		for j in range(X1.shape[1]):
			surface_indices = np.where(surface_cell_mask[:, j])[0]
			if len(surface_indices) > 0:
				i_surface = np.min(surface_indices)
				U[:i_surface, j] = np.nan
				V[:i_surface, j] = np.nan

	# 7. Tracé des streamlines
	fig, ax = plt.subplots(figsize=(12, 6))
	vmin = np.nanmin(speed)
	vmax = np.nanmax(speed)
	
	# Calcul des streamlines
	stream = ax.streamplot(
		X1, X2, U, V, 
		density=density,
		color=speed,  
		cmap='viridis',    
		linewidth=1.5,
		broken_streamlines=False,
		norm=plt.Normalize(vmin=vmin, vmax=vmax))
	
	# Barre de couleur
	cbar = plt.colorbar(stream.lines, ax=ax, shrink=0.8, pad=0.02)
	cbar.set_label('Normalized velocity [m/s]')
	
	# 8. Visualisation des obstacles
	if obstacles:
		# Déterminer les clés selon le plan
		if plane == 'XY':
			rect_keys = ('x_min', 'x_max', 'y_min', 'y_max')
			curve_range_key = 'x_range'
			curve_func_key = 'y_func'
		elif plane == 'XZ':
			rect_keys = ('x_min', 'x_max', 'z_min', 'z_max')
			curve_range_key = 'x_range'
			curve_func_key = 'z_func'
		elif plane == 'YZ':
			rect_keys = ('y_min', 'y_max', 'z_min', 'z_max')
			curve_range_key = 'y_range'
			curve_func_key = 'z_func'
		
		for obs in obstacles:
			if obs['type'] == 'rectangle':
				# Extraction des coordonnées spécifiques au plan
				coord1_min = obs[rect_keys[0]]
				coord1_max = obs[rect_keys[1]]
				coord2_min = obs[rect_keys[2]]
				coord2_max = obs[rect_keys[3]]
				
				# Dessin du rectangle
				w = coord1_max - coord1_min
				h = coord2_max - coord2_min
				rect = patches.Rectangle(
					(coord1_min, coord2_min), w, h,
					linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7
				)
				ax.add_patch(rect)
				
			elif obs['type'] == 'curve':
				# Gestion des courbes (inchangée)
				x_min, x_max = obs[curve_range_key]
				x_vals = np.linspace(x_min, x_max, 100)
				y_vals = obs[curve_func_key](x_vals)
				ax.plot(x_vals, y_vals, 'k-', linewidth=2)
				
				if obs.get('region', 'below') == 'below':
					ax.fill_between(x_vals, y_vals, x2_grid.min(), 
									color='gray', alpha=0.7)
				else:
					ax.fill_between(x_vals, y_vals, x2_grid.max(), 
									color='gray', alpha=0.7)
		
	# 9. Configuration finale
	if plane == 'XY':
		xlabel, ylabel = 'X [m]', 'Y [m]'
	elif plane == 'XZ':
		xlabel, ylabel = 'X [m]', 'Z [m]'
	elif plane == 'YZ':
		xlabel, ylabel = 'Y [m]', 'Z [m]'
	
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True, alpha=0.4, ls="--")
	ax.set_xlim(x1_min, x1_max)
	ax.set_ylim(x2_min, x2_max)
	ax.set_xticks(np.linspace(x1_min, x1_max, 5))
	ax.set_yticks(np.linspace(x2_min, x2_max, 5))
	plt.tight_layout()
	
	if save and savepath is not None:
		plt.savefig(f'{savepath}streamlines.pdf', dpi=30, bbox_inches='tight')
	
	return fig, ax


def plot_quiver(x, y, u, v, save=False, savepath=None): 
	step = 10
	fig, ax = plt.subplots(figsize=(12, 6))

	ax.quiver(x[::step], y[::step], u[::step], v[::step])      
	#z = parabole(np.linspace(8, 12, 100)) 
	#ax.fill_between(np.linspace(8, 12, 100), np.zeros_like(z), z, color='grey') 
	#ax.plot(np.linspace(8, 12, 100), z, color='black', linewidth=5)  
	
	rect = patches.Rectangle( 
		(1.5, 0.211), 0.5, 0.064, 
		linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7 
	)  
	ax.add_patch(rect) 
	
	ax.set_xlabel('Position x [m]') 
	ax.set_ylabel('Height [m]') 
	ax.grid(True, alpha=0.4, ls='--') 
	#ax.set_xlim(1.3, 2.2) 
	#ax.set_xticks(np.linspace(1.3, 2.2, 10)) 
	#ax.set_ylim(0.12, 0.24) 
	#ax.set_yticks(np.linspace(0.12, 0.24, 5)) 
	
	fig.tight_layout()

	if save and savepath is not None:
		plt.savefig(f'{savepath}quiver.pdf', dpi=30, bbox_inches='tight')
