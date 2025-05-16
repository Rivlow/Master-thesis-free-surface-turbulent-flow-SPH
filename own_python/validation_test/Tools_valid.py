import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.colors import Normalize
import numba as nb
import matplotlib.patches as patches
from scipy.spatial import cKDTree


# Local imports
sys.path.append(os.getcwd())
from own_python.write_scenes.Tools_scenes import *
from own_python.validation_test.Tools_valid import *


def configure_latex():
	"""Configure matplotlib to use LaTeX for rendering text."""
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')


# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 22, 22
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
					bounds, thickness):
	
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
				plt.savefig(f'{savepath}_mean_single.pdf', dpi=300)
		else:
			name, dim = get_name(slice_result[0])
			for slice in slice_result:
				plt.scatter(slice['coordinates'], slice['values'], s=5)
				plt.ylabel(f'{name}(t) [{dim}]')
				plt.xlabel(f'Time [t]')
				plt.grid()
			plt.tight_layout()


			if save and savepath is not None:
				plt.savefig(f'{savepath}_single.pdf', dpi=300)

	return slice_result


def get_multiple_slices(vtk_file, attribute, mean=False,
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
	
	points = vtk_file.points
	
	min_coords = np.min(points, axis=0)
	max_coords = np.max(points, axis=0)
	
	if along is None:
		target_min = min_coords[target_idx]
		target_max = max_coords[target_idx]
	else:
		target_min = along[0]
		target_max = along[1]
	
	positions = np.arange(target_min + thickness, target_max - thickness, thickness)
	
	slice_result = []
	
	# Create slice depending on plane and projected axis
	for pos in positions:
		
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
			attr_values = vtk_file.point_data[attribute][mask]
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
				'dt':vtk_file['dt'][0],
				'time':vtk_file['time'][0]
			})

	if plot:
		plt.figure()
		if mean:
			store_val = []
			positions = []
			name, dim = get_name(slice_result[0])
			
			for slice in slice_result:
				store_val.append(np.mean(slice['values']))
				positions.append(slice['position'])
			
			positions = np.array(positions)
			store_val = np.asarray(store_val)
			
			mean_values = np.mean(store_val)
			std_values = np.std(store_val)

			plt.plot(positions, store_val, 'b-o')
			
			plt.axhline(y=mean_values, color='blue', linestyle='-', label=r'$\mu$(u)')
			
			plt.axhspan(mean_values - std_values, mean_values + std_values, 
						alpha=0.2, color='red', label=r'$\sigma$(u)')
			

			plt.grid()
			plt.ylabel(f'{name}({other_axis}) [{dim}]')
			plt.xlabel(f'{axis} [m]')
			plt.legend()
			plt.ylim((0.9*np.min(store_val), 1.1*np.max(store_val)))
			plt.tight_layout()

			if save and savepath is not None:
				plt.savefig(f'{savepath}_mean_mult.pdf', dpi=300)
		else:
			name, dim = get_name(slice_result[0])
			for slice in slice_result:
				plt.scatter(slice['coordinates'], slice['values'], s=5)
				plt.ylabel(f'{name}({other_axis}) [{dim}]')
				plt.xlabel(f'{other_axis} [m]')
			plt.tight_layout()


			if save and savepath is not None:
				plt.savefig(f'{savepath}_mult.pdf', dpi=300)
			
	
	return slice_result
	

def spatial_derivative__(slices, axis, plot=False, save=False, savepath=None):

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
			plt.savefig(f'{savepath}_dx.pdf', dpi=300)
		
	return derivative, positions

def time_derivative(slices, plot=False, save=False, savepath=None):

	mean_vals = []
	time = []

	for slice in slices:
		mean_vals.append(np.mean(slice['values']))
		time.append(slice['time'][0])

	mean_vals = np.array(mean_vals)
	times = np.array(time)
	
	# Forward finite difference method
	derivative = np.zeros_like(mean_vals)
	derivative[1:] = (mean_vals[1:] - mean_vals[:-1])/(times[1:] - times[:-1]) # derivative[1:] = (mean_vals[:-1] - mean_vals[:1])/(times[:-1] - times[:1]) to compare to init

	if plot:

		name, dim = get_name(slices[0])
		plt.figure()
		plt.plot(times[1:], derivative[1:], 'b-o')
		plt.axhline(y=0, color='k', linestyle='--') 
		plt.xlabel('Time t [s]')
		plt.ylabel(f'd{name}/dt [{dim}/s]')
		plt.grid(True)
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}_dt.pdf', dpi=300)
		
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
		
		return plotter  # Retourner le plotter pour permettre show() plus tard
	
	else:
		# Utiliser Matplotlib
		x = points[:, 0]
		y = points[:, 1]
		
		# Créer la figure et l'axe
		fig, ax = plt.subplots()
		
		# Tracer les points avec une coloration selon l'attribut
		norm = Normalize(vmin=0, vmax=2)
		scatter = ax.scatter(x, y, c=magnitude, cmap='viridis', s=1, alpha=0.8, norm=norm)
		
		# Ajouter une barre de couleur
		#cbar = plt.colorbar(scatter, ax=ax)
		#cbar.set_label(attribute)

		z = parabole(np.linspace(8, 12, 100))
		plt.fill_between(np.linspace(8, 12, 100), np.zeros_like(z), z, color='grey')

		
		# Ajouter une grille
		ax.grid(True, linestyle='--', alpha=0.7)
		
		# Configurer les axes
		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		#ax.set_xlim(np.min(x), 24)
		ax.set_ylim(0, np.max(y))
		ax.set_aspect(20)
		ax.set_axisbelow(True)  # Mettre la grille en arrière-plan
		
		# Enregistrer si demandé
		if save and savepath is not None:
			plt.savefig(f'{savepath}.pdf', dpi=300, bbox_inches='tight')
		
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
	error_Q_v = 100 * (Q_v_th - Q_v_mean) / Q_v_th
	error_Q_m = 100 * (Q_m_th - Q_m_mean) / Q_m_th

	print(f'Error on flow rate: {error_Q_v}%')
	print(f'Error on masss flow: {error_Q_v}%')

	if plot:

		# Flow rate
		plt.figure()
		plt.bar(span, Q_v, alpha=0.7, color='royalblue', width=span[1] - span[0])
		plt.hlines(Q_v_th, span[0], span[-1], color='blue', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_v_th:.2f} [m$^2$/s]')
		plt.hlines(Q_v_mean, span[0], span[-1], color='navy', linestyle='-', linewidth=2,
				label=fr'Mean Q: {Q_v_mean:.2f} [m$^2$/s]')
		plt.xlabel('Position x [m]')
		plt.ylabel(r'Flow rate [m$^2$/s]')
		plt.grid(True, alpha=0.7)
		plt.legend(loc='center')
		plt.tight_layout()
		if save and savepath is not None:
			plt.savefig(f'{savepath}flow_rate.pdf', dpi=300)

		# Mass flow
		plt.figure()
		plt.bar(span, Q_m, alpha=0.7, color='green', width=span[1] - span[0])
		plt.hlines(Q_m_th, span[0], span[-1], color='blue', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_m_th:.2f} [Kg/s]')
		plt.hlines(Q_m_mean, span[0], span[-1], color='navy', linestyle='-', linewidth=2,
				label=fr'Mean Q: {Q_m_mean:.2f} [Kg/s]')
		plt.xlabel('Position x [m]')
		plt.ylabel(r'Mass flow [Kg/s]')
		plt.grid(True, alpha=0.7)
		plt.legend(loc='center')
		plt.tight_layout()
		if save and savepath is not None:
			plt.savefig(f'{savepath}mass_flow.pdf', dpi=300)


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
		fig, ax = plt.subplots()
		ax.bar(bin_centers, counts, width=bin_width*0.9, alpha=0.7, color='blue')
		
		ax.set_xlabel('Position y')
		ax.set_ylabel('Number of particles')
		ax.grid(True, alpha=0.7)
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}/particle_distribution.pdf', dpi=300)
	
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
		xlabel, ylabel = 'X', 'Y'
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
		xlabel, ylabel = 'X', 'Z'
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
		xlabel, ylabel = 'Y', 'Z'
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
	
	fig, ax = plt.subplots()
	
	
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
	ax.grid(True, alpha=0.7)
	ax.legend(loc='upper right')
	
	# Définir les limites pour montrer tout le domaine avec un peu de marge
	ax.set_xlim(14.5, 15.5)
	ax.set_ylim(-1.65, 1.65)
	
	plt.tight_layout()
	
	# Enregistrer la figure si demandé
	if save and savepath is not None:
		plt.savefig(f'{savepath}/framed_particles.pdf', dpi=300)
	
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
def compute_grid_values(grid, vtk_file, attribute_name, h, W, component=None):
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
	
	grid['cell_values'] = result
	grid['cell_values_info'] = {
		'attribute': attribute_name,
		'component': component
	}
	return grid

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

