import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.colors import Normalize
import numba as nb
import matplotlib.patches as patches


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


def create_grid(vtk_files, bounds, nb_elem, dimensions='3D', plane='xy', r=None):
	"""
	Crée une grille pour analyser des données VTK avec stockage des particules voisines aux bords.
	
	Args:
		vtk_files: Un fichier VTK ou une liste de fichiers VTK à traiter
		bounds: Les limites de la grille
		nb_elem: Nombre d'éléments par dimension
		dimensions: '3D' ou '2D'
		plane: Plan à utiliser si dimensions='2D', l'un de 'xy', 'xz', 'yz'
		r: Paramètre de rayon pour le calcul des voisinages
	"""
	
	if not isinstance(vtk_files, list):
		vtk_files = [vtk_files]
	
	# Vérification du plan pour le mode 2D
	if dimensions == '2D' and plane not in ['xy', 'xz', 'yz']:
		raise ValueError("Le plan doit être l'un de 'xy', 'xz', 'yz' en mode 2D")
	
	# Préparer les limites en fonction du format d'entrée
	if isinstance(bounds, dict):
		# Format dictionnaire (généralement pour 2D)
		if dimensions == '3D':
			raise ValueError("Pour le mode 3D, les bornes doivent être au format [(x_min, y_min, z_min), (x_max, y_max, z_max)]")
			
		# Créer des limites 3D complètes à partir du dictionnaire 2D
		min_bounds = [0, 0, 0]
		max_bounds = [1, 1, 1]  # Valeurs par défaut pour les dimensions non spécifiées
		
		if plane == 'xy':
			if all(k in bounds for k in ['x_min', 'y_min', 'x_max', 'y_max']):
				min_bounds[0], min_bounds[1] = bounds['x_min'], bounds['y_min']
				max_bounds[0], max_bounds[1] = bounds['x_max'], bounds['y_max']
			else:
				raise ValueError("Les bornes pour le plan 'xy' doivent contenir 'x_min', 'y_min', 'x_max', 'y_max'")
		
		elif plane == 'xz':
			if all(k in bounds for k in ['x_min', 'z_min', 'x_max', 'z_max']):
				min_bounds[0], min_bounds[2] = bounds['x_min'], bounds['z_min']
				max_bounds[0], max_bounds[2] = bounds['x_max'], bounds['z_max']
			else:
				raise ValueError("Les bornes pour le plan 'xz' doivent contenir 'x_min', 'z_min', 'x_max', 'z_max'")
		
		elif plane == 'yz':
			if all(k in bounds for k in ['y_min', 'z_min', 'y_max', 'z_max']):
				min_bounds[1], min_bounds[2] = bounds['y_min'], bounds['z_min']
				max_bounds[1], max_bounds[2] = bounds['y_max'], bounds['z_max']
			else:
				raise ValueError("Les bornes pour le plan 'yz' doivent contenir 'y_min', 'z_min', 'y_max', 'z_max'")
	
	elif isinstance(bounds, list) and len(bounds) == 2:
		# Format liste/tuple standard [(x_min, y_min, z_min), (x_max, y_max, z_max)]
		min_bounds, max_bounds = bounds
		
		# Vérifier si les bornes sont complètes selon la dimension
		if dimensions == '3D' and (len(min_bounds) < 3 or len(max_bounds) < 3):
			raise ValueError("Pour le mode 3D, les bornes doivent contenir x, y et z")
		
		# Pour 2D, adapter selon le plan si nécessaire
		if dimensions == '2D' and len(min_bounds) == 2 and len(max_bounds) == 2:
			# Convertir des bornes 2D en 3D selon le plan
			min_temp, max_temp = [0, 0, 0], [1, 1, 1]
			
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
		raise ValueError("Format de bornes non pris en charge. Utilisez [(x_min, y_min, z_min), (x_max, y_max, z_max)] ou un dictionnaire adapté au plan 2D")
	
	# Create coordinate spans
	x_span = np.linspace(min_bounds[0], max_bounds[0], nb_elem+1)
	y_span = np.linspace(min_bounds[1], max_bounds[1], nb_elem+1)
	z_span = np.linspace(min_bounds[2], max_bounds[2], nb_elem+1)
	
	# Créer le maillage en fonction de la dimension et du plan
	if dimensions == '3D':
		X, Y, Z = np.meshgrid(x_span, y_span, z_span)
	else:  # 2D
		if plane == 'xy':
			X, Y = np.meshgrid(x_span, y_span)
		elif plane == 'xz':
			X, Z = np.meshgrid(x_span, z_span)
		elif plane == 'yz':
			Y, Z = np.meshgrid(y_span, z_span)
	
	# Calculate boundaries and cell sizes
	x_min, y_min, z_min = x_span[0], y_span[0], z_span[0]
	x_max, y_max, z_max = x_span[-1], y_span[-1], z_span[-1]
	dx, dy, dz = (x_max - x_min) / nb_elem, (y_max - y_min) / nb_elem, (z_max - z_min) / nb_elem
	
	# Si r n'est pas spécifié, on utilise la taille de cellule
	if r is None:
		r = min(dx, dy) / 4 if dimensions == '2D' else min(dx, dy, dz) / 4
	
	# Initialize grid cells dictionary with standard structure (just particles)
	if dimensions == '3D':
		grid_cells = {cell_idx: {
			'particles': [],
			'border_neighbors': {
				'left': [], 'right': [], 'bottom': [], 'top': [], 'front': [], 'back': []
			}
		} for cell_idx in np.ndindex((nb_elem, nb_elem, nb_elem))}
	else:  # 2D
		grid_cells = {cell_idx: {
			'particles': [],
			'border_neighbors': {
				'left': [], 'right': [], 'bottom': [], 'top': []
			}
		} for cell_idx in np.ndindex((nb_elem, nb_elem))}
	
	# Process all particles at once
	total_particles = 0
	skipped_particles = 0
	
	# ÉTAPE 1: Assigner chaque particule à sa cellule
	for vtk_idx, vtk in enumerate(vtk_files):
		points = np.array(vtk.points)
		total_particles += len(points)
		
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
		skipped_particles += len(points) - len(valid_points)
		
		# Calculate cell indices for all valid points
		if dimensions == '3D':
			i = np.minimum(((valid_points[:, 0] - x_min) / dx).astype(int), nb_elem - 1)
			j = np.minimum(((valid_points[:, 1] - y_min) / dy).astype(int), nb_elem - 1)
			k = np.minimum(((valid_points[:, 2] - z_min) / dz).astype(int), nb_elem - 1)
			
			for idx, (i_val, j_val, k_val) in enumerate(zip(i, j, k)):
				grid_cells[(i_val, j_val, k_val)]['particles'].append((vtk_idx, valid_indices[idx]))
		
		else:  # 2D
			if plane == 'xy':
				i = np.minimum(((valid_points[:, 0] - x_min) / dx).astype(int), nb_elem - 1)
				j = np.minimum(((valid_points[:, 1] - y_min) / dy).astype(int), nb_elem - 1)
			elif plane == 'xz':
				i = np.minimum(((valid_points[:, 0] - x_min) / dx).astype(int), nb_elem - 1)
				j = np.minimum(((valid_points[:, 2] - z_min) / dz).astype(int), nb_elem - 1)
			elif plane == 'yz':
				i = np.minimum(((valid_points[:, 1] - y_min) / dy).astype(int), nb_elem - 1)
				j = np.minimum(((valid_points[:, 2] - z_min) / dz).astype(int), nb_elem - 1)
			
			for idx, (i_val, j_val) in enumerate(zip(i, j)):
				grid_cells[(i_val, j_val)]['particles'].append((vtk_idx, valid_indices[idx]))
	
	# ÉTAPE 2: Pour chaque cellule, identifier les particules des cellules voisines
	# pour les bords correspondants
	if dimensions == '2D':
		for i in range(nb_elem):
			for j in range(nb_elem):
				# Particules de la cellule courante
				self_particles = grid_cells[(i, j)]['particles'].copy()
				
				# Bord gauche: utiliser les particules de la cellule à gauche (i-1,j) ET de la cellule elle-même
				if i > 0:
					grid_cells[(i, j)]['border_neighbors']['left'] = grid_cells[(i-1, j)]['particles'].copy() + self_particles
				else:
					grid_cells[(i, j)]['border_neighbors']['left'] = self_particles
				
				# Bord droit: utiliser les particules de la cellule à droite (i+1,j) ET de la cellule elle-même
				if i < nb_elem - 1:
					grid_cells[(i, j)]['border_neighbors']['right'] = grid_cells[(i+1, j)]['particles'].copy() + self_particles
				else:
					grid_cells[(i, j)]['border_neighbors']['right'] = self_particles
				
				# Bord bas: utiliser les particules de la cellule du bas (i,j-1) ET de la cellule elle-même
				if j > 0:
					grid_cells[(i, j)]['border_neighbors']['bottom'] = grid_cells[(i, j-1)]['particles'].copy() + self_particles
				else:
					grid_cells[(i, j)]['border_neighbors']['bottom'] = self_particles
				
				# Bord haut: utiliser les particules de la cellule du haut (i,j+1) ET de la cellule elle-même
				if j < nb_elem - 1:
					grid_cells[(i, j)]['border_neighbors']['top'] = grid_cells[(i, j+1)]['particles'].copy() + self_particles
				else:
					grid_cells[(i, j)]['border_neighbors']['top'] = self_particles

	else:  # 3D - code pour 3D
		for i in range(nb_elem):
			for j in range(nb_elem):
				for k in range(nb_elem):
					# Particules de la cellule courante
					self_particles = grid_cells[(i, j, k)]['particles'].copy()
					
					# Bord gauche (direction -x): particules de (i-1,j,k) + cellule elle-même
					if i > 0:
						grid_cells[(i, j, k)]['border_neighbors']['left'] = grid_cells[(i-1, j, k)]['particles'].copy() + self_particles
					else:
						grid_cells[(i, j, k)]['border_neighbors']['left'] = self_particles
					
					# Bord droit (direction +x): particules de (i+1,j,k) + cellule elle-même
					if i < nb_elem - 1:
						grid_cells[(i, j, k)]['border_neighbors']['right'] = grid_cells[(i+1, j, k)]['particles'].copy() + self_particles
					else:
						grid_cells[(i, j, k)]['border_neighbors']['right'] = self_particles
					
					# Bord bas (direction -y): particules de (i,j-1,k) + cellule elle-même
					if j > 0:
						grid_cells[(i, j, k)]['border_neighbors']['bottom'] = grid_cells[(i, j-1, k)]['particles'].copy() + self_particles
					else:
						grid_cells[(i, j, k)]['border_neighbors']['bottom'] = self_particles
					
					# Bord haut (direction +y): particules de (i,j+1,k) + cellule elle-même
					if j < nb_elem - 1:
						grid_cells[(i, j, k)]['border_neighbors']['top'] = grid_cells[(i, j+1, k)]['particles'].copy() + self_particles
					else:
						grid_cells[(i, j, k)]['border_neighbors']['top'] = self_particles
					
					# Bord arrière (direction -z): particules de (i,j,k-1) + cellule elle-même
					if k > 0:
						grid_cells[(i, j, k)]['border_neighbors']['back'] = grid_cells[(i, j, k-1)]['particles'].copy() + self_particles
					else:
						grid_cells[(i, j, k)]['border_neighbors']['back'] = self_particles
					
					# Bord avant (direction +z): particules de (i,j,k+1) + cellule elle-même
					if k < nb_elem - 1:
						grid_cells[(i, j, k)]['border_neighbors']['front'] = grid_cells[(i, j, k+1)]['particles'].copy() + self_particles
					else:
						grid_cells[(i, j, k)]['border_neighbors']['front'] = self_particles
	
	print(f"Particules totales: {total_particles}")
	print(f"Particules ignorées (hors limites): {skipped_particles} ({skipped_particles/total_particles*100:.2f}%)")
	print(f"Particules dans la grille: {total_particles - skipped_particles}")
	
	# Create grid object with all properties
	grid = {
		'nb_elem': nb_elem, 'cells': grid_cells, 
		'dimensions': dimensions,
		'total_particles': total_particles,
		'included_particles': total_particles - skipped_particles,
		'skipped_particles': skipped_particles,
		'neighbor_radius': r
	}
	
	# Ajouter les propriétés pour chaque dimension
	grid.update({
		'x_span': x_span, 'x_min': x_min, 'x_max': x_max, 'dx': dx
	})
	
	grid.update({
		'y_span': y_span, 'y_min': y_min, 'y_max': y_max, 'dy': dy
	})
	
	grid.update({
		'z_span': z_span, 'z_min': z_min, 'z_max': z_max, 'dz': dz
	})
	
	# Ajouter les propriétés spécifiques à la dimension
	if dimensions == '3D':
		grid.update({'X': X, 'Y': Y, 'Z': Z})
	else:  # 2D
		grid['plane'] = plane
		if plane == 'xy':
			grid.update({'X': X, 'Y': Y})
		elif plane == 'xz':
			grid.update({'X': X, 'Z': Z})
		elif plane == 'yz':
			grid.update({'Y': Y, 'Z': Z})

	return grid


def compute_grid_values(grid, vtk_file, attribute_name, r, W, component=None):
	"""
	Calcule les valeurs aux interfaces des cellules en utilisant les voisinages pré-calculés.
	Fonctionne pour les grilles 2D et 3D.
	
	Args:
		grid: Grille créée par create_grid avec les voisinages aux bords
		vtk_file: Fichier VTK contenant les particules
		attribute_name: Nom de l'attribut à interpoler
		r: Rayon du kernel d'interpolation
		W: Fonction de kernel SPH
		component: Composante à extraire si l'attribut est vectoriel (None = toutes)
	"""
	nb_elem = grid['nb_elem']
	dimensions = grid['dimensions']
	
	# Correspondances des noms d'interfaces
	interface_mapping_2d = {
		'left': 'left',    # direction -x
		'right': 'right',  # direction +x
		'down': 'bottom',  # direction -y
		'up': 'top'        # direction +y
	}
	
	interface_mapping_3d = {
		'left': 'left',     # direction -x
		'right': 'right',   # direction +x
		'down': 'bottom',   # direction -y
		'up': 'top',        # direction +y
		'back': 'back',     # direction -z
		'front': 'front'    # direction +z
	}
	
	# Créer le dictionnaire des clés pour chaque direction
	value_keys = {}
	if dimensions == '2D':
		for direction in ['left', 'right', 'up', 'down']:
			suffix = f"_{component}" if component is not None else ""
			value_keys[direction] = f"{attribute_name}{suffix}_{direction}"
	else:  # 3D
		for direction in ['left', 'right', 'up', 'down', 'front', 'back']:
			suffix = f"_{component}" if component is not None else ""
			value_keys[direction] = f"{attribute_name}{suffix}_{direction}"
	
	# Déterminer si l'attribut est vectoriel et son nombre de composantes
	is_vector = False
	num_components = 1
	
	if hasattr(vtk_file, 'point_data') and attribute_name in vtk_file.point_data:
		sample_attr = vtk_file.point_data[attribute_name]
		if len(sample_attr.shape) > 1 and sample_attr.shape[1] > 1:
			is_vector = True
			num_components = sample_attr.shape[1]
	
	kernel_radius = 4*r
	
	# Initialiser les tableaux pour les valeurs aux interfaces
	if dimensions == '2D':
		shape = (nb_elem, nb_elem)
		interface_types = ['left', 'right', 'up', 'down']
	else:  # 3D
		shape = (nb_elem, nb_elem, nb_elem)
		interface_types = ['left', 'right', 'up', 'down', 'front', 'back']
	
	result_shape = shape + ((num_components,) if is_vector and component is None else ())
	
	# Dictionnaires pour stocker les résultats
	grid_values = {interface: np.zeros(result_shape) for interface in interface_types}
	particle_counts = {interface: np.zeros(shape) for interface in interface_types}
	
	# Positions des interfaces pour le calcul des distances
	dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
	x_min, y_min, z_min = grid['x_min'], grid['y_min'], grid['z_min']
	
	# Pré-calculer les positions des interfaces
	if dimensions == '2D':
		# Pour une grille 2D
		i_indices, j_indices = np.meshgrid(np.arange(nb_elem), np.arange(nb_elem), indexing='ij')
		
		# Positions des centres des cellules
		x_centers = x_min + (i_indices + 0.5) * dx
		y_centers = y_min + (j_indices + 0.5) * dy
		
		# Positions des interfaces
		interface_positions = {
			'left': np.stack([x_min + i_indices * dx, y_centers], axis=-1),
			'right': np.stack([x_min + (i_indices + 1) * dx, y_centers], axis=-1),
			'up': np.stack([x_centers, y_min + (j_indices + 1) * dy], axis=-1),
			'down': np.stack([x_centers, y_min + j_indices * dy], axis=-1)
		}
		
		cell_iterator = [(i, j) for i in range(nb_elem) for j in range(nb_elem)]
	
	else:  # 3D
		# Pour une grille 3D
		i_indices, j_indices, k_indices = np.meshgrid(np.arange(nb_elem), 
													np.arange(nb_elem), 
													np.arange(nb_elem), 
													indexing='ij')
		
		# Positions des centres des cellules
		x_centers = x_min + (i_indices + 0.5) * dx
		y_centers = y_min + (j_indices + 0.5) * dy
		z_centers = z_min + (k_indices + 0.5) * dz
		
		# Positions des interfaces
		interface_positions = {
			'left': np.stack([x_min + i_indices * dx, y_centers, z_centers], axis=-1),
			'right': np.stack([x_min + (i_indices + 1) * dx, y_centers, z_centers], axis=-1),
			'down': np.stack([x_centers, y_min + j_indices * dy, z_centers], axis=-1),
			'up': np.stack([x_centers, y_min + (j_indices + 1) * dy, z_centers], axis=-1),
			'back': np.stack([x_centers, y_centers, z_min + k_indices * dz], axis=-1),
			'front': np.stack([x_centers, y_centers, z_min + (k_indices + 1) * dz], axis=-1)
		}
		
		cell_iterator = [(i, j, k) for i in range(nb_elem) for j in range(nb_elem) for k in range(nb_elem)]

	points = vtk_file.points
	
	# Traitement des interfaces pour chaque cellule
	for cell_idx in cell_iterator:
		for interface_type in interface_types:
			# Récupérer le nom de l'interface correspondant dans la grille
			if dimensions == '2D':
				grid_interface_name = interface_mapping_2d[interface_type]
				# Position de l'interface pour cette cellule
				i, j = cell_idx
				interface_pos = interface_positions[interface_type][i, j]
			else:  # 3D
				grid_interface_name = interface_mapping_3d[interface_type]
				# Position de l'interface pour cette cellule
				i, j, k = cell_idx
				interface_pos = interface_positions[interface_type][i, j, k]
			
			# Récupérer les particules voisines de cette interface
			if cell_idx in grid['cells']:
				neighbors = grid['cells'][cell_idx]['border_neighbors'][grid_interface_name]
			else:
				neighbors = []
			
			# Si pas de voisins, passer à l'interface suivante
			if not neighbors:
				continue
			
			# Initialiser les accumulateurs pour cette interface
			weighted_sum = np.zeros(num_components) if is_vector and component is None else 0.0
			total_weight = 0.0
			neighbor_count = 0
			
			# Traiter uniquement les particules voisines pré-calculées
			for vtk_idx, particle_idx in neighbors:
				# Position et attribut de la particule
				particle_pos = np.array(points[particle_idx])
				particle_attr = sample_attr[particle_idx]
				
				# Calculer la distance entre la particule et l'interface
				if dimensions == '2D':
					# En 2D, utiliser les 2 premières coordonnées (selon le plan)
					distance = np.sqrt(np.sum((particle_pos[:2] - interface_pos)**2))
				else:  # 3D
					# En 3D, utiliser les 3 coordonnées
					distance = np.sqrt(np.sum((particle_pos - interface_pos)**2))
				
				# Appliquer le kernel SPH si la distance est dans le rayon
				if distance <= kernel_radius:
					# Calculer le poids SPH
					weight = W(distance, kernel_radius)
					
					# Ajouter la contribution de cette particule
					if is_vector:
						if component is not None:
							weighted_sum += weight * particle_attr[component]
						else:
							weighted_sum += weight * particle_attr
					else:
						weighted_sum += weight * particle_attr
					
					total_weight += weight
					neighbor_count += 1
			
			# Calculer la moyenne pondérée si des particules influencent cette interface
			if neighbor_count > 0 and total_weight > 0:
				if dimensions == '2D':
					grid_values[interface_type][i, j] = weighted_sum / total_weight
					particle_counts[interface_type][i, j] = neighbor_count
				else:  # 3D
					grid_values[interface_type][i, j, k] = weighted_sum / total_weight
					particle_counts[interface_type][i, j, k] = neighbor_count
	
	# Ajouter les valeurs et métadonnées à l'objet grid
	for interface_type, key in value_keys.items():
		if interface_type in grid_values:  # Vérifier que l'interface existe
			grid[key] = grid_values[interface_type]
			
			# Ajouter les informations sur les interfaces
			grid[f"{key}_info"] = {
				'attribute': attribute_name,
				'is_vector': is_vector,
				'num_components': num_components,
				'component': component,
				'particle_counts': particle_counts[interface_type],
				'kernel_radius': kernel_radius,
				'method': 'sph',
				'interface_type': interface_type
			}
	
	return grid


def spatial_derivative(grid, attribute_name, deriv_axis='x', component=None):
	"""
	Calcule la dérivée spatiale d'un attribut dans la grille.
	Fonctionne pour les grilles 2D et 3D.
	
	Args:
		grid: Grille issue de compute_grid_values avec valeurs aux interfaces
		attribute_name: Nom de l'attribut dont on veut calculer la dérivée
		deriv_axis: Axe de dérivation ('x', 'y' ou 'z')
		component: Composante de l'attribut vectoriel (None pour toutes)
	"""
	dimensions = grid['dimensions']
	suffix = f"_{component}" if component is not None else ""
	
	# Construire les noms d'attributs pour les interfaces
	if dimensions == '2D':
		value_keys = {
			'left': f"{attribute_name}{suffix}_left",
			'right': f"{attribute_name}{suffix}_right",
			'up': f"{attribute_name}{suffix}_up",
			'down': f"{attribute_name}{suffix}_down"
		}
		
		# Vérifier la compatibilité avec le plan
		if deriv_axis == 'y' and grid.get('plane') == 'xz':
			raise ValueError(f"L'axe de dérivation 'y' n'est pas dans le plan '{grid['plane']}'")
		elif deriv_axis == 'z' and grid.get('plane') == 'xy':
			raise ValueError(f"L'axe de dérivation 'z' n'est pas dans le plan '{grid['plane']}'")
		
	else:  # 3D
		value_keys = {
			'left': f"{attribute_name}{suffix}_left",
			'right': f"{attribute_name}{suffix}_right",
			'up': f"{attribute_name}{suffix}_up",
			'down': f"{attribute_name}{suffix}_down",
			'front': f"{attribute_name}{suffix}_front",
			'back': f"{attribute_name}{suffix}_back"
		}
	
	# Calculer la dérivée selon l'axe demandé
	if deriv_axis == 'x':
		if value_keys['left'] not in grid or value_keys['right'] not in grid:
			raise ValueError(f"Les attributs d'interface '{value_keys['left']}' ou '{value_keys['right']}' n'existent pas.")
		
		# Récupérer directement la différence entre interfaces
		derivative = (grid[value_keys['right']] - grid[value_keys['left']]) / grid['dx']
		
	elif deriv_axis == 'y':
		if value_keys['down'] not in grid or value_keys['up'] not in grid:
			raise ValueError(f"Les attributs d'interface '{value_keys['down']}' ou '{value_keys['up']}' n'existent pas.")
		
		# Calculer directement la différence
		derivative = (grid[value_keys['up']] - grid[value_keys['down']]) / grid['dy']
		
	elif deriv_axis == 'z':
		if dimensions == '3D':
			if value_keys['back'] not in grid or value_keys['front'] not in grid:
				raise ValueError(f"Les attributs d'interface '{value_keys['back']}' ou '{value_keys['front']}' n'existent pas.")
			
			# Calculer directement la différence
			derivative = (grid[value_keys['front']] - grid[value_keys['back']]) / grid['dz']
		else:  # 2D avec plan xz ou yz
			# Pour 2D, nous utilisons les interfaces down/up pour l'axe z
			if grid['plane'] in ['xz', 'yz']:
				# Dans ce cas, z correspond aux interfaces haut/bas
				derivative = (grid[value_keys['up']] - grid[value_keys['down']]) / grid['dz']
			else:
				raise ValueError(f"L'axe de dérivation 'z' n'est pas dans le plan '{grid['plane']}'")
	else:
		raise ValueError(f"Axe de dérivation '{deriv_axis}' non valide")
	
	# Gestion des transpositions selon la dimension
	if dimensions == '2D':
		# En 2D, déterminer quelles dimensions doivent être transposées
		if derivative.ndim >= 2:
			# Transposer seulement les deux premières dimensions (i, j)
			if derivative.ndim == 2:  # Pour scalaires en 2D
				derivative = derivative.T
			elif derivative.ndim == 3:  # Pour vecteurs en 2D
				# Réorganiser pour transposer i, j tout en conservant la dimension des composantes
				derivative = np.transpose(derivative, (1, 0, 2))
	else:  # 3D
		# En 3D, transposer si nécessaire (normalement pas besoin si les dimensions sont cohérentes)
		pass
	
	# Générer une clé pour stocker la dérivée dans la grille
	comp_str = f"_{component}" if component is not None else ""
	deriv_key = f"d{attribute_name}{comp_str}_d{deriv_axis}"
	
	# Ajouter la dérivée et les métadonnées
	grid[deriv_key] = derivative
	grid[f"{deriv_key}_info"] = {
		'attribute': attribute_name,
		'component': component,
		'axis': deriv_axis,
		'method': 'finite_volume'
	}
	
	return derivative