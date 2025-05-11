import numpy as np
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import pyvista as pv


def configure_latex():
	"""Configure matplotlib to use LaTeX for rendering text."""
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')



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


def get_single_slice(vtk_files, attribute, plane, axis, fixed_coord=None, thickness=0.1, component=None):

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
	for vtk_file in vtk_files:
		points = vtk_file.points
		
		result = []
		
		# Create slice depending on plane and projected axis			
		if plane == 'xy':
			if axis == 'x':
				bounds = (target_min, min_coords[1], target_max, max_coords[1])
			else:  # axis == 'y'
				bounds = (min_coords[0], target_min, max_coords[0], target_max)
		elif plane == 'xz':
			if axis == 'x':
				bounds = (target_min, min_coords[2], target_max, max_coords[2])
			else:  # axis == 'z'
				bounds = (min_coords[0], target_min, max_coords[0], target_max)
		elif plane == 'yz':
			if axis == 'y':
				bounds = (target_min, min_coords[2], target_max, max_coords[2])
			else:  # axis == 'z'
				bounds = (min_coords[1], target_min, max_coords[1], target_max)
		
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
			
			result.append({
				'position': fixed_coord,
				'values': attr_values,
				'time': time_values,
				'coordinates': other_axis_coords
			})

	return result


def get_multiple_slices(vtk_file, attribute, plane, axis, along=None, thickness=0.1, component=None):

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
	
	result = []
	
	# Create slice depending on plane and projected axis
	for pos in positions:
		
		if plane == 'xy':
			if axis == 'x':
				bounds = (pos - thickness, min_coords[1], pos + thickness, max_coords[1])
			else:  # axis == 'y'
				bounds = (min_coords[0], pos - thickness, max_coords[0], pos + thickness)
		elif plane == 'xz':
			if axis == 'x':
				bounds = (pos - thickness, min_coords[2], pos + thickness, max_coords[2])
			else:  # axis == 'z'
				bounds = (min_coords[0], pos - thickness, max_coords[0], pos + thickness)
		elif plane == 'yz':
			if axis == 'y':
				bounds = (pos - thickness, min_coords[2], pos + thickness, max_coords[2])
			else:  # axis == 'z'
				bounds = (min_coords[1], pos - thickness, max_coords[1], pos + thickness)
		
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
			
			result.append({
				'position': pos,
				'values': attr_values,
				'coordinates': other_axis_coords
			})
	
	return result
	

def spatial_derivative(slices):

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
		
	return derivative, positions

def time_derivative(slices):

	mean_vals = []
	time = []

	for slice in slices:
		mean_vals.append(np.mean(slice['values']))
		time.append(slice['time'])

	mean_vals = np.array(mean_vals)
	times = np.array(time)
	
	# Forward finite difference method
	derivative = np.zeros_like(mean_vals)
	derivative[1:] = (mean_vals[:-1] - mean_vals[:1])/(times[:-1] - times[:1])
		
		
	return derivative, times
	
def plot_vtk(vtk_file, mask=None, attribute='velocity'):


	if mask is not None:
		points = vtk_file.points[mask]
	else:
		points = vtk_file.points

	point_cloud = pv.PolyData(points)

	if mask is not None:
		point_cloud.point_data[attribute] = vtk_file.point_data[attribute][mask]
	else:
		point_cloud.point_data[attribute] = vtk_file.point_data[attribute]

	plotter = pv.Plotter()
	plotter.add_mesh(
		point_cloud, 
		render_points_as_spheres=True, 
		scalar_bar_args={"title": f"{attribute}"},
		cmap="viridis", 
		scalars=attribute
	)

	plotter.add_axes()
	plotter.show()

'''
def time_analysis(vtk_data, attributes, dimensions, save_path,
						x, y_min, y_max,
						num_slices, slice_width, 
						remove=False, plot=False, save=False):
	
	x_min = x - slice_width
	x_max = x + slice_width
	
	current_min_particles = float('inf')
	
	# Dictionary to store all data - pre-allocate arrays for known size
	time_steps = len(vtk_data) - 1  # Skip first time step
	all_data = {
		"timesteps": np.arange(1, len(vtk_data)),  # Pre-generate timesteps
		"y": [None] * time_steps
	}
	
	# Initialize lists for each attribute
	for att in attributes:
		all_data[att] = [None] * time_steps
	
	# Process each time step
	for i, single_vtk in enumerate(vtk_data[1:], 1):  # Skip first time step directly
		# Find particles using vectorized operations
		
		inside_mask, rectangle = find_particles_in_rectangle(single_vtk.points, x_min, y_min, x_max, y_max)
		
		# Skip processing if no particles found
		if not np.any(inside_mask):
			continue
			
		projected_points, projected_attributes, vertical_line = project_particles(
			single_vtk, inside_mask, rectangle)
		
		# Extract y positions
		y_data = projected_points[:, 1]
		nb_part = len(y_data)
		
		# Skip if no particles projected
		if nb_part == 0:
			continue
		
		# Create data dictionary for this time step - avoid unnecessary copies
		i_adj = i - 1  # Adjusted index for storage
		
		# Handle minimum particle count update
		if nb_part < current_min_particles:
			current_min_particles = nb_part
			
			# Update all previous time steps to match new minimum - vectorized where possible
			for j in range(i_adj):
				if all_data["y"][j] is not None and len(all_data["y"][j]) > nb_part:
					# Create temporary dict for this time step
					temp_dict = {"y": all_data["y"][j]}
					for att in attributes:
						temp_dict[att] = all_data[att][j]
					
					# Apply filtering
					filtered_dict = remove_part(temp_dict, nb_part)
					
					# Update all_data with filtered values
					all_data["y"][j] = filtered_dict["y"]
					for att in attributes:
						all_data[att][j] = filtered_dict[att]
		
		# Store data for current time step
		if nb_part > current_min_particles:
			# Create temporary dict just once
			time_data = {"y": y_data}
			for att in attributes:
				if att == "velocity":
					time_data[att] = projected_attributes[att][:, 0]  # u_x only
				else:
					time_data[att] = projected_attributes[att]
					
			filtered_dict = remove_part(time_data, current_min_particles)
			all_data["y"][i_adj] = filtered_dict["y"]
			for att in attributes:
				all_data[att][i_adj] = filtered_dict[att]
		else:
			all_data["y"][i_adj] = y_data
			for att in attributes:
				if att == "velocity":
					all_data[att][i_adj] = projected_attributes[att][:, 0]  # u_x only
				else:
					all_data[att][i_adj] = projected_attributes[att]
	
	# Clean up None values
	valid_indices = [i for i, y in enumerate(all_data["y"]) if y is not None]
	all_data["timesteps"] = [all_data["timesteps"][i] for i in valid_indices]
	all_data["y"] = [all_data["y"][i] for i in valid_indices]
	for att in attributes:
		all_data[att] = [all_data[att][i] for i in valid_indices]
	
	# Store minimum particle count
	all_data["num_particles"] = current_min_particles
	
	# Plotting
	if plot:
		configure_latex()
		
		# Dictionary to store min/max values for each attribute
		mins = {att: [] for att in attributes}
		maxs = {att: [] for att in attributes}
		
		# Calculate min/max for each attribute and time step - use numpy vectorized operations
		for i in range(len(all_data["y"])):
			for att in attributes:
				if len(all_data[att][i]) > 0:  # Avoid empty arrays
					mins[att].append(np.min(all_data[att][i]))
					maxs[att].append(np.max(all_data[att][i]))
		
		# Create a plot for each attribute
		for att in attributes:
			if not mins[att] or not maxs[att]:  # Skip if no valid data
				continue
				
			plt.figure(figsize=(6.7, 5))
			
			# Plot data for each time step
			for i in range(len(all_data["y"])):
				plt.scatter(all_data["y"][i], all_data[att][i], s=5)
			
			# Calculate global min/max
			att_min_min = np.min(mins[att])
			att_max_max = np.max(maxs[att])
			
			# Set labels and limits
			if att == "velocity":
				plt.ylabel(f'Velocity u(y) {dimensions.get(att, "")}')
				plt.ylim(att_min_min, att_max_max)
			elif att == "density":
				plt.ylabel(fr'Density $\rho$ {dimensions.get(att, "")}')
				plt.ylim(990, 1010)
			elif att == "p_/_rho^2":
				plt.ylabel(fr'Pressure $p / \rho^2$ {dimensions.get(att, "")}')
				plt.ylim(att_min_min, att_max_max)
			else:
				plt.ylabel(f'{att.capitalize()} {dimensions.get(att, "")}')
				plt.ylim(att_min_min, att_max_max)
			
			plt.xlabel('Distance y [m]')
			plt.tight_layout()
			
			if save:
				plt.savefig(f'Pictures/CH5_valid_test/{save_path}{att}_single_x_{x}.pdf')
		
		if save:
			plt.close('all')  # Close all figures to free memory
		else:
			plt.show()
	
	return all_data
'''

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
		plt.figure(figsize=(12, 8))
		plt.bar(span, Q_v, alpha=0.7, color='royalblue', width=span[1] - span[0])
		plt.hlines(Q_v_th, span[0], span[-1], color='red', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_v_th:.6f} [m$^2$/s]')
		plt.hlines(Q_v_mean, span[0], span[-1], color='navy', linestyle='-', linewidth=2,
				label=fr'Mean Q: {Q_v_mean:.6f} [m$^2$/s] ({error_Q_v:.2f}%)')
		plt.xlabel('Position x [m]')
		plt.ylabel(r'Flow rate [m$^2$/s]')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		if save and savepath is not None:
			plt.savefig(f'{savepath}/flow_rate.pdf', dpi=300)

		# Mass flow
		plt.figure(figsize=(12, 8))
		plt.bar(span, Q_m, alpha=0.7, color='green', width=span[1] - span[0])
		plt.hlines(Q_m_th, span[0], span[-1], color='red', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_m_th:.6f} [Kg/s]')
		plt.hlines(Q_m_mean, span[0], span[-1], color='navy', linestyle='-', linewidth=2,
				label=fr'Mean Q: {Q_m_mean:.6f} [Kg/s] ({error_Q_m:.2f}%)')
		plt.xlabel('Position x [m]')
		plt.ylabel(r'Mass flow [Kg/s]')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		if save and savepath is not None:
			plt.savefig(f'{savepath}/mass_flow.pdf', dpi=300)


	return Q_v, Q_m


def analyze_particle_distribution(vtk_data, x_slice, delta_x=0.1, n_bins=50, plot=True):
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
		fig, ax = plt.subplots(figsize=(6.7, 5))
		ax.bar(bin_centers, counts, width=bin_width*0.9, alpha=0.7, color='steelblue')
		
		if len(high_density_regions) > 0:
			for y_pos in high_density_regions:
				ax.axvline(x=y_pos, color='red', linestyle='--', alpha=0.5)
		
		if len(low_density_regions) > 0:
			for y_pos in low_density_regions:
				ax.axvline(x=y_pos, color='orange', linestyle='--', alpha=0.5)
				
		ax.set_xlabel('Position y')
		ax.set_ylabel('Number of particles')
		ax.set_title(f'Particle distribution at x = {x_slice} +- {delta_x/2}')
		ax.grid(True, alpha=0.3)
		ax.legend()
		plt.tight_layout()
	
	return slice_points, (high_density_regions, low_density_regions, bin_centers)

