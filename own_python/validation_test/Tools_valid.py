import numpy as np
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

def configure_latex():
	"""Configure matplotlib to use LaTeX for rendering text."""
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')


def find_particles_in_rectangle(points, x_min, y_min, x_max, y_max):
	"""
	Find particles within a rectangular region.
	
	Args:
		points (array): Array of particle positions
		x_min, y_min (float): Lower-left corner coordinates
		x_max, y_max (float): Upper-right corner coordinates
		
	Returns:
		tuple: (Boolean mask of particles inside rectangle, Rectangle object)
	"""
	rectangle = box(x_min, y_min, x_max, y_max)
	inside_mask = np.zeros(len(points), dtype=bool)
	
	for i, point in enumerate(points):
		shapely_point = Point(point[0], point[1])
		inside_mask[i] = rectangle.contains(shapely_point)
	
	return inside_mask, rectangle


def project_particles(vtk_data, mask, rectangle):
	"""
	Project particles onto a vertical line in the middle of a rectangle.
	
	Args:
		vtk_data: VTK data containing particle information
		mask (array): Boolean mask indicating particles to project
		rectangle: Shapely rectangle object
		
	Returns:
		tuple: (Projected points, Projected attributes, Vertical line)
	"""
	# Particles inside the rectangle
	points = vtk_data.points
	inside_points = points[mask]
	
	min_x, min_y, max_x, max_y = rectangle.bounds
	middle_x = (min_x + max_x) / 2
	vertical_line = LineString([(middle_x, min_y), (middle_x, max_y)])
	
	# Project (orthogonally) the particles on the vertical line
	projected_points = np.zeros_like(inside_points)
	for i, point_coords in enumerate(inside_points):
		shapely_point = Point(point_coords[0], point_coords[1])
		_, projected = nearest_points(shapely_point, vertical_line)
		
		projected_points[i, 0] = projected.x
		projected_points[i, 1] = projected.y

		# Keep the z coordinate if it exists
		if inside_points.shape[1] > 2:
			projected_points[i, 2] = point_coords[2] 
	
	# Extract and project attributes
	projected_attributes = {}
	point_arrays = list(vtk_data.point_data.keys())
	
	for key in point_arrays:
		values = vtk_data[key]
		projected_attributes[key] = values[mask]
	
	# Sort the projected points by y coordinate
	if len(projected_points) > 0:
		sort_indices = np.argsort(projected_points[:, 1])
		projected_points = projected_points[sort_indices]
		
		for key in projected_attributes:
			projected_attributes[key] = projected_attributes[key][sort_indices]
	
	return projected_points, projected_attributes, vertical_line


def remove_part(slice_data, min_part, sort_key="density"):
	"""
	Remove excess particles to maintain consistent array sizes.
	
	Args:
		slice_data (dict): Dictionary containing data arrays for a slice
		min_part (int): Minimum number of particles to keep
		sort_key (str): Key to use for sorting (default: "density")
	
	Returns:
		dict: Filtered data dictionary
	"""
	# Sort indices based on the specified attribute
	idx_sorted = np.argsort(slice_data[sort_key])
	idx_kept = np.sort(idx_sorted[:min_part])
	
	# Apply filtering to all arrays in the dictionary
	filtered_dict = {}
	for key, value in slice_data.items():
		filtered_dict[key] = value[idx_kept]
	
	return filtered_dict
	
def single_slice(vtk_data, attributes, dimensions, save_path,
						x, y_min, y_max,
						num_slices, slice_width, 
						remove=False, plot=False, save=False):
	"""
	Optimized version of single_slice function with:
	- Vectorized operations
	- Pre-allocation of arrays
	- Early skipping of unnecessary operations
	- Reduced data copying
	"""
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


def multiple_slices_2D(vtk_data, attributes, dimensions, save_path,
						xy_init, xy_final, 
						num_slices, slice_width, 
						remove=False, plot=False, save=False):
	"""
	Optimized version of multiple_slices_2D function with:
	- Vectorized operations
	- Pre-allocation of arrays
	- Early skipping of unnecessary operations
	- Reduced redundant calculations
	"""
	x_span = np.linspace(xy_init[0], xy_final[0], num_slices)
	y_min = xy_init[1]
	y_max = xy_final[1]

	current_min_particles = float('inf')

	# Pre-allocate storage arrays
	all_data = {
		"x_positions": x_span,
		"y": [None] * num_slices
	}

	# Initialize arrays for each attribute
	for att in attributes:
		all_data[att] = [None] * num_slices

	# Process each slice
	for idx, x in enumerate(x_span):
		x_min = x - slice_width
		x_max = x + slice_width
		
		inside_mask, rectangle = find_particles_in_rectangle(vtk_data.points, x_min, y_min, x_max, y_max)

		
		# Skip if no particles found
		if not np.any(inside_mask):
			continue
			
		projected_points, projected_attributes, vertical_line = project_particles(vtk_data, inside_mask, rectangle)
		
		# Extract y positions
		y_data = projected_points[:, 1]
		nb_part = len(y_data)
		
		if nb_part == 0:
			continue
			
		# Create single dict for current slice data
		slice_data = {"y": y_data}
		for att in attributes:
			if att == "velocity":
				slice_data[att] = projected_attributes[att][:, 0]  # u_x only
			else:
				slice_data[att] = projected_attributes[att]
		
		# Update minimum particle count
		if nb_part < current_min_particles:
			current_min_particles = nb_part
			
			# If remove is enabled, update all previous slices to match new minimum
			if remove:
				for j in range(idx):
					if all_data["y"][j] is not None and len(all_data["y"][j]) > current_min_particles:
						# Create temporary dict for this slice
						temp_dict = {"y": all_data["y"][j]}
						for att in attributes:
							temp_dict[att] = all_data[att][j]
						
						# Apply filtering
						filtered_dict = remove_part(temp_dict, current_min_particles)
						
						# Update all_data with filtered values
						all_data["y"][j] = filtered_dict["y"]
						for att in attributes:
							all_data[att][j] = filtered_dict[att]
		
		# Add this slice's data (filter if needed)
		if remove and nb_part > current_min_particles:
			filtered_dict = remove_part(slice_data, current_min_particles)
			all_data["y"][idx] = filtered_dict["y"]
			for att in attributes:
				all_data[att][idx] = filtered_dict[att]
		else:
			all_data["y"][idx] = slice_data["y"]
			for att in attributes:
				all_data[att][idx] = slice_data[att]

	# Clean up None values
	valid_indices = [i for i, y in enumerate(all_data["y"]) if y is not None]
	all_data["x_positions"] = all_data["x_positions"][valid_indices]
	all_data["y"] = [all_data["y"][i] for i in valid_indices]
	for att in attributes:
		all_data[att] = [all_data[att][i] for i in valid_indices]

	# Plotting
	if plot:
		configure_latex()
		
		for att in attributes:
			plt.figure(figsize=(6.7, 5))
			
			att_min, att_max = [], []
			for i in range(len(all_data["y"])):
				if len(all_data[att][i]) > 0:  # Check for non-empty arrays
					
					plt.scatter(all_data["y"][i], all_data[att][i], s=5)
					att_min.append(np.min(all_data[att][i]))
					att_max.append(np.max(all_data[att][i]))
			
			if not att_min or not att_max:  # Skip if no valid data
				plt.close()
				continue
				
			att_min_min = np.min(att_min)
			att_max_max = np.max(att_max)
			
			# Labels
			if att == "velocity":
				plt.ylabel(f'Velocity u(y) {dimensions.get(att, "")}')
			elif att == "p_/_rho^2":
				plt.ylabel(fr'Pressure $p / \rho^2$ {dimensions.get(att, "")}')
			elif att == "density":
				plt.ylabel(fr'Density $\rho$ {dimensions.get(att, "")}')
			else:
				plt.ylabel(f'{att.capitalize()} {dimensions.get(att, "")}')
			
			plt.xlabel('Distance y [m]')
			plt.tight_layout()
			
			if save:
				if att == "p_/_rho^2":
					plt.savefig(fr'Pictures/CH5_valid_test/{save_path}_p_rho2_multiple.pdf')
				else:
					plt.savefig(fr'Pictures/CH5_valid_test/{save_path}_{att}_multiple.pdf')
			
		if save:
			plt.close('all')  # Close all figures to free memory
		else:
			plt.show()

	return all_data


def integrate_slice(multiple_data,
					x_start, x_end,
					Q_init=0.18, rho_0=1000,
					plot=False,
					save=False):
	"""
	Optimized version of integrate_slice function with:
	- Vectorized operations
	- Reduced redundant calculations
	- Early skipping of invalid data
	- More efficient data handling
	"""
	# Validate input data
	if not multiple_data['velocity'] or len(multiple_data['velocity']) == 0:
		print("No velocity data to process")
		return [], []
		
	x_span = np.linspace(x_start, x_end, len(multiple_data['velocity']))
	y_min, y_max = np.min(multiple_data['y'][0]), np.max(multiple_data['y'][0])
	U_0 = multiple_data['velocity'][0][0]
	print(f'Numerical initial flow rate :{(y_max - y_min)* U_0}')

	# Pre-allocate arrays for results
	vol_flow_rates = []
	mass_flow_rates = []
	valid_indices = []

	# Calculate initial mass flow rate once
	Q_init = (y_max - y_min)* U_0
	mass_flow_init = Q_init * rho_0

	# Process each slice using vectorized operations where possible
	for idx, (u, y, rho) in enumerate(zip(multiple_data['velocity'], multiple_data['y'], multiple_data['density'])):
		# Skip early if data is empty
		if len(u) == 0 or len(y) == 0:
			continue
			
		try:
			# Convert to numpy arrays if not already (avoid redundant conversions)
			u = np.asarray(u)
			y = np.asarray(y)
			rho = np.asarray(rho)
			
			# Skip early if data is invalid
			if np.any(np.isnan(u)) or np.any(np.isnan(y)) or np.any(np.isnan(rho)):
				print(f"Invalid data at slice {idx}, position x = {x_span[idx]}")
				continue
				
			# Sort data by y-coordinate for integration - use argsort once
			sort_idx = np.argsort(y)
			y_sorted = y[sort_idx]
			u_sorted = u[sort_idx]
			rho_sorted = rho[sort_idx]
			
			# Calculate volumetric flow rate (m³/s per unit depth)
			vol_flow = np.trapezoid(u_sorted, x=y_sorted)
			
			# Skip negative flow rates (likely integration errors)
			if vol_flow < 0:
				print(f"Negative volumetric flow at x = {x_span[idx]}")
				continue
				
			# Calculate mass flow rate (kg/s per unit depth)
			# Pre-compute the product for mass flow integration
			mass_flow = np.trapezoid(rho_sorted * u_sorted, x=y_sorted)
			
			# Store valid results
			vol_flow_rates.append(vol_flow)
			mass_flow_rates.append(mass_flow)
			valid_indices.append(idx)
			
		except Exception as e:
			print(f"Error processing slice {idx} at x = {x_span[idx]}: {str(e)}")

	# Convert to numpy arrays once
	valid_indices = np.array(valid_indices)
	
	# Skip processing if no valid data
	if len(valid_indices) == 0:
		print("No valid data to process")
		return [], []
		
	filtered_x_span = x_span[valid_indices]
	vol_flow_rates = np.array(vol_flow_rates)
	mass_flow_rates = np.array(mass_flow_rates)

	# Calculate statistics once
	mean_vol_flow = np.mean(vol_flow_rates)
	mean_mass_flow = np.mean(mass_flow_rates)
	vol_flow_error = 100 * (Q_init - mean_vol_flow) / Q_init
	mass_flow_error = 100 * (mass_flow_init - mean_mass_flow) / mass_flow_init

	print(f"Valid slices: {len(valid_indices)} out of {len(multiple_data['velocity'])}")
	print(f"Mean volumetric flow rate: {mean_vol_flow:.6f} m³/s")
	print(f"Volumetric flow error: {vol_flow_error:.2f}%")
	print(f"Mean mass flow rate: {mean_mass_flow:.6f} kg/s")
	print(f"Mass flow error: {mass_flow_error:.2f}%")

	# Plot results - reuse common calculations
	if plot:
		x_max = np.max(x_span)
		bar_width = x_span[1]-x_span[0] if len(x_span) > 1 else 0.1
		
		# Volumetric flow rate
		plt.figure(figsize=(10, 6))
		plt.bar(filtered_x_span, vol_flow_rates, alpha=0.7, color='royalblue', width=bar_width)
		plt.hlines(Q_init, 0, x_max, color='red', linestyle='--', linewidth=2, 
				label=fr'Initial Q: {Q_init:.6f} m$^2$/s')
		plt.hlines(mean_vol_flow, 0, x_max, color='navy', linestyle='-', linewidth=2,
				label=f'Mean Q: {mean_vol_flow:.6f} m³/s ({vol_flow_error:.2f}%)')
		plt.xlabel('Position x [m]')
		plt.ylabel(r'Volumetric flow rate [m$^2$/s]')
		plt.title('Volumetric Flow Rate Conservation')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		if save:
			plt.savefig('Pictures/CH5_valid_test/turbulent/volumetric_flow_conservation.pdf', dpi=300)
			plt.close()  # Close to free memory
		
		# Mass flow rate
		plt.figure(figsize=(10, 6))
		plt.bar(filtered_x_span, mass_flow_rates, alpha=0.7, color='seagreen', width=bar_width)
		plt.hlines(mass_flow_init, 0, x_max, color='red', linestyle='--', linewidth=2,
				label=f'Initial mass flow: {mass_flow_init:.6f} kg/s')
		plt.hlines(mean_mass_flow, 0, x_max, color='darkgreen', linestyle='-', linewidth=2,
				label=f'Mean mass flow: {mean_mass_flow:.6f} kg/s ({mass_flow_error:.2f}%)')
		plt.xlabel('Position x [m]')
		plt.ylabel('Mass flow rate [kg/s]')
		plt.title('Mass Flow Rate Conservation')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		if save:
			plt.savefig('Pictures/CH5_valid_test/turbulent/mass_flow_conservation.pdf', dpi=300)
			plt.close()  # Close to free memory
		else:
			plt.show()

	return Q_init


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

