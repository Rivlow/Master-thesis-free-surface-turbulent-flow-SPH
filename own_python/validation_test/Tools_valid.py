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
    

    x_min = x - slice_width
    x_max = x + slice_width
    
    current_min_particles = float('inf')
    
    # Dictionary to store all data
    all_data = {
        "timesteps": [],
        "y": []
    }
    
    # Initialize lists for each attribute
    for att in attributes:
        all_data[att] = []
    
    # Process each time step
    for i, single_vtk in enumerate(vtk_data):
        if i == 0:
            continue  # Skip first time step as in original code
        
        # Store time step index
        all_data["timesteps"].append(i)
        
        inside_mask, rectangle = find_particles_in_rectangle(single_vtk.points, x_min, y_min, x_max, y_max)
        projected_points, projected_attributes, vertical_line = project_particles(
            single_vtk, inside_mask, rectangle)
        
        # Extract y positions
        y_data = projected_points[:, 1]
        nb_part = len(y_data)
        
        # Create data dictionary for this time step
        time_data = {"y": y_data}
        for att in attributes:
            if att == "velocity":
                time_data[att] = projected_attributes[att][:, 0]  # u_x only
            else:
                time_data[att] = projected_attributes[att]
        
        # Ensure consistent number of particles across time steps
        if nb_part < current_min_particles:
            current_min_particles = nb_part
            
            # Update all previous time steps to match new minimum
            for j in range(len(all_data["y"])):
                if len(all_data["y"][j]) > nb_part:
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
                    
                    assert len(all_data["y"][j]) == nb_part, f"Error: after remove_part_dict, j={j}, len(all_data['y'][j])={len(all_data['y'][j])}, should be {nb_part}"
        
        # Add this time step's data (filter if needed)
        if nb_part > current_min_particles:
            filtered_dict = remove_part(time_data, current_min_particles)
            all_data["y"].append(filtered_dict["y"])
            for att in attributes:
                all_data[att].append(filtered_dict[att])
            
            assert len(all_data["y"][-1]) == current_min_particles, f"Error: after remove_part_dict for current time step, len(all_data['y'][-1])={len(all_data['y'][-1])}, should be {current_min_particles}"
        else:
            all_data["y"].append(time_data["y"])
            for att in attributes:
                all_data[att].append(time_data[att])
    
    # Store minimum particle count
    all_data["num_particles"] = current_min_particles
    
    # Plotting
    if plot:
        configure_latex()
        
        # Dictionary to store min/max values for each attribute
        mins = {}
        maxs = {}
        
        # Initialize min/max arrays for each attribute
        for att in attributes:
            mins[att] = []
            maxs[att] = []
        
        # Calculate min/max for each attribute and time step
        for i in range(len(all_data["y"])):
            for att in attributes:
                mins[att].append(np.min(all_data[att][i]))
                maxs[att].append(np.max(all_data[att][i]))
        
        # Create a plot for each attribute
        for att in attributes:
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
                # Keeping the original hard-coded limits for density
                plt.ylim(990, 1010)
            elif att == "p_/_rho^2":
                plt.ylabel(fr'Pressure $p / \rho^2$ {dimensions.get(att, "")}')
                plt.ylim(att_min_min, att_max_max)
            else:
                # For other attributes, use a generic label with the attribute name
                plt.ylabel(f'{att.capitalize()} {dimensions.get(att, "")}')
                plt.ylim(att_min_min, att_max_max)
            
            plt.xlabel('Distance y [m]')
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plt.tight_layout()
            
            if save:
                plt.savefig(f'Pictures/CH5_valid_test/{save_path}{att}_single_x_{x}.pdf')
        
        plt.show()
    
    return all_data


def multiple_slices_2D(vtk_data, attributes, dimensions, save_path,
						xy_init, xy_final, 
						num_slices, slice_width, 
						remove=False, plot=False, save=False):
	"""
	Process multiple 2D slices of VTK data and store results in a dictionary.

	Args:
		vtk_data: VTK data object
		attributes (list): List of attributes to extract
		xy_init (tuple): Initial (x,y) coordinates
		xy_final (tuple): Final (x,y) coordinates
		num_slices (int): Number of slices
		slice_width (float): Width of each slice
		remove (bool): Whether to remove excess particles to maintain consistent count
		plot (bool): Whether to plot results
		save (bool): Whether to save results
		sort_key (str): Key to use for sorting when removing particles (default: "density")

	Returns:
		dict: Dictionary containing data for all slices
	"""
	x_span = np.linspace(xy_init[0], xy_final[0], num_slices)

	current_min_particles = float('inf')

	# Dictionary to store all data
	all_data = {
		"x_positions": x_span,
		"y": []
	}

	# Initialize lists for each attribute
	for att in attributes:
		all_data[att] = []

	# Process each slice
	for x in x_span:
		x_min = x - slice_width
		x_max = x + slice_width
		y_min = xy_init[1]
		y_max = xy_final[1]
		
		inside_mask, rectangle = find_particles_in_rectangle(vtk_data.points, x_min, y_min, x_max, y_max)
		projected_points, projected_attributes, vertical_line = project_particles(vtk_data, inside_mask, rectangle)
		
		# Extract y positions
		y_data = projected_points[:, 1]
		nb_part = len(y_data)
		
		# Create data dictionary for this slice
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
				for j in range(len(all_data["y"])):
					if len(all_data["y"][j]) > current_min_particles:
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
						
						assert len(all_data["y"][j]) == current_min_particles, f"Error: after remove_part_dict, j={j}, len(all_data['y'][j])={len(all_data['y'][j])}, should be {current_min_particles}"
		
		# Add this slice's data (filter if needed)
		if remove and nb_part > current_min_particles:
			filtered_dict = remove_part(slice_data, current_min_particles)
			all_data["y"].append(filtered_dict["y"])
			for att in attributes:
				all_data[att].append(filtered_dict[att])
			
			assert len(all_data["y"][-1]) == current_min_particles, f"Error: after remove_part_dict for current slice, len(all_data['y'][-1])={len(all_data['y'][-1])}, should be {current_min_particles}"
		else:
			all_data["y"].append(slice_data["y"])
			for att in attributes:
				all_data[att].append(slice_data[att])
				

	# Plotting
	if plot:
		configure_latex()
		
		for att in attributes:
			plt.figure(figsize=(6.7, 5))
			
			att_min, att_max = [], []
			for i in range(len(all_data["y"])):
				plt.scatter(all_data["y"][i], all_data[att][i], s=5)
				att_min.append(np.min(all_data[att][i]))
				att_max.append(np.max(all_data[att][i]))
			
			att_min_min = np.min(att_min)
			att_max_max = np.max(att_max)
			
			# Étiquettes des axes
			if att == "velocity":
				plt.ylabel(f'Velocity u(y) {dimensions.get(att, "")}')
			elif att == "p_/_rho^2":
				plt.ylabel(fr'Pressure $p / \rho^2$ {dimensions.get(att, "")}')
			else:
				# Pour d'autres attributs, utiliser une étiquette générique avec le nom de l'attribut
				if att == "density":
					plt.ylabel(fr'Density $\rho$ {dimensions.get(att, "")}')
				else:
					plt.ylabel(f'{att.capitalize()} {dimensions.get(att, "")}')
			
			plt.xlabel('Distance y [m]')
			plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
			plt.tight_layout()
			
			if save:
				plt.savefig(f'Pictures/CH5_valid_test/{save_path}{att}_multiple.pdf')
		
		plt.show()

	return all_data

def integrate_slice(multiple_data,
					x_start, x_end,
					Q_init=0.18, rho_0=1000,
					save=False):
    
	x_span = np.linspace(x_start, x_end, len(multiple_data['velocity']))

	# Initialize arrays to store results
	vol_flow_rates = []
	mass_flow_rates = []
	valid_indices = []

	# Calculate initial mass flow rate
	mass_flow_init = Q_init * rho_0

	# Process each slice
	for idx, (u, y, rho) in enumerate(zip(multiple_data['velocity'], multiple_data['y'], multiple_data['density'])):
		try:
			# Convert to numpy arrays if not already
			u = np.array(u)
			y = np.array(y)
			rho = np.array(rho)
			
			# Skip if empty
			if len(u) == 0 or len(y) == 0:
				print(f"Empty data at slice {idx}, position x = {x_span[idx]}")
				continue
				
			# Sort data by y-coordinate for integration
			sort_idx = np.argsort(y)
			y = y[sort_idx]
			u = u[sort_idx]
			rho = rho[sort_idx]
			
			# Calculate volumetric flow rate (m³/s per unit depth)
			vol_flow = np.trapezoid(u, x=y)
			
			# Calculate mass flow rate (kg/s per unit depth)
			mass_flow = np.trapezoid(rho*u, x=y)
			
			# Skip negative flow rates (likely integration errors)
			if vol_flow < 0:
				print(f"Negative volumetric flow at x = {x_span[idx]}")
				continue
				
			# Store valid results
			vol_flow_rates.append(vol_flow)
			mass_flow_rates.append(mass_flow)
			valid_indices.append(idx)
			
		except Exception as e:
			print(f"Error processing slice {idx} at x = {x_span[idx]}: {str(e)}")

	# Convert to numpy arrays for easier manipulation
	valid_indices = np.array(valid_indices)
	filtered_x_span = x_span[valid_indices]
	vol_flow_rates = np.array(vol_flow_rates)
	mass_flow_rates = np.array(mass_flow_rates)

	# Calculate statistics
	mean_vol_flow = np.mean(vol_flow_rates)
	mean_mass_flow = np.mean(mass_flow_rates)
	vol_flow_error = 100 * (Q_init - mean_vol_flow) / Q_init
	mass_flow_error = 100 * (mass_flow_init - mean_mass_flow) / mass_flow_init

	print(f"Valid slices: {len(valid_indices)} out of {len(multiple_data['velocity'])}")
	print(f"Mean volumetric flow rate: {mean_vol_flow:.6f} m³/s")
	print(f"Volumetric flow error: {vol_flow_error:.2f}%")
	print(f"Mean mass flow rate: {mean_mass_flow:.6f} kg/s")
	print(f"Mass flow error: {mass_flow_error:.2f}%")

	# Plot results
	if save or True:  # Always plot for now
		# Volumetric flow rate
		plt.figure(figsize=(10, 6))
		plt.bar(filtered_x_span, vol_flow_rates, alpha=0.7, color='royalblue', width=x_span[1]-x_span[0])
		plt.hlines(Q_init, 0, np.max(x_span), color='red', linestyle='--', linewidth=2, 
					label=f'Initial Q: {Q_init:.6f} m³/s')
		plt.hlines(mean_vol_flow, 0, np.max(x_span), color='navy', linestyle='-', linewidth=2,
					label=f'Mean Q: {mean_vol_flow:.6f} m³/s ({vol_flow_error:.2f}%)')
		plt.xlabel('Position x [m]')
		plt.ylabel('Volumetric flow rate [m³/s]')
		plt.title('Volumetric Flow Rate Conservation')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		if save:
			plt.savefig('Pictures/CH5_valid_test/turbulent/volumetric_flow_conservation.pdf', dpi=300)
		
		# Mass flow rate
		plt.figure(figsize=(10, 6))
		plt.bar(filtered_x_span, mass_flow_rates, alpha=0.7, color='seagreen', width=x_span[1]-x_span[0])
		plt.hlines(mass_flow_init, 0, np.max(x_span), color='red', linestyle='--', linewidth=2,
					label=f'Initial mass flow: {mass_flow_init:.6f} kg/s')
		plt.hlines(mean_mass_flow, 0, np.max(x_span), color='darkgreen', linestyle='-', linewidth=2,
					label=f'Mean mass flow: {mean_mass_flow:.6f} kg/s ({mass_flow_error:.2f}%)')
		plt.xlabel('Position x [m]')
		plt.ylabel('Mass flow rate [kg/s]')
		plt.title('Mass Flow Rate Conservation')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		if save:
			plt.savefig('Pictures/CH5_valid_test/turbulent/mass_flow_conservation.pdf', dpi=300)
		
		plt.show()

	return vol_flow_rates, mass_flow_rates


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

