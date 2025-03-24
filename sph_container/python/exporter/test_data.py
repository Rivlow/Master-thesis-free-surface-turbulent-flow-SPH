import pyvista as pv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configure project path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
from python.validation_test.turbulent_flow_pipe import *
from python.exporter.Tools import *
from plot_vtk import *

def load_vtk_files(folder_path, start_idx=None, end_idx=None, step=1):
    """
    Load VTK files containing fluid particles only.
    Supports selecting a range of timesteps.
    """
    folder = Path(folder_path)
    vtk_paths = folder.glob('ParticleData_Fluid*.vtk')  # get only fluid particle files
    
    # Sort numerically by extracting the number from the filename
    def get_file_number(path):
        try:
            filename = path.stem  # Name without extension
            number_part = filename.split('_')[-1]
            return int(number_part)
        except (ValueError, IndexError):
            return 0  # Fallback
    
    # Sort by extracted number
    vtk_paths = sorted(vtk_paths, key=get_file_number)
    
    # Select subset of files if specified
    if start_idx is not None or end_idx is not None:
        vtk_paths = vtk_paths[start_idx:end_idx:step]
    else:
        vtk_paths = vtk_paths[::step]
    
    all_vtk = []
    for path in vtk_paths:
        try:
            print(f"Loading {path.name}...")
            mesh = pv.read(path)
            all_vtk.append(mesh)
        except Exception as e:
            print(f"Error loading {path.name}: {e}")
    
    if not all_vtk:
        print("No VTK files found.")
        return None
    
    return all_vtk

def compute_average_velocity_profile(vtk_files, x_section, y_min, y_max, n_points, neighbourhood, kernel='cubic'):
    """
    Calculates the average velocity profile by sampling along a vertical line
    at the specified x-position, across multiple timesteps.
    """
    # Create sampling line coordinates
    y_coords = np.linspace(y_min, y_max, n_points)
    x_coords = np.ones(n_points) * x_section
    
    # Arrays to store valid points and their velocities across all timesteps
    all_valid_points = {}  # Dictionary to store velocities at each y-position
    
    # Process each timestep
    for i, vtk in enumerate(vtk_files):
        print(f"Processing timestep {i+1}/{len(vtk_files)}...")
        
        # Compute profile for this timestep
        valid_y, u_sph = compute_u_sph(vtk, y_coords, x_coords, neighbourhood, kernel)
        
        if len(valid_y) > 0:
            # Store velocities for each valid position
            for j, y_val in enumerate(valid_y):
                # Round to nearest sampling point to handle floating point differences
                y_key = round(y_val, 6)
                
                if y_key not in all_valid_points:
                    all_valid_points[y_key] = []
                
                all_valid_points[y_key].append(u_sph[j])
    
    # Compile results only from points that have data in multiple timesteps
    # (This ensures robust statistics)
    valid_y_values = []
    vel_mean = []
    vel_std = []
    
    for y_key, velocities in all_valid_points.items():
        # Only include points with sufficient samples
        if len(velocities) >= max(3, len(vtk_files) // 5):  # At least 3 samples or 20% of timesteps
            valid_y_values.append(y_key)
            vel_mean.append(np.mean(velocities, axis=0))
            vel_std.append(np.std(velocities, axis=0))
    
    # Sort by y-position
    if valid_y_values:
        sort_idx = np.argsort(valid_y_values)
        valid_y_values = np.array(valid_y_values)[sort_idx]
        vel_mean = np.array(vel_mean)[sort_idx]
        vel_std = np.array(vel_std)[sort_idx]
    
    return valid_y_values, vel_mean, vel_std

def extract_profile_by_slices(vtk_files, x_section, y_min, y_max, n_slices, slice_width=0.05):
    """
    Extracts the average velocity profile by dividing the channel into horizontal slices.
    More robust for turbulent flows with complex structures.
    """
    # Define slices
    slice_height = (y_max - y_min) / n_slices
    slice_centers = np.linspace(y_min + slice_height/2, y_max - slice_height/2, n_slices)
    
    # Initialize arrays to store velocities for each slice
    velocities_by_slice = [[] for _ in range(n_slices)]
    
    # Process each timestep
    for i, vtk in enumerate(vtk_files):
        print(f"Processing timestep {i+1}/{len(vtk_files)} (slice method)...")
        
        # Extract coordinates and velocities
        points = vtk.points
        velocities = vtk['velocity']
        
        # Filter points near the desired x-section
        x_mask = np.abs(points[:, 0] - x_section) <= slice_width/2
        
        if np.any(x_mask):
            filtered_points = points[x_mask]
            filtered_velocities = velocities[x_mask]
            
            # Assign points to slices
            y_values = filtered_points[:, 1]
            for j in range(n_slices):
                slice_min = y_min + j * slice_height
                slice_max = slice_min + slice_height
                
                # Find points in this slice
                mask = (y_values >= slice_min) & (y_values < slice_max)
                if np.any(mask):
                    velocities_by_slice[j].extend(filtered_velocities[mask])
    
    # Calculate statistics for each slice
    mean_velocities = []
    std_velocities = []
    valid_slice_centers = []
    
    for i in range(n_slices):
        if velocities_by_slice[i]:
            slice_velocities = np.array(velocities_by_slice[i])
            valid_slice_centers.append(slice_centers[i])
            mean_velocities.append(np.mean(slice_velocities, axis=0))
            std_velocities.append(np.std(slice_velocities, axis=0))
    
    # Convert to numpy arrays
    if valid_slice_centers:
        valid_slice_centers = np.array(valid_slice_centers)
        mean_velocities = np.array(mean_velocities)
        std_velocities = np.array(std_velocities)
    
    return valid_slice_centers, mean_velocities, std_velocities

def plot_mean_velocity_profile(y, vel_mean, vel_std=None, component='x', normalize=True, U_0=5.0):
    """
    Plots the mean velocity profile with optional error bands.
    Normalizes by the inlet velocity U_0 instead of the maximum velocity.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract velocity component
    if vel_mean.ndim > 1:  # Multi-dimensional velocity data
        if component == 'x':
            velocities = vel_mean[:, 0]
            std_devs = vel_std[:, 0] if vel_std is not None else None
        elif component == 'y':
            velocities = vel_mean[:, 1]
            std_devs = vel_std[:, 1] if vel_std is not None else None
        elif component == 'magnitude':
            velocities = np.linalg.norm(vel_mean, axis=1)
            std_devs = np.linalg.norm(vel_std, axis=1) if vel_std is not None else None
    else:  # Already extracted component
        velocities = vel_mean
        std_devs = vel_std
    
    # Normalize by inlet velocity U_0 if requested
    if normalize and len(velocities) > 0:
        velocities = velocities / U_0
        if std_devs is not None:
            std_devs = std_devs / U_0
    
    # Plot mean velocity profile
    ax.plot(y, velocities, 'o-', label='Mean velocity profile')
    
    # Add error bands (±1 standard deviation)
    if std_devs is not None:
        ax.fill_between(y, velocities - std_devs, velocities + std_devs, 
                        alpha=0.3, label='±1 standard deviation')
    
    ax.set_xlabel('Channel position y')
    ax.set_ylabel(f'{"Normalized " if normalize else ""}Velocity $u/U_0$ ({component}-component)')
    ax.grid(True)
    ax.legend()
    ax.set_title('Mean Velocity Profile Across Multiple Timesteps')
    
    return fig

def compute_fedoseyev_profiles(y_norm, delta=0.05, gamma=0.01):
    """
    Computes the theoretical velocity profiles according to Fedoseyev's theory.
    
    Parameters:
    - y_norm: normalized positions (0 at wall, 1 at centerline)
    - delta: boundary layer parameter for turbulent solution
    - gamma: weighting coefficient between laminar and turbulent solutions
    
    Returns:
    - Dictionary with laminar, turbulent, and combined (GHE) solutions
    """
    # Laminar (parabolic) solution
    u_laminar = 4 * y_norm * (1 - y_norm)
    
    # Turbulent (super-exponential) solution
    u_turbulent = 1 - np.exp(1 - np.exp(y_norm/delta))
    
    # Combined GHE solution
    u_ghe = gamma * u_turbulent + (1 - gamma) * u_laminar
    
    return {
        'laminar': u_laminar,
        'turbulent': u_turbulent,
        'ghe': u_ghe
    }

def compare_with_fedoseyev(y_data, vel_data, delta=0.05, gamma=0.29, normalize=True, U_0=5.0):
    """
    Compares measured velocity profile with Fedoseyev's analytical solutions.
    Uses inlet velocity U_0 for normalization.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize the data
    if normalize:
        # Find the center of the channel
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_center = (y_min + y_max) / 2
        y_half_width = (y_max - y_min) / 2
        
        # Normalize positions (0 at wall, 1 at center)
        y_norm = np.abs(y_data - y_center) / y_half_width
        
        # Ensure y_norm is between 0 and 1
        y_norm = 1 - y_norm  # 0 at wall, 1 at center
        
        # Normalize velocities by inlet velocity U_0
        vel_norm = vel_data / U_0
    else:
        y_norm = y_data
        vel_norm = vel_data
    
    # Sort data by normalized position
    sort_idx = np.argsort(y_norm)
    y_norm = y_norm[sort_idx]
    vel_norm = vel_norm[sort_idx]
    
    # Compute theoretical profiles
    # Note: Fedoseyev's profiles are already normalized to max velocity of 1
    # We may need to scale them by a factor if comparing to U_0-normalized data
    profiles = compute_fedoseyev_profiles(y_norm, delta, gamma)
    
    # Get max theoretical velocity to scale properly
    max_theory = max(np.max(profiles['laminar']), np.max(profiles['turbulent']), np.max(profiles['ghe']))
    scaling_factor = 1.0  # Adjust this if needed to match SPH data
    
    # Plot the data and theoretical profiles
    ax.plot(y_norm, vel_norm, 'o', label='SPH Simulation')
    ax.plot(y_norm, profiles['laminar'] * scaling_factor, '-', label='Laminar (Parabolic)')
    ax.plot(y_norm, profiles['turbulent'] * scaling_factor, '-', label='Turbulent (Super-exponential)')
    ax.plot(y_norm, profiles['ghe'] * scaling_factor, '-', label='GHE Combined Solution')
    
    ax.set_xlabel('Normalized position (0=wall, 1=center)')
    ax.set_ylabel('Normalized velocity $u/U_0$')
    ax.grid(True)
    ax.legend()
    ax.set_title('Comparison with Fedoseyev Analytical Solutions')
    
    return fig

def main():
    # Parameters
    r = 0.008  # Particle radius from your original simulation
    nu = 1e-6  # Kinematic viscosity
    neighbourhood = 4*r  # SPH interpolation radius
    U_0 = 5
    
    # Channel dimensions based on your simulation
    # Note: These should be adjusted based on your specific geometry
    channel_half_width = 0.5  # Half-width of the channel
    x_section_positions = [2]  # Multiple positions to extract profiles
    y_min, y_max = -channel_half_width, channel_half_width
    
    # Sampling parameters
    n_sampling_points = 50  # Points along vertical line
    n_slices = 30  # Number of horizontal slices
    
    # VTK loading parameters
    vtk_folder = "output_host/channel_curve_2D/DFSPH/5_m_s/angle_35"
    start_idx = 1000  # Start from a frame where flow is developed
    end_idx = 3000    # End frame
    frame_step = 200   # Take every 20th frame to reduce computation
    
    # Load VTK files
    print("Loading VTK files...")
    vtk_files = load_vtk_files(vtk_folder, start_idx, end_idx, frame_step)
    
    if not vtk_files:
        print("No VTK files loaded. Exiting.")
        return
    
    print(f"Loaded {len(vtk_files)} VTK files")
    
    # Visualize the last frame
    print("Visualizing the last timestep...")
    visualize_particles(vtk_files[-1], dimension='2D', plot_backend='pyplot', show=True)
    
    # Process each x-section
    for x_section in x_section_positions:
        print(f"\nAnalyzing velocity profile at x = {x_section}")
        
        # Method 1: Interpolation approach
        print("\nComputing velocity profile using SPH interpolation...")
        y_interp, vel_mean_interp, vel_std_interp = compute_average_velocity_profile(
            vtk_files, x_section, y_min, y_max, n_sampling_points, neighbourhood
        )
        
        if len(y_interp) > 0:
            # Plot the profile (interpolation method)
            fig_interp = plot_mean_velocity_profile(y_interp, vel_mean_interp, vel_std_interp, component='x', U_0=U_0)
            fig_interp.savefig(f"profile_interp_x{x_section:.1f}.png")
            
            # Extract the x-component for comparison with theory
            vel_x = vel_mean_interp[:, 0] if vel_mean_interp.ndim > 1 else vel_mean_interp
            
            # Compare with Fedoseyev's theory
            fig_compare = compare_with_fedoseyev(y_interp, vel_x, delta=0.05, gamma=0.95, U_0=U_0)

            fig_compare.savefig(f"fedoseyev_comparison_x{x_section:.1f}.png")
        else:
            print("Insufficient valid points for interpolation method.")
        
        # Method 2: Slice approach
        print("\nComputing velocity profile using horizontal slices...")
        y_slices, vel_mean_slices, vel_std_slices = extract_profile_by_slices(
            vtk_files, x_section, y_min, y_max, n_slices, slice_width=0.1
        )
        
        if len(y_slices) > 0:
            # Plot the profile (slice method)
            fig_slices = plot_mean_velocity_profile(y_slices, vel_mean_slices, vel_std_slices, component='x')
            fig_slices.savefig(f"profile_slices_x{x_section:.1f}.png")
            
            # Extract the x-component for comparison
            vel_x_slices = vel_mean_slices[:, 0] if vel_mean_slices.ndim > 1 else vel_mean_slices
            
            # Compare with Fedoseyev's theory (using slice method data)
            fig_compare_slices = compare_with_fedoseyev(y_slices, vel_x_slices, delta=0.05, gamma=0.29)
            fig_compare_slices.savefig(f"fedoseyev_comparison_slices_x{x_section:.1f}.png")
        else:
            print("Insufficient valid points for slice method.")
    
    # Calculate Reynolds number and print flow parameters
    D = 2 * channel_half_width  # Channel diameter
    U_0 = 5.0  # Inlet velocity
    Re = (U_0 * D) / nu
    print(f"\nFlow parameters:")
    print(f"Reynolds number: Re = {Re:.1f}")
    print(f"Channel width: D = {D} m")
    print(f"Inlet velocity: U_0 = {U_0} m/s")
    
    print("\nAnalysis complete. Figures saved.")
    plt.show()

if __name__ == "__main__":
    main()