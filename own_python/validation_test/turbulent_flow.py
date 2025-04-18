#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for analysis of turbulent flow profiles and particle distributions.
"""

# Standard libraries
import os
import sys

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, curve_fit
from scipy.integrate import simpson
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points

# Local imports
sys.path.append(os.getcwd())

# Style configuration
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=MEDIUM_SIZE)    
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=MEDIUM_SIZE)   
plt.rc('ytick', labelsize=MEDIUM_SIZE)   
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)   


def configure_latex():
    """Configure matplotlib to use LaTeX for rendering text."""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


def compute_u_ghe(y_line, U_0, delta, gamma):
    """
    Compute velocity profile using the Generalized Hybrid Equation (GHE) model.
    
    Args:
        y_line (array): Array of y-positions
        U_0 (float): Maximum velocity
        delta (float): Boundary layer thickness parameter
        gamma (float): Weighting parameter between laminar and turbulent profiles
        
    Returns:
        tuple: (Full velocity profile, Laminar profile, Turbulent profile)
    """
    if len(y_line) % 2 == 0:  
        size_half = len(y_line) // 2
    else: 
        size_half = (len(y_line) + 1) // 2

    y_half = np.linspace(0, 0.5, size_half)
    
    U_L = U_0 * 4 * y_half * (1 - y_half)  # laminar
    U_T = U_0 * (1 - np.exp(1 - np.exp(y_half / delta)))  # turbulent
    U_GHE_half = gamma * U_T + (1 - gamma) * U_L  # hybrid model

    if len(y_line) % 2 == 0:
        y_full = np.concatenate((-y_half[::-1], y_half))
        U_full = np.concatenate((U_GHE_half, U_GHE_half[::-1]))
        U_L_full = np.concatenate((U_L, U_L[::-1]))
        U_T_full = np.concatenate((U_T, U_T[::-1]))
    else:
        y_full = np.concatenate((-y_half[::-1], y_half[1:]))
        U_full = np.concatenate((U_GHE_half, U_GHE_half[1:][::-1]))
        U_L_full = np.concatenate((U_L, U_L[1:][::-1]))
        U_T_full = np.concatenate((U_T, U_T[1:][::-1]))
    
    return U_full, U_L_full, U_T_full


def fit_ghe_model(u_all, y_all, y_min, plot=False, save=False):
    """
    Fit the GHE model to velocity profiles.
    
    Args:
        u_all (list): List of velocity arrays
        y_all (list): List of position arrays
        y_min (float): Minimum y-value for plotting
        plot (bool): If True, generates a plot
        save (bool): If True, saves the plot
        
    Returns:
        tuple: (Optimal delta parameter, Optimal gamma parameter)
    """
    # Flatten all slices into single arrays
    u_flat = np.concatenate([u_slice for u_slice in u_all])
    y_flat = np.concatenate([y_slice for y_slice in y_all])
    
    # Sort by y position for cleaner plotting
    sort_idx = np.argsort(y_flat)
    u_flat_sorted = u_flat[sort_idx]
    y_flat_sorted = y_flat[sort_idx]
    
    # Mean profile
    u_mean_exp = np.mean(u_all, axis=0)
    y_exp_single = y_all[0]
    
    # Maximum velocity for normalization
    U_0 = np.max(u_flat_sorted)
    
    # Define fitting function
    def fit_func(y, delta, gamma):
        u_model = compute_u_ghe(y, U_0, delta, gamma)[0]
        return u_model
    
    # Initial parameter guess
    initial_guess = [0.01, 0.5]
    
    # Fit using all sorted data points
    popt, pcov = curve_fit(fit_func, y_flat_sorted, u_flat_sorted, p0=initial_guess)
    delta_opt, gamma_opt = popt
    perr = np.sqrt(np.diag(pcov))
    
    # Compute fitted model
    u_fitted, u_laminar, u_turbulent = compute_u_ghe(u_flat_sorted, U_0, delta_opt, gamma_opt)
    y_plot = np.linspace(y_min, -y_min, len(u_fitted))
    
    print(f"Optimal param: delta = {delta_opt} +- {perr[0]}, gamma = {gamma_opt} +- {perr[1]}")

    if plot:
        configure_latex()
        plt.figure(figsize=(6.7, 5))
        plt.scatter(y_flat_sorted, u_flat_sorted, s=7, alpha=0.7, color='blue', label='SPH data')
        plt.plot(y_plot, u_fitted, color='red', label='Fitted model')
        plt.xlabel('Diameter y [m]')
        plt.ylabel('Velocity u(y) [m/s]')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        plt.legend(loc='best')
        
        if save:
            plt.savefig('Pictures/CH5_valid_test/turbulent/u_fit_single.pdf')
    
    return delta_opt, gamma_opt


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


def compute_reynolds_number(nu, D, U_0):
    """
    Compute Reynolds number and determine flow regime.
    
    Args:
        nu (float): Kinematic viscosity
        D (float): Characteristic length (e.g., diameter)
        U_0 (float): Maximum velocity
        
    Returns:
        int: Reynolds number
    """
    Re = int((U_0*D)/nu)
    print(f"Maximum velocity = {U_0}")
    print(f'Reynolds number Re = {Re}')

    if Re < 2300:
        print('Laminar flow')
    else:
        print('Turbulent flow')
        
    return Re


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


def remove_part(u_data, rho_data, y_data, min_part):
    """
    Remove excess particles to maintain consistent array sizes.
    
    Args:
        u_data (array): Velocity data
        rho_data (array): Density data
        y_data (array): Position data
        min_part (int): Minimum number of particles to keep
        
    Returns:
        tuple: (Filtered velocity, Filtered density, Filtered position)
    """
    idx_sorted = np.argsort(rho_data)
    idx_kept = np.sort(idx_sorted[:min_part])
    return u_data[idx_kept], rho_data[idx_kept], y_data[idx_kept]
    

def single_slice(fully_dev_vtk, y_min, y_max, x_pos, slice_width, plot=False, save=False):
    """
    Analyze particle data in a single slice over time.
    
    Args:
        fully_dev_vtk (list): List of VTK data objects for different time steps
        y_min, y_max (float): Y-coordinate boundaries
        x_pos (float): X-coordinate of the slice
        slice_width (float): Width of the slice
        plot (bool): If True, generates a plot
        save (bool): If True, saves the plot
        
    Returns:
        tuple: (Velocity data, Density data, Position data)
    """
    u_all = []
    rho_all = []
    y_all = []

    x_min = x_pos - slice_width
    x_max = x_pos + slice_width

    current_min_particles = float('inf')

    for i, single_vtk in enumerate(fully_dev_vtk):
        if i == 0:
            continue

        inside_mask, rectangle = find_particles_in_rectangle(single_vtk.points, x_min, y_min, x_max, y_max)
        projected_points, projected_attributes, vertical_line = project_particles(
            single_vtk, inside_mask, rectangle)
        
        u_data = projected_attributes['velocity'][:, 0]
        rho_data = projected_attributes['density']
        y_data = projected_points[:, 1]

        nb_part = len(y_data)
        
        # Ensure consistent number of particles across time steps
        if nb_part < current_min_particles:
            current_min_particles = nb_part
            
            for j in range(len(u_all)):
                if len(u_all[j]) > nb_part:
                    u_all[j], rho_all[j], y_all[j] = remove_part(u_all[j], rho_all[j], y_all[j], nb_part)
                    
                    assert len(u_all[j]) == nb_part, f"Error: after remove_part, j={j}, len(u_all[j])={len(u_all[j])}, should be {nb_part}"
        
        rho_all.append(rho_data)
        u_all.append(u_data)
        y_all.append(y_data)
        
        if len(u_data) > current_min_particles:
            u_all[-1], rho_all[-1], y_all[-1] = remove_part(u_data, rho_data, y_data, current_min_particles)
            
            assert len(u_all[-1]) == current_min_particles

    # Plotting
    if plot:
        configure_latex()
        u_min, u_max = [], []
        rho_min, rho_max = [], []
        
        # Velocity plot
        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], u_all[i], s=5)
            u_min.append(np.min(u_all[i]))
            u_max.append(np.max(u_all[i]))
            rho_min.append(np.min(rho_all[i]))
            rho_max.append(np.max(rho_all[i]))

        u_min_min = np.min(u_min)
        u_max_max = np.max(u_max)
        rho_min_min = np.min(rho_min)
        rho_max_max = np.max(rho_max)

        plt.ylabel('Velocity u(y) [m/s]')
        plt.xlabel('Distance y [m]')
        plt.ylim(u_min_min, u_max_max)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/u_single_x_{x_pos}.pdf')

        # Density plot
        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], rho_all[i], s=5)
        plt.ylim(990, 1010)
        plt.ylabel(r'Density $\rho$ [kg/$m^3$]')
        plt.xlabel('Distance y [m]')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/rho_single_x_{x_pos}.pdf')

    return u_all, rho_all, y_all


def multiple_slices(vtk_data, x_start, x_end, num_slices, y_min, y_max, 
                   slice_width, plot=False, save=False):
    """
    Analyze particle data in multiple slices along the x-axis.
    
    Args:
        vtk_data: VTK data containing particle information
        x_start, x_end (float): X-coordinate range for slices
        num_slices (int): Number of slices to analyze
        y_min, y_max (float): Y-coordinate boundaries
        slice_width (float): Width of each slice
        plot (bool): If True, generates plots
        save (bool): If True, saves the plots
        
    Returns:
        tuple: (Velocity data, Position data)
    """
    x_positions = np.linspace(x_start, x_end, num_slices)
    u_all = []
    rho_all = []
    y_all = []
    current_min_particles = float('inf')
        
    for x_pos in x_positions:
        x_min = x_pos - slice_width
        x_max = x_pos + slice_width
        
        inside_mask, rectangle = find_particles_in_rectangle(vtk_data.points, x_min, y_min, x_max, y_max)   
          
        projected_points, projected_attributes, vertical_line = project_particles(
            vtk_data, inside_mask, rectangle)
        
        u_data = projected_attributes['velocity'][:, 0]
        rho_data = projected_attributes['density']
        y_data = projected_points[:, 1]
        
        nb_part = len(y_data)
        
        # Ensure consistent number of particles across slices
        if nb_part < current_min_particles:
            current_min_particles = nb_part
            
            for j in range(len(u_all)):
                if len(u_all[j]) > nb_part:
                    u_all[j], rho_all[j], y_all[j] = remove_part(u_all[j], rho_all[j], y_all[j], nb_part)
                    
                    assert len(u_all[j]) == nb_part, f"Error: after remove_part, j={j}, len(u_all[j])={len(u_all[j])}, should be {nb_part}"
        
        u_all.append(u_data)
        rho_all.append(rho_data)
        y_all.append(y_data)

        if len(u_data) > current_min_particles:
            u_all[-1], rho_all[-1], y_all[-1] = remove_part(u_data, rho_data, y_data, current_min_particles)
            
            assert len(u_all[-1]) == current_min_particles, f"Error: after remove_part for last element, len(u_all[-1])={len(u_all[-1])}, should be {current_min_particles}"
    
    # Plotting
    if plot:
        configure_latex()
        u_min, u_max = [], []
        rho_min, rho_max = [], []

        # Velocity plot
        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], u_all[i], s=5)
            u_min.append(np.min(u_all[i]))
            u_max.append(np.max(u_all[i]))
        
        u_min_min = np.min(u_min)
        u_max_max = np.max(u_max)
        
        plt.ylabel('Velocity u(y) [m/s]')
        plt.xlabel('Distance y [m]')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/u_multiple.pdf')

        # Density plot
        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], rho_all[i], s=5)
            rho_min.append(np.min(rho_all[i]))
            rho_max.append(np.max(rho_all[i]))

        rho_min_min = np.min(rho_min)
        rho_max_max = np.max(rho_max)

        plt.ylabel(r'Density $\rho$ [kg/$m^3$]')
        plt.xlabel('Distance y [m]')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/rho_multiple.pdf')

        plt.show()

    return u_all, y_all


def visualize_results(vtk_data, inside_mask, projected_points, rectangle, vertical_line):
    """
    Visualize the projection process of particles onto a vertical line.
    
    Args:
        vtk_data: VTK data containing particle information
        inside_mask (array): Boolean mask indicating particles inside rectangle
        projected_points (array): Coordinates of projected points
        rectangle: Shapely rectangle object
        vertical_line: Shapely line object representing projection line
    """
    plt.figure(figsize=(6.7, 5))
    configure_latex()

    points = vtk_data.points
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5, label='All particles')
    inside_points = points[inside_mask]
    plt.scatter(inside_points[:, 0], inside_points[:, 1], s=3, c='red', label='Framed particles')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=3, c='green')
    
    x, y = rectangle.exterior.xy
    plt.plot(x, y, 'b-')
    x, y = vertical_line.xy
    plt.plot(x, y, 'g-', linewidth=2, label='Projected particles')
    
    plt.xlim((34.5, 36))
    plt.ylim(rectangle.bounds[1] - 0.3, 1.2*rectangle.bounds[3])
    plt.ylabel('Distance y [m]')
    plt.xlabel('Position x [m]')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('Pictures/CH5_valid_test/turbulent/euler_area.pdf')


def integrate_slice(Q_init, x_span, u_all, y_all, save=False):
    """
    Integrate velocity profiles to calculate flow rate and compare with initial value.
    
    Args:
        Q_init (float): Initial flow rate
        x_span (array): X-coordinates of slices
        u_all (list): List of velocity arrays
        y_all (list): List of position arrays
        save (bool): If True, saves the plot
        
    Returns:
        tuple: (All integrals, Filtered integrals)
    """
    integrals = []
    valid_indices = []
    
    for idx, (u, y) in enumerate(zip(u_all, y_all)):
        u, y = np.array(u), np.array(y)
        
        # Ensure y is sorted for integration
        if not np.all(np.diff(y) > 0):
            sort_idx = np.argsort(y)
            y, u = y[sort_idx], u[sort_idx]
        
        int_value = simpson(u, x=y)
        if int_value < 0:
            print(f"Negative integral at x = {x_span[idx]}")
            continue
            
        integrals.append(int_value)
        valid_indices.append(idx)
    
    filtered_x_span = x_span[valid_indices]
    print(f"Valid number of integrals: {len(integrals)} over {len(u_all)}")
    
    # Filter outliers
    p95 = np.percentile(integrals, 100)
    outlier_filter = [i for i, val in enumerate(integrals) if val <= p95]
    filtered_integrals = [integrals[i] for i in outlier_filter]
    filtered_mean = np.mean(filtered_integrals)
    filtered_x = filtered_x_span[outlier_filter]

    print(f'Mass flow error: {100*(Q_init - filtered_mean)/Q_init} %')
    
    # Plot
    plt.figure(figsize=(6.7, 5))
    
    plt.bar(filtered_x, filtered_integrals, alpha=0.7, color='seagreen')
    plt.hlines(Q_init, 0, np.max(x_span), color='red', linestyle='--', linewidth=2, label=r'$Q_{\text{init}}$')
    plt.hlines(filtered_mean, 0, np.max(x_span), color='darkgreen', linestyle='-', linewidth=2, label=r'$Q_{\text{SPH}} [m^3/s]$ ')
    plt.xlabel('Position x [m]')
    plt.ylabel(r'Mass flow rate $\int u(y)$ dy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('Pictures/CH5_valid_test/turbulent/massflow_conservation.pdf', dpi=300)
    
    return integrals, filtered_integrals


def center_line(x_span, u_all, save=False):
    """
    Analyze the centerline velocity along the x-axis.
    
    Args:
        x_span (array): X-coordinates of slices
        u_all (list): List of velocity arrays
        save (bool): If True, saves the plot
        
    Returns:
        tuple: (Middle velocity, Lower bound, Upper bound)
    """
    u_center = np.array([np.max(u) for u in u_all])
    
    u_middle = u_center[len(u_center)//2]
    lower_bound = u_middle * 0.95  # -5%
    upper_bound = u_middle * 1.05  # +5%
    
    plt.figure(figsize=(6.7, 5))
    
    # Add fill_between to show ±5% interval
    plt.fill_between(x_span, lower_bound, upper_bound, color='lightblue', alpha=0.5)
    
    # Reference line (middle value)
    plt.axhline(y=u_middle, color='r', linestyle='--', alpha=0.7, 
                label=r'Reference value with margin $\pm$ 5\%')
    
    # Original data points
    plt.scatter(x_span, u_center, s=8, color='red')
    
    plt.ylabel('Centerline velocity u [m/s]')
    plt.ylim(4, 10)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Position x [m]')
    plt.tight_layout()
    plt.legend()
    
    if save:
        plt.savefig('Pictures/CH5_valid_test/turbulent/velo_centerline.pdf', dpi=300)
    
    return u_middle, lower_bound, upper_bound


def main():
    """Main function for demonstration."""
    U_0 = 5
    delta = 0.1
    gamma = 0.7

    u_ghe = compute_u_ghe(np.linspace(-0.5, 0.5, 100), U_0, delta, gamma)[0]
    plt.figure()
    plt.plot(np.linspace(-0.5, 0.5, 100), u_ghe)
    plt.show()


if __name__ == "__main__":
    main()