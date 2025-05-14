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


def fit_ghe_model(slices, y_min, plot=False, save=False):
	"""
	Fit the GHE model to velocity profiles with U_0 calculated from central flow.
	
	Args:
		slices (list): List of data slices
		y_min (float): Minimum y-value for plotting
		plot (bool): If True, generates a plot
		save (bool): If True, saves the plot
		
	Returns:
		tuple: (Optimal delta parameter, Optimal gamma parameter)
	"""
	# Flatten all slices into single arrays
	u_flat = np.concatenate([slice['values'] for slice in slices])
	y_flat = np.concatenate([slice['coordinates'] for slice in slices])
	
	# Sort by y position for cleaner plotting
	sort_idx = np.argsort(y_flat)
	u_flat_sorted = u_flat[sort_idx]
	y_flat_sorted = y_flat[sort_idx]
	
	# Calculate U_0 from particles near the center (y ≈ 0)
	# Define "near center" as particles within a small range around y=0
	center_range = 0.1  # Ajustez cette valeur selon l'échelle de vos données
	center_mask = np.abs(y_flat) < center_range
	
	if np.sum(center_mask) > 0:
		# Calculer la vitesse moyenne au centre
		U_0 = np.mean(u_flat[center_mask])
		print(f"U_0 calculated from {np.sum(center_mask)} central particles: {U_0} m/s")
	else:
		# Fallback si aucune particule n'est suffisamment proche du centre
		U_0 = 7
		print(f"No particles found within {center_range} of center, using default U_0 = {U_0} m/s")
	
	# Define fitting function
	def fit_func(y, delta, gamma):
		u_model = compute_u_ghe(y, U_0, delta, gamma)[0]
		return u_model
	
	# Initial parameter guess
	initial_guess = [0.01, 0.5]
	
	# Fit using all sorted data points (pas de filtrage)
	popt, pcov = curve_fit(fit_func, y_flat_sorted, u_flat_sorted, p0=initial_guess)
	delta_opt, gamma_opt = popt
	perr = np.sqrt(np.diag(pcov))
	
	# Compute fitted model
	y_plot = np.linspace(y_min, -y_min, 1000)
	u_fitted, u_laminar, u_turbulent = compute_u_ghe(y_plot, U_0, delta_opt, gamma_opt)
	
	print(f"Optimal param: delta = {delta_opt} +- {perr[0]}, gamma = {gamma_opt} +- {perr[1]}")

	if plot:
		configure_latex()
		plt.figure(figsize=(6.7, 5))
		
		# Afficher toutes les données
		plt.scatter(y_flat_sorted, u_flat_sorted, s=7, alpha=0.7, color='blue', label='SPH data')
		
		# Mettre en évidence les particules utilisées pour calculer U_0
		if np.sum(center_mask) > 0:
			plt.scatter(y_flat[center_mask], u_flat[center_mask], s=10, color='green', 
					label=f'Central particles')
		
		# Afficher le modèle ajusté
		plt.plot(y_plot, u_fitted, color='red', linewidth=2, label='Fitted model')
		
		plt.xlabel('Diameter y [m]')
		plt.ylabel('Velocity u(y) [m/s]')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.grid()
		
		if save:
			plt.savefig('Pictures/CH5_valid_test/turbulent/u_fit_model.pdf')
	
	return delta_opt, gamma_opt




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
    
    U_0 = 5
    delta = 0.1
    gamma = 0.7

    u_ghe = compute_u_ghe(np.linspace(-0.5, 0.5, 100), U_0, delta, gamma)[0]
    plt.figure()
    plt.plot(np.linspace(-0.5, 0.5, 100), u_ghe)
    plt.show()


if __name__ == "__main__":
    main()