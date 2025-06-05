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

def plot_u_ghe():

	U_0 = 5
	delta = 0.1
	gamma = 0.7
	y = np.linspace(-0.5, 0.5, 100)
	u_ghe, u_l, u_t = compute_u_ghe(y, U_0, delta, gamma)

	u_ghe = u_ghe / U_0  
	u_l = u_l / U_0  
	u_t = u_t / U_0 

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.plot(y, u_ghe, 'b-', label='$U_{GHE}$', linewidth=2)
	ax.plot(y, u_l, 'orange', linestyle='--', label='$U_L$', linewidth=2)
	ax.plot(y, u_t, 'green', linestyle='--', label='$U_T$', linewidth=2)

	ax.set_xlabel('Diameter y/h [-]')
	ax.set_xlim(-0.5, 0.5)
	ax.set_xticks(np.arange(-0.5, 0.5+0.2, 0.2))

	ax.set_ylabel('Velocity u/$U_0$ [-]')
	ax.set_ylim(0, 1)
	ax.set_yticks(np.arange(0, 1+0.2, 0.2))


	plt.grid(True, alpha=0.3, ls='--')
	plt.legend()

	plt.tight_layout()
	plt.savefig('Pictures/CH6_valid_test/turbulent/u_ghe.pdf', dpi=300, bbox_inches='tight')
	plt.show()


def fit_ghe_model(slices, y_min, method='raw', plot=False, save=False, dy=0.01, center_range=0.1):
	"""
	Fit the GHE model to velocity profiles using either raw or binned data.
	
	Args:
		slices (list): List of data slices
		y_min (float): Minimum y-value for plotting
		method (str): 'raw' for raw data or 'binned' for averaged data
		plot (bool): If True, generates a plot
		save (bool): If True, saves the plot
		dy (float): Bin size for 'binned' method
		center_range (float): Range for central particles in 'raw' method
		
	Returns:
		tuple: (Optimal delta parameter, Optimal gamma parameter)
	"""
	# Flatten all slices into single arrays
	u_flat = np.concatenate([slice['values'] for slice in slices])
	y_flat = np.concatenate([slice['coordinates'] for slice in slices])
	
	if method == 'binned':
		# Binned averaging method
		y_span = np.arange(-y_min, y_min, dy)
		y_val = []
		u_val = []

		for y in y_span:
			indices = np.where(np.abs(y_flat - y) < dy / 2)[0]
			if len(indices) > 0:
				u_val.append(np.mean(u_flat[indices]))
				y_val.append(np.mean(y_flat[indices]))
		
		y_data = np.array(y_val)
		u_data = np.array(u_val)
		sort_idx = np.argsort(y_data)
		y_data = y_data[sort_idx]
		u_data = u_data[sort_idx]
		U_0 = np.max(u_data)
		print(f"U_0 taken as maximum velocity: {U_0} m/s")

	else:  # Raw data method
		y_data = y_flat
		u_data = u_flat
		sort_idx = np.argsort(y_data)
		y_data = y_data[sort_idx]
		u_data = u_data[sort_idx]
		
		# Calculate U_0 from central particles
		center_mask = np.abs(y_data) < center_range
		if np.sum(center_mask) > 0:
			U_0 = np.mean(u_data[center_mask])
			print(f"U_0 calculated from {np.sum(center_mask)} central particles: {U_0} m/s")
		else:
			U_0 = 7
			print(f"No central particles, using default U_0 = {U_0} m/s")

	# Define fitting function
	def fit_func(y, delta, gamma):
		u_model = compute_u_ghe(y, U_0, delta, gamma)[0]
		return u_model
	
	# Fit model to data
	initial_guess = [0.01, 0.5]
	popt, pcov = curve_fit(fit_func, y_data, u_data, p0=initial_guess)
	delta_opt, gamma_opt = popt
	perr = np.sqrt(np.diag(pcov))
	
	# Generate fitted curve
	y_plot = np.linspace(np.min(y_data), np.max(y_data), 1000)
	u_fitted, _, _ = compute_u_ghe(y_plot, U_0, delta_opt, gamma_opt)
	
	print(f"Optimal parameters: delta = {delta_opt:.4f} ± {perr[0]:.4f}, gamma = {gamma_opt:.4f} ± {perr[1]:.4f}")

	if plot:
		plt.figure(figsize=(12, 6))
		
		# Plot data points
		if method == 'binned':
			plt.scatter(y_data, u_data, s=30, alpha=0.7, color='blue', 
					label='SPH data')
		else:
			plt.scatter(y_data[::5], u_data[::5], s=7, alpha=0.3, color='blue', 
					label='SPH data')
		
		# Plot fitted model
		plt.plot(y_plot, u_fitted, 'r-', linewidth=2.5, label='Fitted GHE model')
		
		plt.xlabel('Diameter y [m]')
		plt.ylabel('Velocity u(y) [m/s]')
		plt.legend(loc='best')
		plt.grid(True, alpha=0.3, linestyle='--')
		
		# Set consistent axes limits
		plt.xlim(-y_min, y_min)
		plt.xticks(np.linspace(-y_min, y_min, 5))
		plt.ylim(0, 1.05*max(u_data))
		plt.yticks(np.linspace(0, 1.05*max(u_data), 5))
		
		plt.tight_layout()
		
		if save:
			fname = f'u_fit_model_{method}.pdf'
			plt.savefig(f'Pictures/CH6_valid_test/turbulent/{fname}', 
					dpi=5, bbox_inches='tight')
	
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
	
	plt.savefig('Pictures/CH6_valid_test/turbulent/euler_area.pdf')





def centreline_velocity(u_slices, save=False):
	u = []
	x_positions = []
	
	# iterate over timestep
	for u_insta in u_slices:
		u_inside = []
		# iterate over each slice
		for u_slice in u_insta:
			u_inside.append(u_slice['values'])
		u.append(u_inside)
	
	u = np.array(u)
	
	# Positions x (prendre depuis le premier timestep)
	x_positions = [u_slice['position'] for u_slice in u_slices[0]]
	
	# Time averaged velocity and standard deviation across slices
	u_centre = np.nanmean(u, axis=0)  # axis=0 pour moyenner sur le temps
	u_std = np.nanstd(u, axis=0)      # écart-type temporel
	
	# Moyenne globale
	u_mean_global = np.nanmean(u_centre)
	
	plt.figure(figsize=(12,6))
	
	# Zone d'incertitude (moyenne ± écart-type)
	plt.fill_between(x_positions, u_centre - u_std, u_centre + u_std, 
					alpha=0.3, color='lightblue', label=r"Deviation $\sigma(u)$ [m/s]")
	
	# Points de données originaux
	plt.scatter(x_positions, u_centre, s=12, color='blue', zorder=3)
	
	# Ligne de la moyenne globale
	plt.axhline(y=u_mean_global, color='red', linestyle='--', 
				alpha=0.8, label=fr'Mean $\mu(u)$= {u_mean_global:.2f} [m/s]')
	
	plt.ylabel('Centerline velocity u [m/s]')
	plt.xlabel('Position x [m]')
	plt.xlim(0,45)
	plt.xticks(np.linspace(0,45,5))
	plt.ylim(5.5, 7.5)
	plt.grid(True, alpha=0.3, ls="--")
	plt.legend(loc='best')
	plt.tight_layout()
	
	if save:
		plt.savefig('Pictures/CH6_valid_test/turbulent/velo_centreline.pdf', dpi=300, bbox_inches='tight')
	
	return u_centre, x_positions, u_std

def main():

	a = 1
	

if __name__ == "__main__":
	main()