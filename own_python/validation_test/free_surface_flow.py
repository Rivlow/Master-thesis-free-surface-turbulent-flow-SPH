#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for calculating and visualizing water heights
and hydraulic jumps.
"""

# Standard libraries
import os
import sys

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.optimize import fsolve
from scipy.interpolate import splev, splrep
from scipy import stats


# Local imports
sys.path.append(os.getcwd())
from own_python.write_scenes.Tools_scenes import *

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
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')



# Constants
cm = 1e-2  # centimeter in meters
m = 1      # meter
s = 1      # second
g = 9.81   # gravitational acceleration (m/s²)


# Fonctions hydrauliques
def compute_Fr(q, g, h):
	"""
	Calculate the Froude number.

	Args:
		q (float): Flow rate per unit width (m²/s)
		g (float): Gravitational acceleration (m/s²)
		h (float): Water height (m)

	Returns:
		float: Froude number
	"""
	return q / (np.sqrt(g) * h**(3/2))


def solve_height(q, g, x, z_b, H):
	"""
	Solve the water height equation.

	Args:
		q (float): Flow rate per unit width (m²/s)
		g (float): Gravitational acceleration (m/s²)
		x (float): Horizontal position (m)
		z_b (function): Bed profile function
		H (float): Total hydraulic head

	Returns:
		tuple: Water height solution and convergence information
	"""
	def eq(h):
		if h <= 0.001:
			return float('inf')
		return z_b(x) + h + q**2 / (2 * g * h**2) - H

	h0 = 0.1  # Initial guess
	return fsolve(eq, h0, full_output=True)


def conjugate_height(q, g, h):
	"""
	Calculate the conjugate height after a hydraulic jump.

	Args:
		q (float): Flow rate per unit width (m²/s)
		g (float): Gravitational acceleration (m/s²)
		h (float): Water height upstream of the jump (m)

	Returns:
		float: Conjugate water height (m)
	"""
	Fr = compute_Fr(q, g, h)
	return h * 0.5 * (np.sqrt(1 + 8 * Fr**2) - 1)


def solve_height_amont(q, g, x, z_b, H):
	"""
	Solve the upstream water height equation.

	Args:
		q (float): Flow rate per unit width (m²/s)
		g (float): Gravitational acceleration (m/s²)
		x (float): Horizontal position (m)
		z_b (function): Bed profile function
		H (float): Total hydraulic head upstream

	Returns:
		float: Upstream water height (m)
	"""
	# Équation: h³ + (z_b(x) - H)·h² + 0·h + q²/(2*g)
	a = z_b(x) - H
	b = 0
	c = q**2 / (2 * g)

	coeffs = [1, a, b, c]
	roots = np.roots(coeffs)

	real_positive_roots = np.array([root.real for root in roots if root.real > 0])

	if 10 < x <= 12:
		return np.min(real_positive_roots)
	elif x == 10:
		return real_positive_roots[0]
	else:
		return np.max(real_positive_roots)
	

def solve_height_aval(q, g, x, z_b, H):
	"""
	Solve the downstream water height equation.
	
	Args:
		q (float): Flow rate per unit width (m²/s)
		g (float): Gravitational acceleration (m/s²)
		x (float): Horizontal position (m)
		z_b (function): Bed profile function
		H (float): Total hydraulic head downstream
		
	Returns:
		float: Downstream water height (m)
	"""
	# Équation: h³ + (z_b(x) - H)·h² + 0·h + q²/(2*g)
	a = z_b(x) - H
	b = 0 
	c = q**2 / (2 * g)
	
	coeffs = [1, a, b, c]
	roots = np.roots(coeffs)
	
	real_positive_roots = np.array([root.real for root in roots if root.real > 0])

	if 10 < x <= 11:
		return np.min(real_positive_roots)
	elif x == 10:
		return real_positive_roots[0]
	else:
		return np.max(real_positive_roots)
	
	
def extract_water_height(vtk_file, plot=False, save=False):
	"""
	Extract water height from a VTK file.
	
	Args:
		vtk_file: VTK file containing the data
		plot (bool): If True, generates a plot
		save (bool): If True, saves the plot
		
	Returns:
		tuple: (particle positions, water heights, surface velocities)
	"""
	g = 9.81
	pos = np.array(vtk_file.points)
	
	# Only particles with y > 0
	positive_y_mask = pos[:, 1] > 0
	pos = pos[positive_y_mask]
	u = np.array(vtk_file.point_data['velocity'])[positive_y_mask]

	# Sort particles along x axis
	sorted_indices = np.argsort(pos[:, 0])
	sorted_points = pos[sorted_indices]
	sorted_velocity = u[sorted_indices]
	
	# Sample distance x in bins
	x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
	num_bins = 500
	bin_edges = np.linspace(x_min, x_max, num_bins + 1)
	
	# Highest point (y_max) on each bin and its velocity
	pos_surf = []
	u_surf = []
	
	for i in range(num_bins):
		bin_start, bin_end = bin_edges[i], bin_edges[i+1]
		mask = (sorted_points[:, 0] >= bin_start) & (sorted_points[:, 0] < bin_end)
		bin_points = sorted_points[mask]
		bin_velocity = sorted_velocity[mask]
		
		if len(bin_points) > 0:
			max_y_index = np.argmax(bin_points[:, 1])  # index of y_max
			u_surf.append(bin_velocity[max_y_index])
			pos_surf.append(bin_points[max_y_index])
	
	h = np.array(pos_surf)
	u_surf = np.array(u_surf)
	Fr = np.linalg.norm(u_surf, axis=1) / np.sqrt(g * (h[:, 1]))
	
	if plot:
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
		
		# Plot particle positions
		ax1.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.3, color='blue', label='Particles')
		ax1.scatter(h[:, 0], h[:, 1], s=20, color='red', label='Water height')
		ax1.set_xlabel('Position x [m]')
		ax1.set_ylabel('Height h(x) [m]')
		ax1.legend()
		ax1.grid(True)

		# Graphique du nombre de Froude
		ax2.scatter(h[:, 0], Fr, s=20, color='green', label='Froude number')
		ax2.set_xlabel('Position x [m]')
		ax2.set_ylabel('Froude number Fr [-]')
		ax2.legend()
		ax2.grid(True)

		plt.tight_layout()
		if save:
			plt.savefig('Pictures/CH5_valid_test/free_surface/water_height_velocity_sph.pdf', dpi=300)
		plt.show()

	return pos, h, u_surf


def compute_theoretical_water_height(Q_init=0.18):
	"""
	Calculate the theoretical water height.

	Args:
		U_0 (float): Initial flow velocity (m/s)

	Returns:
		tuple: (x positions, bed heights z, water heights h, Froude numbers Fr)
	"""
	g = 9.81 * (m/(s**2))

	# Parabola function (bed profile)
	z_b = lambda x: 0.2 - 0.05*((x-10)**2) if 8 <= x <= 12 else 0
	

	# Boundary conditions
	q = Q_init *(m**2/s)          # inlet mass flow
	h_aval = 33 * cm             # outlet water height

	# Compute critical height
	h_cr = (q**2/g)**(1/3)
	x_cr = 10 * m
	x_ressaut = 11.6657          # previously evaluated

	# Using Bernoulli, upstream/downstream total head
	H_amont = z_b(10) + h_cr + q**2/(2*g*h_cr**2)
	H_aval = 0 + h_aval + q**2/(2*g*h_aval**2)

	nb_elem = 2000
	x_amont = np.linspace(0, x_ressaut, nb_elem)
	x_aval = np.linspace(25, x_ressaut, nb_elem)

	# Topography
	z_aval = np.array([z_b(x) for x in x_aval])
	z_amont = np.array([z_b(x) for x in x_amont])

	# Compute water height from "z_b(x) + h(x) + (q²/2g)(1/h²(x)) = H_aval (or H_amont)"
	h_amont = np.array([solve_height_amont(q, g, x, z_b, H_amont) for x in x_amont])
	h_aval = np.array([solve_height_aval(q, g, x, z_b, H_aval) for x in x_aval])

	h_inter = np.linspace(h_amont[-1], h_aval[-1], nb_elem)
	x_inter = np.linspace(x_amont[-1], x_aval[-1], nb_elem)
	z_inter = np.array([z_b(x) for x in x_inter])

	# Froude numbers over whole distance
	Fr_amont = np.array([compute_Fr(q, g, h) for h in h_amont])
	Fr_aval = np.array([compute_Fr(q, g, h) for h in h_aval])
	Fr_inter = np.array([compute_Fr(q, g, h) for h in h_inter])

	# Conjugated heights h2 from h1(x_aval)
	h2_conj = np.array([conjugate_height(q, g, h1) for h1 in h_amont])

	# Concaténation des tableaux
	x_all = np.concatenate((x_amont, x_inter, x_aval))
	z_all = np.concatenate((z_amont, z_inter, z_aval))
	h_all = np.concatenate((h_amont, h_inter, h_aval))
	Fr_all = np.concatenate((Fr_amont, Fr_inter, Fr_aval))

	# Tri des tableaux par position x
	indices = np.argsort(x_all)
	x_all = x_all[indices]
	z_all = z_all[indices]
	h_all = h_all[indices]
	Fr_all = Fr_all[indices]

	return x_all, z_all, h_all, Fr_all

def annotate_hydraulic_regions(x_all, z_all, h_all, Fr_all, save=False):
	"""
	Plot water heights with annotations for different hydraulic regions.
	
	Args:
		x_all (array): x-coordinates along the channel
		z_all (array): bed elevations
		h_all (array): water heights
		Fr_all (array): Froude numbers
		save (bool): If True, saves the plot to a file
	"""
	# Constants
	x_cr = 10.0  # Critical point
	x_ressaut = 11.6657  # Hydraulic jump position
	
	# Create figure
	fig, ax = plt.subplots(figsize=(12, 7))
	
	# Plot bed profile and water surface
	ax.plot(x_all, z_all + h_all, 'b-', linewidth=2)
	ax.plot(x_all, z_all, 'k-', linewidth=2)
	
	# Fill water area
	#ax.fill_between(x_all, z_all, z_all + h_all, color='lightblue', alpha=0.5)
	
	# Mark critical point
	ax.plot(x_cr, z_all[np.abs(x_all - x_cr).argmin()] + 
		h_all[np.abs(x_all - x_cr).argmin()], 'ko', markersize=8, label='Critical point')
	
	# Mark hydraulic jump position
	ax.plot(x_ressaut, z_all[np.abs(x_all - x_ressaut).argmin()] + 
		h_all[np.abs(x_all - x_ressaut).argmin()], 'ro', markersize=8, label='Hydraulic jump')
	
	# Define region boundaries
	inlet_region = x_all < x_cr
	critical_region = (x_all >= x_cr) & (x_all < x_ressaut)
	outlet_region = x_all >= x_ressaut
	
	# Color regions (using alpha to maintain visibility of the water surface)
	ax.fill_between(x_all[inlet_region], 
				z_all[inlet_region], 
				z_all[inlet_region] + h_all[inlet_region] , 
				color='blue', label= 'Inlet region', alpha=0.2)
	
	ax.fill_between(x_all[critical_region], 
				z_all[critical_region], 
				z_all[critical_region] + h_all[critical_region], 
				color='red', label= 'Jump region',alpha=0.2)
	
	ax.fill_between(x_all[outlet_region], 
				z_all[outlet_region], 
				z_all[outlet_region] + h_all[outlet_region], 
				color='yellow', label= 'Outlet region',alpha=0.2)
	
	'''
	# Add annotations for regions
	ax.annotate('Inlet region', xy=(5, z_all[np.abs(x_all - 5).argmin()] + 
			h_all[np.abs(x_all - 5).argmin()] + 0.1), fontsize=16)
	
	ax.annotate('Critical region', xy=(x_cr + 0.5, z_all[np.abs(x_all - (x_cr + 0.5)).argmin()] + 
			h_all[np.abs(x_all - (x_cr + 0.5)).argmin()] + 0.1), fontsize=16)
	
	ax.annotate('Outlet region', xy=(x_ressaut + 2, z_all[np.abs(x_all - (x_ressaut + 2)).argmin()] + 
			h_all[np.abs(x_all - (x_ressaut + 2)).argmin()] + 0.1), fontsize=16)
	'''


	
	# Add annotation for x_cr
	h_cr = z_all[np.abs(x_all - x_cr).argmin()] + h_all[np.abs(x_all - x_cr).argmin()]
	ax.set_yticks([h_all[0], h_cr, h_all[-1]], [r'$h_{\text{inlet}}$', r'$h_{\text{cr}}$', r'$h_{\text{outlet}}$'])
	ax.set_xticks([x_cr, x_ressaut], [r'$x_{\text{cr}}$', r'$x_{\text{jump}}$'])
	
	# Add Roman numerals for flow regimes (similar to the image)
	ax.annotate('I', xy=(x_cr - 1, z_all[np.abs(x_all - (x_cr + 0.5)).argmin()] + 
			h_all[np.abs(x_all - (x_cr + 0.5)).argmin()] - 0.1), fontsize=20, 
			bbox=dict(boxstyle="circle", fc="lightblue", ec="blue", alpha=0.7))
	
	ax.annotate('II', xy=(x_cr + 0.5, z_all[np.abs(x_all - (x_cr + 0.5)).argmin()] + 
			h_all[np.abs(x_all - (x_cr + 0.5)).argmin()] - 0.1), fontsize=20, 
			bbox=dict(boxstyle="circle", fc="lightblue", ec="blue", alpha=0.7))
	
	ax.annotate('III', xy=(x_cr + 2, z_all[np.abs(x_all - (x_cr + 0.5)).argmin()] + 
			h_all[np.abs(x_all - (x_cr + 0.5)).argmin()] - 0.1), fontsize=20, 
			bbox=dict(boxstyle="circle", fc="lightblue", ec="blue", alpha=0.7))
	
	# Add vertical lines for region boundaries
	#ax.axvline(x=x_cr, color='red', linestyle='--', linewidth=1.5)
	#ax.axvline(x=x_ressaut, color='red', linestyle='--', linewidth=1.5)
	
	# Plot configuration
	ax.set_xlim(7, 13)
	ax.set_xlabel('Distance x [m]', fontsize=14)
	ax.set_ylabel('Height [m]', fontsize=14)
	ax.legend(loc='best')
	ax.grid(True, alpha=0.3)
	
	# Add Froude number subplot

	#plt.title('Hydraulic Flow Profile with Jump', fontsize=16)
	plt.tight_layout()
	
	if save:
		plt.savefig("Pictures/CH5_valid_test/free_surface/annotated_hydraulic_jump.pdf", dpi=300)
	
	return fig, ax

def plot_conjugate_height(x_all, z_all, h_all, Fr_all, save=False):
	"""
	Plot water heights with conjugate heights and mark known hydraulic jump location.
	
	Args:
		x_all (array): x-coordinates along the channel
		z_all (array): bed elevations
		h_all (array): water heights
		Fr_all (array): Froude numbers
		save (bool): If True, saves the plot to a file
	"""
	# Constants
	g = 9.81  # m/s²
	x_cr = 10.0  # Critical point (m)
	x_ressaut = 11.6657  # Known hydraulic jump position (m)
	
	# Create figure
	fig, ax = plt.subplots(figsize=(12, 7))
	
	# Find indices for critical point and hydraulic jump
	cr_idx = np.abs(x_all - x_cr).argmin()
	ressaut_idx = np.abs(x_all - x_ressaut).argmin()
	
	# Calculate conjugate heights for the entire region from x_cr to slightly past x_ressaut
	# First determine the range to calculate conjugate heights for
	start_idx = cr_idx-300
	print(start_idx)
	# End slightly past the hydraulic jump (20% more than the distance from critical to jump)
	extend_past_jump = int((ressaut_idx - cr_idx) * 0.2)+200
	end_idx = min(ressaut_idx + extend_past_jump, len(x_all) - 1)
	
	# Extract data for this region
	x_conj_region = x_all[start_idx:end_idx]
	z_conj_region = z_all[start_idx:end_idx]
	h_conj_region = h_all[start_idx:end_idx]
	Fr_conj_region = Fr_all[start_idx:end_idx]
	
	# Calculate conjugate heights
	h_conj = np.zeros_like(h_conj_region)
	for i, h in enumerate(h_conj_region):
		Fr = Fr_conj_region[i]
		h_conj[i] = h * 0.5 * (np.sqrt(1 + 8 * Fr**2) - 1)
	
	# Plot bed profile and water surface
	ax.plot(x_all, z_all + h_all, 'b-', linewidth=2, label='Water surface')
	ax.plot(x_all, z_all, 'k-', linewidth=2, label='Bed profile')
	
	# Fill water area
	ax.fill_between(x_all, z_all, z_all + h_all, color='lightblue', alpha=0.5)
	
	# Plot conjugate height curve
	ax.plot(x_conj_region, z_conj_region + h_conj, 'r--', linewidth=2, label='Conjugate height')
	
	# Mark hydraulic jump location with a colored point
	jump_height = h_all[ressaut_idx] + z_all[ressaut_idx]
	ax.plot(x_ressaut, jump_height, 'ro', markersize=10, label=f'Hydraulic jump')
	
	# Add a vertical line at the jump location
	#ax.axvline(x=x_ressaut, color='red', linestyle=':', linewidth=1.5)
		
	# Plot configuration
	ax.set_xlabel('Distance x [m]', fontsize=14)
	ax.set_xlim(7, 13)  # Focus on the relevant region
	ax.set_ylabel('Height [m]', fontsize=14)
	ax.legend(loc='best')
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	if save:
		plt.savefig("Pictures/CH5_valid_test/free_surface/conjugate_height.pdf", dpi=300)
	
	return fig, ax


def plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False):
	"""
	Plot water height.

	Args:
		Ly (float): Domain height
		x_th (array): Theoretical x positions
		z_th (array): Theoretical bed heights
		h_th (array): Theoretical water heights
		points (array): Particle positions
		h_sph (array): SPH water heights
		save (bool): If True, saves the plot
	"""
	z_sph = parabole(h_sph[:, 0])
	h_sph[:, 1] -= Ly
	points[:, 1] -= Ly

	fig, ax1 = plt.subplots(figsize=(10, 6))

	# Plot points and surfaces
	ax1.scatter(points[:, 0], points[:, 1], s=5, color='blue', label='Particles')
	ax1.scatter(h_sph[:, 0], h_sph[:, 1], s=20, color='orange', label='SPH free surface')

	ax1.plot(x_th, z_th, 'k-', label='Topography')
	ax1.fill_between(x_th, z_th, z_th + h_th, color='lightblue', alpha=0.5)
	ax1.plot(x_th, h_th + z_th, color='darkblue', label='Theoretical free surface')

	# Plot configuration
	#ax1.set_xlim(7, 13)
	ax1.set_xlabel('Distance x [m]')
	ax1.set_ylabel('Height [m]')
	ax1.legend(loc='best')
	ax1.grid(True)

	plt.tight_layout()
	if save:
		plt.savefig("Pictures/CH5_valid_test/free_surface/water_height_last.pdf")
		
def check_hydrostatic(all_data, dt,
					y_start, y_end, 
					plot=False, save=False):
	"""
	Simple verification of hydrostatic pressure distribution.
	Computes mean pressure for each y-value, performs linear regression,
	and compares with theoretical hydrostatic pressure.
	"""
	# Extract data
	p_rho2 = all_data['p_/_rho^2']
	y_data = all_data["y"]
	rho = all_data["density"]

	p_reg = []
	p_raw = []
	
	
	for i in range(len(p_rho2)):

		p_r2 = np.array(p_rho2[i])
		r = np.array(rho[i])
		p_raw.append(p_r2 * r/(dt*dt))
		slope, intercept, r_value, p_value, std_err = stats.linregress(y_data[i], p_raw[i])

		y_range = np.linspace(y_start, y_end, 100)
		p_reg.append(slope * y_range + intercept)

	if plot:
		plt.figure()
		for i in range(len(p_rho2)):
			plt.scatter(y_data[i], p_raw[i], s=1)
			plt.plot(y_range, p_reg[i])

	p_mean = np.mean(p_reg, axis=0)
	g = 9.81  # gravitational acceleration
	p_theoretical =  1000 * g * y_range[::-1]
	
	if plot:
		plt.plot(y_range, p_mean, 'k-', label='Mean Regression', linewidth=2)
		plt.plot(y_range, p_theoretical, 'r--', label='Theoretical Hydrostatic Pressure', linewidth=2)
		plt.xlabel('Hauteur y [m]')
		plt.ylabel('Pression [Pa]')
		plt.legend()
		plt.grid(True, alpha=0.3)
		
		
		plt.show()
	

def main():

	# Initial velocity
	U_0 = 0.36 * (m/s)

	# Calculate theoretical water heights
	x_all, z_all, h_all, Fr_all = compute_theoretical_water_height(U_0*0.5)


	# Create plot
	fig, ax1 = plt.subplots(figsize=(10, 6))

	# Plot free surface and topography
	#ax1.plot(x_all, h_all + z_all, label='Free surface')
	#ax1.plot(x_all, z_all, 'k-', label='Topography')
	#ax1.fill_between(x_all, z_all, z_all + h_all, color='lightblue', alpha=0.5)
	# ax1.plot(x_amont, z_amont + h2_conj, linestyle='--', label=r'conjugated $h2_{\text{(x_amont)}}

	annotate_hydraulic_regions(x_all, z_all, h_all, Fr_all, save=False)


if __name__ == "__main__":
	main()