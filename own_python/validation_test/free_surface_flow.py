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
import matplotlib


# Local imports
sys.path.append(os.getcwd())
from own_python.write_scenes.Tools_scenes import *
from own_python.validation_test.Tools_valid import *

# Style configuration
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


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
	
	
def extract_water_height(vtk_file, mask=None, plot=False, save=False):

	if mask is not None:
		pos = vtk_file.points[mask]
		u = np.array(vtk_file.point_data['velocity'])[mask] 
	else:	
		pos = np.array(vtk_file.points)
		u = np.array(vtk_file.point_data['velocity'])
	
	# Only particles with y > 0
	positive_y_mask = pos[:, 1] > 0
	pos = pos[positive_y_mask]
	u = u[positive_y_mask]

	# Sort particles along x axis
	sorted_indices = np.argsort(pos[:, 0])
	sorted_points = pos[sorted_indices]
	sorted_velocity = u[sorted_indices]
	
	# Sample distance x in bins
	x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
	num_bins = 500
	bin_edges = np.linspace(x_min, x_max, num_bins + 1)
	
	# Highest point (y_max) on each bin and its velocity
	pos_top, u, h, Fr =  [], [], [], []
	Head_rel, Head_abs = [], []
	
	for i in range(num_bins):
		bin_start, bin_end = bin_edges[i], bin_edges[i+1]
		mask = (sorted_points[:, 0] >= bin_start) & (sorted_points[:, 0] < bin_end)
		bin_points = sorted_points[mask]
		bin_velocity = sorted_velocity[mask]
		
		if len(bin_points) > 0:
			max_y_index = np.argmax(bin_points[:, 1])  # index of y_max
			min_y_index = np.argmin(bin_points[:, 1])
			
			# Calculer la norme de vitesse pour chaque particule
			velocity_magnitudes = np.linalg.norm(bin_velocity, axis=1)
			
			# Vitesse caractéristique = moyenne des normes de vitesse
			u_characteristic = np.mean(velocity_magnitudes)
			


			u.append(bin_velocity[max_y_index])
			
			y_topo = bin_points[min_y_index, 1]
			y_surface = bin_points[max_y_index, 1]
			h_local = y_surface - y_topo
			h.append(y_surface)
			pos_top.append(bin_points[max_y_index])
			
			# Froude avec la vitesse caractéristique
			Fr.append(u_characteristic / np.sqrt(g * h_local))
			
			# Charge avec la vitesse caractéristique
			Head_abs.append(y_surface + u_characteristic**2 / (2 * g))
			Head_rel.append(h_local + u_characteristic**2 / (2 * g))
	
	h = np.asarray(pos_top)
	u = np.asarray(u)
	Fr = np.asarray(Fr)
	Head_abs = np.asarray(Head_abs)
	Head_rel = np.asarray(Head_rel)

	
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

	return np.array(pos), np.array(h), np.array(u), np.array(Fr), np.array(Head_rel), np.array(Head_abs)


def compute_theoretical_water_height(Q_init=0.18, nb_points=500):
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

	nb_elem = nb_points
	x_amont = np.linspace(0, x_ressaut, nb_elem)
	x_aval = np.linspace(25, x_ressaut, nb_elem)[:-1]

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

	return x_all, z_all, h_all, Fr_all, H_amont, H_aval

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
	
	# Plot configuration
	ax.set_xlim(7, 13)
	ax.set_xlabel('Distance x [m]', fontsize=14)
	ax.set_ylabel('Height [m]', fontsize=14)
	ax.legend(loc='best')
	ax.grid(True, alpha=0.3)
	
	# Add Froude number subplot
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


def plot_water_height(Ly, x_th, z_th, h_th, points, h_sph, save=False, savepath=None):
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

	fig, ax = plt.subplots(figsize=(10, 6))

	# Plot points and surfaces
	ax.scatter(points[:, 0], points[:, 1], s=5, color='blue', alpha=0.2)
	ax.scatter(h_sph[:, 0], h_sph[:, 1], s=20, color='orange', label='Free surface particle')

	ax.plot(x_th, z_th,'k-', linewidth=3 , label='Topography')
	ax.fill_between(x_th, z_th, z_th + h_th, color='lightblue', alpha=0.6)
	ax.plot(x_th, h_th + z_th, color='darkblue', label='1D model')

	ax.set_xlabel('Distance x [m]')
	ax.set_ylabel('Height [m]')
	ax.legend(loc='best')

	ax.set_xlim((7, 15))
	ax.set_xticks(np.arange(7, 15+1, 1))
	ax.set_ylim(0, 0.5)
	ax.set_yticks(np.arange(0, 0.6, 0.1))
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend()

	plt.tight_layout()
	if save and savepath is not None:
		plt.savefig(f"{savepath}water_height.pdf", dpi=300, bbox_inches='tight')

def plot_Fr(x, x_th, Fr, Fr_th, save=False, savepath=None):

	fig, ax = plt.subplots()
	
	ax.scatter(x, Fr, s=20, marker='o', color='blue', label=r'Fr$_{\text{SPH}}$ [-]')
	plt.plot(x_th, Fr_th, label=r'Fr$_{\text{model}}$ [-]', color='red')
	
	ax.set_xlabel('Position x [m]')
	ax.set_ylabel('Froude number Fr [-]')
	ax.set_xlim((7, 15))
	ax.set_xticks(np.arange(7, 15+1, 1))
	ax.set_ylim(0, 3)
	ax.set_yticks(np.arange(0, 3+0.5, 0.5))
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend()
	plt.tight_layout()
		
	if save and savepath is not None:
		plt.savefig(f"{savepath}/Fr_number.pdf", dpi=300, bbox_inches='tight')
	

def plot_Head(x, H, H_inlet=0, H_outlet=0, save=False, 
			savepath=None):
	
	fig, ax = plt.subplots()
	x_array = np.asarray(x)
	

	x_jump = 11.6657
	
	ax.scatter(x_array, H, s=20, marker='o', color='blue', label="Total head")
	
	ax.axhline(y=H_inlet, xmin=0, xmax=(x_jump-x_array[0])/(x_array[-1]-x_array[0]), 
			linestyle='--', color='red', linewidth=1.5, label=f"Inlet head ({H_inlet:.4f})")
	ax.axhline(y=H_outlet, xmin=(x_jump-x_array[0])/(x_array[-1]-x_array[0]), xmax=1, 
			linestyle='--', color='green', linewidth=1.5, label=f"Outlet head ({H_outlet:.4f})")
	
	ax.set_xlabel('Position x [m]')
	ax.set_ylabel('Total head H [m]')
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend(loc='upper right')

	ax.set_xlim((7, 15))
	ax.set_xticks(np.arange(7, 15+1, 1))
	ax.set_ylim(0.3, 0.55)
	ax.set_yticks(np.arange(0.3, 0.55, 0.05))
	
	plt.tight_layout()
	
	if save and savepath is not None:
		plt.savefig(f"{savepath}/head_conservation.pdf", dpi=300, bbox_inches='tight')
	
		
def is_hydrostatic(p_rho2_slices, rho_slices, y_start, y_end, plot=False, save=False, savepath=None):
    """
    Simple verification of hydrostatic pressure distribution with temporal averaging.
    Computes mean pressure for each y-value across all timesteps, performs linear regression,
    and compares with theoretical hydrostatic pressure.
    Calculates RMSE between regression-based pressure and theoretical pressure.
    
    Args:
        p_rho2_slices: List of timesteps, each containing a list of p_rho2 slices
        rho_slices: List of timesteps, each containing a list of rho slices
        y_start: Starting y-coordinate
        y_end: Ending y-coordinate
        plot: Boolean to generate plots
        save: Boolean to save plots
        savepath: Path to save plots
    """
    # Initialize lists to store results for all timesteps
    p_reg_all_time = []
    p_raw_all_time = []
    y_data_all_time = []
    
    # Get dt from first timestep, first slice
    dt = p_rho2_slices[0][0]['dt']
    y_range = np.linspace(y_start, y_end, 100)
    
    # Get slice positions from first timestep (assuming they're constant)
    slice_positions = []
    for p_rho2_slice in p_rho2_slices[0]:
        slice_positions.append(p_rho2_slice['position'])
    
    # Iterate over timesteps
    for p_rho2_timestep, rho_timestep in zip(p_rho2_slices, rho_slices):
        p_reg_timestep = []
        p_raw_timestep = []
        y_data_timestep = []
        
        # Iterate over slices in this timestep
        for p_rho2, rho in zip(p_rho2_timestep, rho_timestep):
            p_r2 = np.array(p_rho2['values'])
            r = np.array(rho['values'])
            y_data_timestep.append(np.array(p_rho2['coordinates']))
            
            # Calculer la pression
            pressure = p_r2 * r / (dt * dt)
            p_raw_timestep.append(pressure)
            
            # Faire la régression sur tout le tableau
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.array(p_rho2['coordinates']), pressure
            )
            
            # Calculer la ligne de régression
            p_regression = slope * y_range + intercept
            p_reg_timestep.append(p_regression)
        
        p_reg_all_time.append(p_reg_timestep)
        p_raw_all_time.append(p_raw_timestep)
        y_data_all_time.append(y_data_timestep)
    
    # Convertir en arrays numpy pour faciliter le calcul
    p_reg_all_time = np.array(p_reg_all_time)  # shape: (n_timesteps, n_slices, n_y_points)
    
    # Moyenner d'abord sur les timesteps
    p_reg_time_averaged = np.mean(p_reg_all_time, axis=0)  # shape: (n_slices, n_y_points)
    
    # Puis moyenner sur les slices pour obtenir la pression moyenne finale
    p_mean = np.mean(p_reg_time_averaged, axis=0)  # shape: (n_y_points,)
    
    # Calcul de la pression théorique
    g = 9.81  # gravitational acceleration
    p_theoretical = 1000 * g * (y_range[::-1] - y_range[0])
    
    # Calcul du RMSE global
    squared_errors = (p_mean - p_theoretical)**2
    global_rmse = np.sqrt(np.mean(squared_errors))
    
    # Calcul du RMSE pour chaque tranche (après moyennage temporel)
    slice_rmse = []
    for p_regression in p_reg_time_averaged:
        tranche_squared_errors = (p_regression - p_theoretical)**2
        tranche_rmse = np.sqrt(np.mean(tranche_squared_errors))
        slice_rmse.append(tranche_rmse)
    
    if plot:
        # Figure 1: Pression vs hauteur (moyennée en temps)
        plt.figure(figsize=(10, 6))
        plt.plot(y_range, p_mean, 'k-', label='Mean Regression (time-averaged)', linewidth=2)
        plt.plot(y_range, p_theoretical, 'r--', label='Theoretical Hydrostatic Pressure', linewidth=2)
        plt.xlabel('Hauteur y [m]')
        plt.ylabel('Pression [Pa]')
        plt.title(f'Pression moyennée en temps (RMSE globale = {global_rmse:.3f} Pa)')
        plt.legend()
        plt.grid(True, alpha=0.7)
        
        if save and savepath:
            plt.savefig(f'{savepath}_pressure_vs_height.pdf', dpi=300)
        
        # Figure 2: RMSE par tranche (après moyennage temporel)
        plt.figure(figsize=(10, 6))
        plt.plot(slice_positions, slice_rmse, 'o-', color='red')
        plt.ylabel('RMSE [Pa]')
        plt.xlabel('Position x [m]')
        plt.title('RMSE par tranche (après moyennage temporel)')
        plt.grid(True)
        
        if save and savepath:
            plt.savefig(f'{savepath}_rmse_by_slice.pdf', dpi=300)
        
        # Figure 3: Évolution temporelle de la pression moyenne (optionnel)
        # Montrer comment la pression moyenne varie dans le temps pour quelques hauteurs
        plt.figure(figsize=(10, 6))
        n_heights = 5
        height_indices = np.linspace(0, len(y_range)-1, n_heights, dtype=int)
        
        for idx in height_indices:
            # Moyenner sur les slices pour chaque timestep
            p_at_height = np.mean(p_reg_all_time[:, :, idx], axis=1)
            timesteps = np.arange(len(p_reg_all_time))
            plt.plot(timesteps, p_at_height, 'o-', label=f'y = {y_range[idx]:.2f} m')
        
        plt.xlabel('Timestep')
        plt.ylabel('Pression [Pa]')
        plt.title('Évolution temporelle de la pression à différentes hauteurs')
        plt.legend()
        plt.grid(True)
        
        if save and savepath:
            plt.savefig(f'{savepath}_temporal_evolution.pdf', dpi=300)
        
        plt.tight_layout()
        
        if not save:
            plt.show()
    
    return p_mean, p_theoretical, y_range, global_rmse, slice_rmse, slice_positions

def is_uniform(u_slices, save=False, savepath=None):

		std = np.zeros_like(u_slices)
		dist = np.zeros_like(u_slices)
		for i, u in enumerate(u_slices):
			std[i] = np.std(u['values'])
			dist[i] = u['position']

		plt.figure()
		plt.plot(dist, std, 'b-o')
		plt.axhline(y=0, xmin=dist[0], xmax=dist[-1], color='red', linestyle='--', label='Target value')
		plt.xlabel('Position x [m]')
		plt.ylabel(r'Velocity std $\sigma_{u}$(x)')
		plt.legend()
		plt.grid(True, alpha=0.7)
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}velo_uniform.pdf', dpi=300)

	

def main():

	# Initial velocity
	U_0 = 0.36 * (m/s)

	# Calculate theoretical water heights
	x_all, z_all, h_all, Fr_all = compute_theoretical_water_height(U_0*0.5)


	# Create plot
	fig, ax1 = plt.subplots()

	# Plot free surface and topography
	#ax1.plot(x_all, h_all + z_all, label='Free surface')
	#ax1.plot(x_all, z_all, 'k-', label='Topography')
	#ax1.fill_between(x_all, z_all, z_all + h_all, color='lightblue', alpha=0.5)
	# ax1.plot(x_amont, z_amont + h2_conj, linestyle='--', label=r'conjugated $h2_{\text{(x_amont)}}

	annotate_hydraulic_regions(x_all, z_all, h_all, Fr_all, save=False)


if __name__ == "__main__":
	main()