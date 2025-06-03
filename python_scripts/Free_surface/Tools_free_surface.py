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
from python_scripts.Tools_scenes import *
from python_scripts.Tools_global import *

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
	g = 9.81  # Accélération gravitationnelle
	rho_water = 1000.0  # Densité de l'eau (kg/m³)
	
	# Extraction des données de base
	if mask is not None:
		pos = vtk_file.points[mask]
		u = np.array(vtk_file.point_data['velocity'])[mask] 

		p_rho2 = np.array(vtk_file.point_data['p_/_rho^2'])[mask]/(vtk_file.point_data['dt'][mask]**2) #/((np.array(vtk_file.point_data['dt'])[mask])**2)
		rho = np.array(vtk_file.point_data['density'])[mask]
	else:
		pos = np.array(vtk_file.points)
		u = np.array(vtk_file.point_data['velocity'])
		p_rho2 = np.array(vtk_file.point_data['p_/_rho^2'])
		rho = np.array(vtk_file.point_data['density'])
	
	
	# Tri selon l'axe x
	sorted_indices = np.argsort(pos[:, 0])
	sorted_points = pos[sorted_indices]
	sorted_velocity = u[sorted_indices]
	sorted_p_rho2 = p_rho2[sorted_indices]
	sorted_rho = rho[sorted_indices]
	
	# Création des bins
	x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
	num_bins = 500
	bin_edges = np.linspace(x_min, x_max, num_bins + 1)
	
	# Initialisation des listes de résultats (identique à l'original)
	pos_top = []  # Points de surface [x, y]
	u_top = []    # Vitesse aux points de surface
	h = []        # Hauteur d'eau (y_surface)
	Fr = []       # Nombre de Froude
	Head_rel = [] # Charge relative
	Head_abs = [] # Charge absolue

	for i in range(num_bins):
		bin_mask = (sorted_points[:, 0] >= bin_edges[i]) & (sorted_points[:, 0] < bin_edges[i+1])
		bin_points = sorted_points[bin_mask]
		
		if len(bin_points) == 0:
			continue
			
		# Identification surface et fond
		max_y_idx = np.argmax(bin_points[:, 1])
		min_y_idx = np.argmin(bin_points[:, 1])
		
		y_surface = bin_points[max_y_idx, 1]
		y_bottom = bin_points[min_y_idx, 1]
		h_local = y_surface - y_bottom
		
		# Calcul pression au fond (point le plus bas)
		p_bottom = sorted_p_rho2[bin_mask][max_y_idx]
		pressure_head = p_bottom / (g)
		
		# Calcul vitesse moyenne dans le bin
		velocity_vectors = sorted_velocity[bin_mask]
		u_mean_mag = np.mean(np.linalg.norm(velocity_vectors, axis=1))
		
		# Calcul charge
		velocity_head = u_mean_mag**2 / (2 * g)
		Head_abs.append(y_surface + pressure_head + velocity_head)
		Head_rel.append(h_local + pressure_head + velocity_head)
		
		# Stockage des résultats (format identique à l'original)
		pos_top.append([bin_edges[i] + (bin_edges[i+1] - bin_edges[i])/2, y_surface])  # Centre du bin
		u_top.append(velocity_vectors[max_y_idx])  # Vitesse au point de surface
		Fr.append(u_mean_mag / np.sqrt(g * h_local))
		h.append(y_surface)

	# Conversion en arrays numpy
	pos_top = np.array(pos_top)
	u_top = np.array(u_top)
	Fr = np.array(Fr)
	Head_abs = np.array(Head_abs)
	Head_rel = np.array(Head_rel)
	
	# Tracé des graphiques (identique à l'original)
	if plot:
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
		
		ax1.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.3, color='blue', label='Particles')
		ax1.scatter(pos_top[:, 0], pos_top[:, 1], s=20, color='red', label='Water height')
		ax1.set_xlabel('Position x [m]')
		ax1.set_ylabel('Height h(x) [m]')
		ax1.legend()
		ax1.grid(True)

		ax2.scatter(pos_top[:, 0], Fr, s=20, color='green', label='Froude number')
		ax2.set_xlabel('Position x [m]')
		ax2.set_ylabel('Froude number Fr [-]')
		ax2.legend()
		ax2.grid(True)

		plt.tight_layout()
		if save:
			plt.savefig('Pictures/CH5_valid_test/free_surface/water_height_velocity_sph.pdf', dpi=30)
		plt.show()

	return pos, pos_top, u_top, Fr, Head_rel, Head_abs


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
	x_amont = np.linspace(0, x_ressaut, nb_elem//3+1)
	x_aval = np.linspace(25, x_ressaut, nb_elem//3+1)[:-1]

	# Topography
	z_aval = np.array([z_b(x) for x in x_aval])
	z_amont = np.array([z_b(x) for x in x_amont])

	# Compute water height from "z_b(x) + h(x) + (q²/2g)(1/h²(x)) = H_aval (or H_amont)"
	h_amont = np.array([solve_height_amont(q, g, x, z_b, H_amont) for x in x_amont])
	h_aval = np.array([solve_height_aval(q, g, x, z_b, H_aval) for x in x_aval])

	h_inter = np.linspace(h_amont[-1], h_aval[-1], nb_elem//3+1)
	x_inter = np.linspace(x_amont[-1], x_aval[-1], nb_elem//3+1)
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
	fig, ax = plt.subplots(figsize=(12, 6))
	
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
	ax.set_xlabel('Distance x [m]')
	ax.set_ylim(0, 0.5)
	ax.set_ylabel('Height [m]')
	ax.legend(loc='best')
	ax.grid(True, alpha=0.3, ls="--")
	
	# Add Froude number subplot
	plt.tight_layout()
	
	if save:
		plt.savefig("Pictures/CH5_valid_test/free_surface/annotated_hydraulic_jump.pdf", dpi=30)
	
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
	fig, ax = plt.subplots(figsize=(12, 6))
	
	# Find indices for critical point and hydraulic jump
	cr_idx = np.abs(x_all - x_cr).argmin()
	ressaut_idx = np.abs(x_all - x_ressaut).argmin()
	
	# Calculate conjugate heights for the entire region around the critical point
	# Start from upstream (subcritical region) to downstream past the hydraulic jump
	start_idx = max(0, cr_idx - 200)  # Start well before critical point
	end_idx = min(len(x_all) - 1, ressaut_idx + 100)  # End past hydraulic jump
	
	# Extract data for this extended region
	x_conj_region = x_all[start_idx:end_idx+1]
	z_conj_region = z_all[start_idx:end_idx+1]
	h_conj_region = h_all[start_idx:end_idx+1]
	Fr_conj_region = Fr_all[start_idx:end_idx+1]
	
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
	ax.set_xlabel('Distance x [m]')
	ax.set_xlim(7, 13)  # Focus on the relevant region
	ax.set_ylim(0, 0.5)
	ax.set_yticks(np.arange(0, 0.6, 0.1))
	ax.set_ylabel('Height [m]')
	ax.legend(loc='best')
	ax.grid(True, alpha=0.3, ls="--")
	
	plt.tight_layout()
	
	if save:
		plt.savefig("Pictures/CH6_valid_test/free_surface/conjugate_height.pdf", dpi=30, bbox_inches="tight")
	
	return fig, ax


def plot_water_height(particle, x_th=None, z_th=None, h_th=None, points=None, h_sph=None,
					label=None, obstacles=None,
					save=False, savepath=None):
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
	#z_sph = parabole(h_sph[:, 0])
	

	fig, ax = plt.subplots(figsize=(12, 6))

	if isinstance(points, list) and isinstance(h_sph, list):
		for points_i, h_sph_i, label_i in zip(points, h_sph, label):
			h_sph_i[:, 1] += particle
			points_i[:, 1] += particle
			#ax.scatter(points_i[:, 0], points_i[:, 1], s=5, color='blue', alpha=0.2)
			ax.scatter(h_sph_i[:, 0], h_sph_i[:, 1], s=20, label=label_i)
	else:

		h_sph[:, 1] += particle
		points[:, 1] += particle

		# Plot points and surfaces
		ax.scatter(points[:, 0], points[:, 1], s=5, color='blue', alpha=0.2)
		ax.scatter(h_sph[:, 0], h_sph[:, 1], s=20, color='orange', label='Free surface particle')

	if x_th is not None and z_th is not None and h_th is not None:
		ax.plot(x_th, z_th,'k-', linewidth=3 , label='Topography')
		ax.fill_between(x_th, z_th, z_th + h_th, color='lightblue', alpha=0.6)
		ax.plot(x_th, h_th + z_th, color='darkblue', label='1D model')

	
	rect = patches.Rectangle(
		(1.5, 0.211), 0.5, 0.064,
		linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7
	)
	ax.add_patch(rect)

	ax.set_xlabel('Distance x [m]')
	ax.set_ylabel('Height [m]')
	ax.legend(loc='best')

	ax.set_xlim((0, 3.5))
	#ax.set_xticks(np.arange(7, 15+1, 1))
	ax.set_ylim(0, 0.45)
	ax.set_yticks(np.linspace(0, 0.45, 5))
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend()

	plt.tight_layout()
	if save and savepath is not None:
		plt.savefig(f"{savepath}water_height.pdf", dpi=30, bbox_inches='tight')

def plot_Fr(x, x_th, Fr, Fr_th, save=False, savepath=None):

	fig, ax = plt.subplots(figsize=(12,6))
	
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
		plt.savefig(f"{savepath}/Fr_number.pdf", dpi=30, bbox_inches='tight')
	

def plot_Head(x, H, label=None, H_inlet=None, H_outlet=None, save=False, 
			savepath=None):
	
	fig, ax = plt.subplots(figsize=(12,6))
	
	if isinstance(x, list) and isinstance(H, list):
		for x_i, H_i, label_i in zip(x, H, label):
			ax.scatter(x_i, H_i, s=20, label=label_i)

	else:
		ax.scatter(x, H, s=20, marker='o', color='blue', label="Total head")

	'''
	x_jump = 11.6657
	if H_inlet is not None and H_outlet is not None:
		ax.axhline(y=H_inlet, xmin=0, xmax=(x_jump-x_array[0])/(x_array[-1]-x_array[0]), 
				linestyle='--', color='red', linewidth=1.5, label=f"Inlet head ({H_inlet:.4f})")
		ax.axhline(y=H_outlet, xmin=(x_jump-x_array[0])/(x_array[-1]-x_array[0]), xmax=1, 
				linestyle='--', color='green', linewidth=1.5, label=f"Outlet head ({H_outlet:.4f})")
	'''
	
	ax.set_xlabel('Position x [m]')
	ax.set_ylabel('Total head H [m]')
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend(loc='upper right')

	#ax.set_xlim((7, 15))
	#ax.set_xticks(np.arange(7, 15+1, 1))
	#ax.set_ylim(0.3, 0.55)
	#ax.set_yticks(np.arange(0.3, 0.55, 0.05))
	
	plt.tight_layout()
	
	if save and savepath is not None:
		plt.savefig(f"{savepath}/head_conservation.pdf", dpi=30, bbox_inches='tight')

def mean_head(all_vtk, particle, mm=1e-3, 
			  save=False, savepath=None):
		
		x_min, x_max = 0.0, 3.5  # À adapter selon votre domaine
		x_common = np.linspace(x_min, x_max, 500)  # Grille fixe de 500 points

		# Étape 2: Liste pour stocker les données interpolées
		all_head_interp_1 = []
		all_head_interp_2 = []

		for vtk in all_vtk[50:]:
			_, mask_XY_1 = project_surface(vtk.points, "z", 150*mm, thickness=10*particle)
			_, mask_XY_2 = project_surface(vtk.points, "z", -150*mm, thickness=10*particle)

			_, pos_top_1, _, _, _, Head_1 = extract_water_height(vtk, mask=mask_XY_1, plot=False, save=False)
			_, pos_top_2, _, _, _, Head_2 = extract_water_height(vtk, mask=mask_XY_2, plot=False, save=False)
			
			head_interp_1 = np.interp(x_common, pos_top_1[:, 0], Head_1, left=np.nan, right=np.nan)
			all_head_interp_1.append(head_interp_1)
			head_interp_2 = np.interp(x_common, pos_top_2[:, 0], Head_2, left=np.nan, right=np.nan)
			all_head_interp_2.append(head_interp_2)

		# Étape 3: Conversion en array et calcul de la moyenne
		head_array_1 = np.array(all_head_interp_1)
		Head_mean_1 = np.nanmean(head_array_1, axis=0) 
		head_array_2 = np.array(all_head_interp_2)
		Head_mean_2 = np.nanmean(head_array_2, axis=0) 

		# Tracé du résultat
		plt.figure(figsize=(12, 6))
		plt.scatter(x_common, Head_mean_1, s=5, label='Slice z = 150 mm')
		plt.scatter(x_common, Head_mean_2, s=5, label='Slice z = -150 mm')

		plt.xlabel('Position x [m]')
		plt.xlim(x_min, x_max)
		plt.xticks(np.arange(x_min, x_max + 0.5, 0.5))


		plt.ylabel('Total head H [m]')
		plt.legend()


		plt.grid(True, alpha=0.4, ls="--")
		plt.ylim(0.15, 0.4)
		plt.yticks(np.arange(0.15, 0.4+0.05, 0.05))
		plt.tight_layout()
		

		if save and savepath is not None:
			plt.savefig(f"{savepath}/mean_head.pdf", dpi=30, bbox_inches='tight')
			
	
		
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
	y_range = np.linspace(y_start, y_end, 100)
	
	# Get slice positions from first timestep (assuming they're constant)
	slice_positions = []
	for p_rho2_slice in p_rho2_slices[0]:
		slice_positions.append(p_rho2_slice['position']+7)
	
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
			pressure = p_r2 * r / (p_rho2['dt'] * p_rho2['dt'])
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

	print(f"MSE globale = {global_rmse:.3f}")
	
	if plot:
		# Figure 1: Pression vs hauteur (moyennée en temps)
		plt.figure(figsize=(12, 6))
		plt.plot(y_range, p_mean, 'k-', label='Mean Regression (time-averaged)', linewidth=2)
		plt.plot(y_range, p_theoretical, 'r--', label='Theoretical Hydrostatic Pressure', linewidth=2)
		plt.xlabel('Height y [m]')
		plt.ylabel('Pressure P [Pa]')
		plt.xlim(0, 0.33)
		plt.ylim(0, 3500)
		plt.xticks(np.linspace(0, 0.33, 5))
		plt.yticks(np.linspace(0, 3500, 5))
		
		plt.legend()
		plt.tight_layout()
		plt.grid(True, alpha=0.4, ls="--")
		
		if save and savepath:
			plt.savefig(f'{savepath}_pressure_vs_height.pdf', dpi=30)
		
		# Figure 2: RMSE par tranche (après moyennage temporel)
		plt.figure(figsize=(12, 6))
		plt.plot(slice_positions, slice_rmse, 'o-', color='red')
		plt.ylabel('RMSE [Pa]')
		plt.xlabel('Position x [m]')
		plt.xlim(7, 15)
		plt.xticks(np.arange(7, 16, 1))
		plt.ylim(0, 1500)
		plt.yticks(np.arange(0, 1750, 250))
		plt.grid(True, alpha=0.4, ls="--")
		plt.tight_layout()
		
		if save and savepath:
			plt.savefig(f'{savepath}_rmse_by_slice.pdf', dpi=30)
		
	return p_mean, p_theoretical, y_range, global_rmse, slice_rmse, slice_positions

def is_uniform(u_slices, save=False, savepath=None):

		std = np.zeros_like(u_slices)
		dist = np.zeros_like(u_slices)
		for i, u in enumerate(u_slices):
			std[i] = np.std(u['values'])
			dist[i] = u['position']+7

		plt.figure(figsize=(12, 6))
		plt.plot(dist, std, 'b-o')
		plt.xlabel('Position x [m]')
		plt.xlim(7,15)
		plt.xticks(np.arange(7, 16, 1))
		plt.ylim(0, 0.6)
		plt.yticks(np.arange(0, 0.7, 0.1))
		plt.ylabel(r'Deviation $\sigma_{u}$(x)')
		plt.legend()
		plt.grid(True, alpha=0.4, ls="--")
		plt.tight_layout()

		if save and savepath is not None:
			plt.savefig(f'{savepath}velo_uniform.pdf', dpi=30, bbox_inches='tight')

	

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

	#annotate_hydraulic_regions(x_all, z_all, h_all, Fr_all, save=False)
	plot_conjugate_height(x_all, z_all, h_all, Fr_all, save=False)


if __name__ == "__main__":
	main()