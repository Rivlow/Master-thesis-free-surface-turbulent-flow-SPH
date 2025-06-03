import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
sys.path.append((os.getcwd()))
from python_scripts.Turbulent.Tools_turbulent import compute_u_ghe

# Style configuration
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 26, 26
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


def analyse_delta_gamma():
	# Configure LaTeX rendering
	configure_latex()
	
	# Parameters
	nb_points = 100
	bins = 100
	
	y = np.linspace(-1, 1, nb_points)
	
	delta_0 = 0.05
	gamma_0 = 0.8
	eps = 1e-3
	
	delta = np.linspace(eps, 1, bins)
	gamma = np.linspace(eps, 1, bins)  # Starting at eps to potentially avoid 0
	
	# Create output directory if it doesn't exist
	output_dir = 'Pictures/CH6_valid_test/turbulent'
	os.makedirs(output_dir, exist_ok=True)
	
	# ---- FIRST FIGURE: VELOCITY PROFILES FOR DELTA ----
	fig1, ax1 = plt.subplots(figsize=(12, 6))
	delta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
	for d_val in delta_values:
		if d_val < eps:
			continue
		u = compute_u_ghe(y, 1, d_val, gamma_0)[0]
		u_normalized = u / np.max(u)
		# Use LaTeX formatting for delta in the label
		ax1.plot(y, u_normalized, linewidth=2, label=fr'$\delta = {d_val}$')
	
	ax1.set_xlabel(r'Position $y$ [-]')
	ax1.set_xlim(-1, 1)
	ax1.set_xticks(np.arange(-1, 1.2, 0.2))

	ax1.set_ylabel(r'Velocity $u/U_0$ [-]')
	ax1.set_ylim(0, 1)
	ax1.set_yticks(np.arange(0, 1+0.2, 0.2))
	
	ax1.grid(True, alpha=0.3, ls='--')
	ax1.legend()
	
	plt.tight_layout()
	plt.savefig(f'{output_dir}/velocity_profiles_delta.pdf', dpi=300, bbox_inches='tight')
	
	
	# ---- PREPARE DATA FOR HEATMAPS ----
	y_mesh = np.linspace(-1, 1, 50)  # Fewer points for the heatmap
	delta_mesh = np.linspace(eps, 1, 50)
	gamma_mesh = np.linspace(eps, 1, 50)
	
	# Calculate velocities for different delta values
	u_delta_matrix = np.zeros((len(delta_mesh), len(y_mesh)))
	for i, d_val in enumerate(delta_mesh):
		u = compute_u_ghe(y_mesh, 1, d_val, gamma_0)[0]
		u_delta_matrix[i, :] = u / np.max(u)
	
	# Calculate velocities for different gamma values
	u_gamma_matrix = np.zeros((len(gamma_mesh), len(y_mesh)))
	for i, g_val in enumerate(gamma_mesh):
		u = compute_u_ghe(y_mesh, 1, delta_0, g_val)[0]
		u_gamma_matrix[i, :] = u / np.max(u)
	
	# ---- THIRD FIGURE: HEATMAP FOR DELTA ----
	fig3, ax3 = plt.subplots(figsize=(12, 6))
	Y_mesh, DELTA_mesh = np.meshgrid(y_mesh, delta_mesh)
	im1 = ax3.contourf(Y_mesh, DELTA_mesh, u_delta_matrix, levels=50, cmap='viridis')
	ax3.set_xlabel(r'Position $y$ [-]')
	ax3.set_ylabel(r'Delta $\delta$ [-]')
	ax3.set_yticks(np.arange(0, 1+0.2, 0.2))
	cbar1 = plt.colorbar(im1, ax=ax3)
	cbar1.set_label(r'Velocity $u/U_0$ [-]')
	
	# Add contours for better visualization
	CS = ax3.contour(Y_mesh, DELTA_mesh, u_delta_matrix, levels=[0.2, 0.4, 0.6, 0.8], colors='white', linewidths=1)
	ax3.clabel(CS, inline=True)
	
	plt.tight_layout()
	plt.savefig(f'{output_dir}/heatmap_delta.pdf', dpi=300, bbox_inches='tight')

	plt.show()
	

if __name__=="__main__":
	analyse_delta_gamma()