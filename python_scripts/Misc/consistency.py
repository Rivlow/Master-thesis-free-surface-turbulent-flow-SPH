import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import time

from kernels import *

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



def plot_results(s_eval, x_particles, y_particles, x_line, y_line, 
				A1_exact, A1_approx,
				domain, kernel_name,
				plot_domain, plot_func, plot_error):
	
	path = r"Pictures/CH2_sph_fund"
	(x_min, y_min), (x_max, y_max) = domain
	A1_error = np.abs(A1_exact - A1_approx)/np.max(A1_exact)

	if plot_domain:

		fig = plt.figure(figsize=(15, 10))
		plt.scatter(x_particles, y_particles, s=50, alpha=0.7, color='royalblue', edgecolor='navy')
		plt.plot(x_line, y_line, color='darkred', linewidth=2)
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.grid(True, linestyle='--', alpha=0.4)

		plt.savefig(f"{path}/domain_2D.pdf")
		plt.show()

	if plot_func:

		fig = plt.figure()
		plt.plot(s_eval, A1_approx, 'o-', color='blue', markersize=4, alpha=0.5, label='Interpolated function')
		plt.plot(s_eval, A1_exact, color='blue', linewidth=2, label='Exact function')
		plt.xlabel('Representation line')
		plt.ylabel('Approximation')
		plt.xlim(0,1.4)
		plt.xticks(np.arange(0, 1.4+0.35, 0.35))
		plt.ylim(-0.5, 1.5)
		plt.yticks(np.arange(-0.5, 1.6, 0.4))
		plt.grid(True, linestyle='--', alpha=0.4)
		plt.legend(loc = "best")
		plt.tight_layout()

		plt.savefig(f"{path}/func_2D.pdf", dpi=300, bbox_inches='tight')
		plt.show()

	if plot_error:

		fig = plt.figure()
		
		plt.plot(s_eval, A1_error, color='blue', linewidth=2, label='$A_1 - \\langle A_1 \\rangle$')
		plt.xlabel('$s$')
		plt.ylabel('$Error$')
		plt.axhline(y=0, color='black', linestyle=':', alpha=0.8)
		plt.grid(True, linestyle='--', alpha=0.4)
		plt.legend(loc = "best")
		plt.savefig(f"{path}/error_2D.pdf")
		plt.show()   
	
	#fig.suptitle(f'Field Discretization - SPH Approximation with {kernel_name.capitalize()} Kernel', fontsize=16)
	
	print(f"Kernel: {kernel_name}")
	print(f"Number of particles: {len(x_particles)}")
	print(f"Max error A1: {np.max(np.abs(A1_error)):.4f}")


def main():
	
	configure_latex()

	kernel_name = "cubic"
	kernel = extractKernel(kernel_name)
	gradient = extractGradients(kernel_name)
	
	# Domain params
	domain =  [(-2, 0), (2, 1.5)]
	n_x, n_y = 30, 30
	x_particles, y_particles, dx, dy = creat_particle_distribution(domain, n_x, n_y)
	
	# Evaluation line
	coords_line = [(-1.5, 0.2), (1.8, 1.4)]
	x_line, y_line = eval_line(coords_line, domain, nb_points=100)
	
	# Dummy function
	func = lambda x,y : 0.5 * (x+y)*np.tanh(x*y) + np.cos(x**2)
	A1_values = func(x_particles, y_particles) 
	A1_exact = func(x_line, y_line)

	h = 1.5 * dx
	A1_approx = sph_approximation(x_particles, y_particles, x_line, y_line, A1_values, kernel, h, dx, dy)


	plot_results(y_line, x_particles, y_particles, x_line, y_line, 
				A1_exact, A1_approx,
				domain, kernel_name,
				plot_domain=False, plot_func=True, plot_error=False)
	

if __name__ == "__main__":
	main()