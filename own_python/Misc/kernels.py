import numpy as np
pi = np.pi

class SPHKernels:

	@staticmethod
	def cubic_kernel(r, radius):
		h3 = radius * radius * radius
		k = 8.0 / (pi * h3)
		
		# Vérifier si r est un scalaire et le convertir en tableau si nécessaire
		is_scalar = np.isscalar(r)
		if is_scalar:
			r = np.array([r])
		
		q = r / radius
		result = np.zeros_like(r)
		
		mask1 = q <= 0.5
		result[mask1] = k * (6.0 * q[mask1]**3 - 6.0 * q[mask1]**2 + 1.0)
		
		mask2 = np.logical_and(q > 0.5, q <= 1.0)
		result[mask2] = k * (2.0 * (1.0 - q[mask2])**3)
		
		# Si l'entrée était un scalaire, retourner un scalaire
		if is_scalar:
			return result[0]
		
		return result
	
	@staticmethod
	def cubic_kernel_gradient(r, radius):

		h3 = radius * radius * radius
		k = 8.0 / (pi * h3)
		l = 48.0 / (pi * h3)

		q = r / radius
		result = np.zeros_like(r)
		
		mask_nonzero = r > 1.0e-9
		
		mask1 = np.logical_and(mask_nonzero, q <= 0.5)
		result[mask1] = l * q[mask1] * (3.0 * q[mask1] - 2.0) / radius
		
		mask2 = np.logical_and(mask_nonzero, np.logical_and(q > 0.5, q <= 1.0))
		result[mask2] = l * (-1.0) * ((1.0 - q[mask2])**2) / radius
		
		return result

	@staticmethod
	def poly6_kernel(r, radius):

		k = 315.0 / (64.0 * pi * radius**9)
		l = -945.0 / (32.0 * pi * radius**9)
		r2 = r * r
		radius2 = radius * radius
		result = np.zeros_like(r)
		
		mask = r2 <= radius2
		result[mask] = k * (radius2 - r2[mask])**3
		return result
	
	@staticmethod
	def poly6_kernel_gradient(r, radius):

		k = 315.0 / (64.0 * pi * radius**9)
		l = -945.0 / (32.0 * pi * radius**9)
		r2 = r * r
		radius2 = radius * radius
		result = np.zeros_like(r)
		
		mask_nonzero = r > 1.0e-9
		mask = np.logical_and(mask_nonzero, r2 <= radius2)
		
		result[mask] = l * r[mask] * (radius2 - r2[mask])**2
		
		return result
	
	@staticmethod
	def spiky_kernel(r, radius):

		radius6 = radius**6
		k = 15.0 / (pi * radius6)
		l = -45.0 / (pi * radius6)
		result = np.zeros_like(r)
		
		mask = r <= radius
		result[mask] = k * (radius - r[mask])**3
		return result
	
	@staticmethod
	def spiky_kernel_gradient(r, radius):

		radius6 = radius**6
		k = 15.0 / (pi * radius6)
		l = -45.0 / (pi * radius6)
		result = np.zeros_like(r)
		
		mask_nonzero = r > 1.0e-9
		mask = np.logical_and(mask_nonzero, r <= radius)
		
		result[mask] = l * (radius - r[mask])**2
		
		return result
	
	@staticmethod
	def wendland_kernel(r, radius):

		h3 = radius * radius * radius
		k = 21.0 / (2.0 * pi * h3)
		l = -210.0 / (pi * h3)

		q = r / radius
		result = np.zeros_like(r)
		
		mask = q <= 1.0
		result[mask] = k * np.power(1.0 - q[mask], 4) * (4.0 * q[mask] + 1.0)
		return result
	
	
def extractKernel(kernel_name: str):
	kernels_all = SPHKernels()
	
	kernel_dict = {
		'cubic': kernels_all.cubic_kernel, 
		'poly6': kernels_all.poly6_kernel,
		'spiky': kernels_all.spiky_kernel,
		'wendland': kernels_all.wendland_kernel
	}
	
	if kernel_name.lower() not in kernel_dict:
		raise ValueError(f"Kernel {kernel_name} not found. Available kernels: {list(kernel_dict.keys())}")
	
	return kernel_dict[kernel_name.lower()]

def extractGradients(kernel_name: str):
	kernels_all = SPHKernels()
	
	kernel_dict = {
		'cubic': kernels_all.cubic_kernel_gradient,  
		'spiky': kernels_all.spiky_kernel_gradient,
		'poly6': None,  
		'wendland': None
	}
	
	if kernel_name.lower() not in kernel_dict:
		raise ValueError(f"Kernel gradient {kernel_name} not found. Available kernels: {list(kernel_dict.keys())}")
	
	return kernel_dict[kernel_name.lower()]

def creat_particle_distribution(domain, n_x, n_y, add_jitter=True):

	(x_min, y_min), (x_max, y_max) = domain

	dx = (x_max - x_min) / (n_x - 1)
	dy = (y_max - y_min) / (n_y - 1)
	x_grid = np.linspace(x_min, x_max, n_x)
	y_grid = np.linspace(y_min, y_max, n_y)
	
	X, Y = np.meshgrid(x_grid, y_grid)
	x_particles = X.flatten()
	y_particles = Y.flatten()
	
	# Add noise to particle distribution
	if add_jitter:
		np.random.seed(42)
		jitter_amount = 0.35 * min(dx, dy)
		x_particles += np.random.uniform(-jitter_amount, jitter_amount, size=len(x_particles))
		y_particles += np.random.uniform(-jitter_amount, jitter_amount, size=len(y_particles))
	
	return x_particles, y_particles, dx, dy



def eval_line(coords_line, domain, nb_points):

	(x1, y1), (x2, y2) = coords_line
	(x_min, y_min), (x_max, y_max) = domain

	# Check boundaries
	if not all(x_min <= x <= x_max and y_min <= y <= y_max for x, y in coords_line):
		print("Warning: points outside domain !")

	slope = (y2-y1)/(x2-x1)
	x_line = np.linspace(x1, x2, nb_points)
	y_line = slope*(x_line-x1) + y1

	return x_line, y_line


def check_sph_integral(x_particles, y_particles, eval_points, kernel, h, dx, dy):

	volume = dx * dy
	integral_values = np.zeros(len(eval_points))
	
	for i, (x_eval, y_eval) in enumerate(eval_points):

		r = np.sqrt((x_eval - x_particles)**2 + (y_eval - y_particles)**2)
		w = kernel(r, h)
		integral = np.sum(w * volume)
		integral_values[i] = integral
	
	return integral_values

def sph_approximation(x_particles, y_particles, x_line, y_line, func_values, kernel, h, dx, dy, neighbour_radius=None):

	pairs = np.column_stack((x_line, y_line))
	approx = np.zeros_like(x_line)
	volume = dx * dy
	
	for i, (x, y) in enumerate(pairs):

		r = np.sqrt((x - x_particles)**2 + (y - y_particles)**2)
		
		# Only neighbourhood considered
		if neighbour_radius is not None:

			mask = r <= neighbour_radius
			r_filtered = r[mask]

			func_values_filtered = func_values[mask]
			w = kernel(r_filtered, h)
			kernel_sum = np.sum(w * volume)
			
			# Normalisation 
			if kernel_sum > 1e-10:
				approx[i] = np.sum(func_values_filtered * w * volume) / kernel_sum
			else:
				approx[i] = 0.0

		# All particles considered
		else:
			w = kernel(r, h)
			kernel_sum = np.sum(w * volume)
			
			# Normalisation
			if kernel_sum > 1e-10:
				approx[i] = np.sum(func_values * w * volume) / kernel_sum
			else:
				approx[i] = 0.0
				
	return approx


