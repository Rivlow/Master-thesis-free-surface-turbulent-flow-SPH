import numpy as np
pi = np.pi

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


