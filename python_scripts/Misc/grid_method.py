import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def configure_latex():
	"""Configure matplotlib to use LaTeX for rendering text."""
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')


# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 22, 22
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

configure_latex()

# Create figure and axis
fig, ax = plt.subplots(1, 1)

# Define grid parameters
grid_size = 5
x_grid = np.linspace(0, 4, grid_size)
y_grid = np.linspace(0, 4, grid_size)

# Create grid lines
for x in x_grid:
    ax.axvline(x, color='black', linewidth=1, alpha=0.4, linestyle='--')
for y in y_grid:
    ax.axhline(y, color='black', linewidth=1, alpha=0.4, linestyle='--')

# Generate random particle positions
np.random.seed(42)
all_particles_x = np.random.uniform(0.1, 3.9, 50)
all_particles_y = np.random.uniform(0.1, 3.9, 50)

# Define single grid point and kernel radius (at center of a cell)
kernel_center = (1.5, 1.5)
kernel_radius = 0.8

# Classify particles based on distance to kernel center
green_particles_x = []
green_particles_y = []
red_particles_x = []
red_particles_y = []

for x, y in zip(all_particles_x, all_particles_y):
    distance = np.sqrt((x - kernel_center[0])**2 + (y - kernel_center[1])**2)
    if distance <= kernel_radius:
        green_particles_x.append(x)
        green_particles_y.append(y)
    else:
        red_particles_x.append(x)
        red_particles_y.append(y)

# Plot particles
ax.scatter(red_particles_x, red_particles_y, c='red', s=60, alpha=0.7, 
           edgecolors='darkred', linewidth=1, label='Particles outside kernel')
ax.scatter(green_particles_x, green_particles_y, c='green', s=60, alpha=0.7,
           edgecolors='darkgreen', linewidth=1, label='Particles inside kernel')

# Draw SPH kernel circle (support domain)
circle = Circle(kernel_center, kernel_radius, fill=False, linestyle='--', 
               color='blue', linewidth=3, alpha=0.9)
ax.add_patch(circle)

# Add grid point at center
ax.plot(kernel_center[0], kernel_center[1], 'o', color='blue', markersize=10, alpha=0.9)

# Set axis properties
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')


# Add legend
ax.legend(loc='upper right')

# Add text annotations
ax.text(2.25, 2.5, 'Grid cells', style='italic', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))

plt.tight_layout()

plt.savefig('Pictures/CH5_data_treatment/grid_method.pdf', dpi=300, bbox_inches="tight")
plt.show()

