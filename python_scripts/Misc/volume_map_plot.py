import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# Set up the style for professional appearance
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

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

# ============= Figure 4.5: Intersection volume with SDF =============
configure_latex()
fig1, ax1 = plt.subplots()

ax1.set_xlim(-1, 4)
ax1.set_ylim(-0.5, 3)
ax1.set_aspect('equal')

# Create boundary shape (curved boundary)
boundary_x = np.linspace(-0.5, 3.5, 100)
boundary_y = 1.2 + 0.3 * np.sin(2 * boundary_x) + 0.1 * np.sin(5 * boundary_x)

# Fill the boundary region (solid part) - only below the boundary
boundary_y_bottom = np.zeros_like(boundary_x)
ax1.fill_between(boundary_x, boundary_y_bottom, boundary_y, 
                alpha=0.15, color='gray', label='Solid Domain')

# Draw boundary line
ax1.plot(boundary_x, boundary_y, 'k-', linewidth=2.5, label='Boundary')

# Particle properties
particle_center = (1.75, 1.75)
particle_radius = 0.7

# Draw particle circle
particle = Circle(particle_center, particle_radius, 
                fill=False, edgecolor='blue', linewidth=2.5, linestyle='-')
ax1.add_patch(particle)

# Create intersection area (approximate)
theta = np.linspace(0, 2*np.pi, 1000)
particle_x = particle_center[0] + particle_radius * np.cos(theta)
particle_y = particle_center[1] + particle_radius * np.sin(theta)

# Find intersection points and create intersection area
intersection_mask = particle_y <= np.interp(particle_x, boundary_x, boundary_y)
if np.any(intersection_mask):
    # Create intersection polygon
    intersection_x = particle_x[intersection_mask]
    intersection_y = particle_y[intersection_mask]
    
    # Add boundary points within the particle
    particle_x_range = (particle_center[0] - particle_radius, 
                    particle_center[0] + particle_radius)
    boundary_in_particle = (boundary_x >= particle_x_range[0]) & (boundary_x <= particle_x_range[1])
    boundary_x_intersect = boundary_x[boundary_in_particle]
    boundary_y_intersect = boundary_y[boundary_in_particle]
    
    # Filter boundary points that are within the particle
    dist_to_center = np.sqrt((boundary_x_intersect - particle_center[0])**2 + 
                    (boundary_y_intersect - particle_center[1])**2)
    boundary_in_circle = dist_to_center <= particle_radius
    
    if np.any(boundary_in_circle):
        boundary_x_final = boundary_x_intersect[boundary_in_circle]
        boundary_y_final = boundary_y_intersect[boundary_in_circle]
        
        # Combine points for intersection polygon
        all_x = np.concatenate([intersection_x, boundary_x_final])
        all_y = np.concatenate([intersection_y, boundary_y_final])
        
        # Sort points to create proper polygon
        center_x, center_y = np.mean(all_x), np.mean(all_y)
        angles = np.arctan2(all_y - center_y, all_x - center_x)
        sorted_indices = np.argsort(angles)
        
        intersection_poly = Polygon(list(zip(all_x[sorted_indices], all_y[sorted_indices])), 
                                alpha=0.6, facecolor='lightcoral', 
                                edgecolor='red', linewidth=1.5)
        ax1.add_patch(intersection_poly)

# Add SDF arrow and point - positioned relative to particle center
sdf_point = (particle_center[0], particle_center[1])
ax1.plot(sdf_point[0], sdf_point[1], 'o', color='darkblue', markersize=8, zorder=10, label="Particle")

# SDF arrow pointing towards boundary
closest_boundary_x = particle_center[0]  # Use particle center x-coordinate
closest_boundary_y = np.interp(closest_boundary_x, boundary_x, boundary_y)
sdf_arrow = FancyArrowPatch(sdf_point, (1.577, 1.326),
                        arrowstyle='->', mutation_scale=20, 
                        color='green', linewidth=2, zorder=9)
ax1.add_patch(sdf_arrow)

# Labels and annotations
ax1.text(sdf_point[0] + 0.1, sdf_point[1]-0.15, 'SDF', 
        color='green', fontweight='bold')
ax1.text(1.5, 0.5, 'Boundary', ha='center', fontweight='bold')

ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_xlim(0,3)
ax1.set_ylim(0,3)
# Define grid parameters
grid_start, grid_end = 0, 3
grid_step = 0.5  # Changed to 0.5 to match your nodes positions

# Create custom grid lines that align with the nodes
xticks = np.arange(grid_start, grid_end + grid_step, grid_step)
yticks = np.arange(grid_start, grid_end + grid_step, grid_step)

# Set the ticks first
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.grid(True, alpha=0.3, ls='--')

# Add legend for first plot
legend_elements1 = [
    mpatches.Patch(color='lightcoral', alpha=0.6, label='Intersection Volume')
]
ax1.legend(handles=legend_elements1, loc='upper left')

# Save and show first plot
plt.tight_layout()
plt.savefig('Pictures/CH4_splishsplash/intersection_volume_sdf.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ============= Figure 4.6: Grid interpolation =============
fig2, ax2 = plt.subplots()

ax2.set_xlim(0, 3)
ax2.set_ylim(0, 3)
ax2.set_aspect('equal')

# Create boundary (similar but slightly different)
boundary_x2 = np.linspace(-0.5, 3.5, 100)
boundary_y2 = 1.2 + 0.3 * np.sin(2 * boundary_x2) + 0.1 * np.sin(5 * boundary_x2)

# Fill boundary region
ax2.fill_between(boundary_x2, np.zeros_like(boundary_x2), boundary_y2, 
                alpha=0.15, color='gray')

# Draw boundary
ax2.plot(boundary_x2, boundary_y2, 'k-', linewidth=2.5)

# Define grid parameters
grid_start, grid_end = 0, 3
grid_step = 0.5  # Changed to 0.5 to match your nodes positions

# Create custom grid lines that align with the nodes
xticks = np.arange(grid_start, grid_end + grid_step, grid_step)
yticks = np.arange(grid_start, grid_end + grid_step, grid_step)

# Set the ticks first
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)

# Then add the grid
ax2.grid(True, alpha=0.3, ls="--")

# Define the specific cell and its nodes (aligned with grid)
cell_center_x, cell_center_y = 1.75, 1.75
grid_nodes_x = np.array([1.5, 2.0, 1.5, 2.0])  # These should match grid_step
grid_nodes_y = np.array([1.5, 1.5, 2.0, 2.0])

# Draw only the 4 grid nodes of the selected cell
ax2.scatter(grid_nodes_x, grid_nodes_y, c='black', s=60, zorder=10, label='Grid Nodes')

# Central particle for interpolation - positioned at the center of the cell
ax2.plot(cell_center_x, cell_center_y, 'o', 
        color='blue', markersize=12, zorder=15, label='Particle')

# Draw interpolation connections - connect all 4 cell nodes to the particle
for gx, gy in zip(grid_nodes_x, grid_nodes_y):
    line = plt.Line2D([cell_center_x, gx], [cell_center_y, gy],
                    color='red', linewidth=1.5, alpha=0.7, zorder=8)
    ax2.add_line(line)

# Add interpolation arrows - one arrow from each of the 4 nodes
for i, (gx, gy) in enumerate(zip(grid_nodes_x, grid_nodes_y)):
    arrow = FancyArrowPatch((gx, gy), 
                        (cell_center_x + 0.5*(gx-cell_center_x), 
                            cell_center_y + 0.5*(gy-cell_center_y)),
                        arrowstyle='->', mutation_scale=15, 
                        color='red', linewidth=1.2, alpha=0.8)
    ax2.add_patch(arrow)

# Labels and annotations
ax2.text(1.5, 0.5, 'Boundary', ha='center', fontweight='bold')

ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')

# Add legend for second plot
legend_elements2 = [
    plt.Line2D([0], [0], marker='o', color='black', linewidth=0, 
            markersize=6, label='Grid Nodes'),
    plt.Line2D([0], [0], color='red', linewidth=1.5, label='Interpolation')
]
ax2.legend(handles=legend_elements2, loc='upper left')

# Save and show second plot
plt.tight_layout()
plt.savefig('Pictures/CH4_splishsplash/grid_interpolation.pdf', dpi=300, bbox_inches='tight')
plt.show()