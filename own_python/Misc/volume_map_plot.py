import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Wedge

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

def isLatex(latex):
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')



def sdf_sphere(points, center, radius):
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    return distances - radius


def compute_volume_intersection(grid_points, sphere_center, sphere_radius, particle_radius=0.5):
    distances = np.sqrt(np.sum((grid_points - sphere_center)**2, axis=1))
    vol = np.zeros_like(distances)
    
    # Points à l'intérieur de la sphère
    inside_mask = distances <= sphere_radius
    vol[inside_mask] = 0.8
    
    # Points proches de la surface
    near_mask = (distances > sphere_radius) & (distances <= sphere_radius + particle_radius)
    d_rel = (distances[near_mask] - sphere_radius) / particle_radius
    vol[near_mask] = 0.8 * (1 - d_rel)
    
    return vol


def add_wireframe_sphere(ax, center, radius, color, alpha=0.2):
    """Ajoute une sphère wireframe à l'axe 3D"""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)


def calculate_circle_intersection(circle1_center, circle1_radius, circle2_center, circle2_radius):
    """Calcule les points d'intersection entre deux cercles"""
    d = np.sqrt(np.sum((circle1_center - circle2_center)**2))
    
    if d > circle1_radius + circle2_radius or d < abs(circle1_radius - circle2_radius):
        return None
    
    a = (circle1_radius**2 - circle2_radius**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, circle1_radius**2 - a**2))
    
    x2 = circle1_center[0] + a * (circle2_center[0] - circle1_center[0]) / d
    y2 = circle1_center[1] + a * (circle2_center[1] - circle1_center[1]) / d
    
    x3 = x2 + h * (circle2_center[1] - circle1_center[1]) / d
    y3 = y2 - h * (circle2_center[0] - circle1_center[0]) / d
    
    x4 = x2 - h * (circle2_center[1] - circle1_center[1]) / d
    y4 = y2 + h * (circle2_center[0] - circle1_center[0]) / d
    
    return np.array([x3, y3]), np.array([x4, y4])


def create_3d_visualization(grid_size, particle_positions, sphere_center, sphere_radius, particle_radius):
    """Crée la visualisation 3D de la grille et des particules"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Créer la grille
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    z = np.linspace(-2, 2, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Calculer les volumes d'intersection
    grid_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    volumes = compute_volume_intersection(grid_points, sphere_center, sphere_radius, particle_radius)
    
    # Colorer les nœuds selon leur valeur d'intersection
    scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                        c=volumes, s=20, alpha=1, cmap='inferno', label='Nodes')
    
    # Tracer la sphère (frontière)
    u = np.linspace(0, 2 * np.pi, 300)
    v = np.linspace(0, np.pi, 150)
    x_sphere = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.8, label='Boundary')
    
    # Dessiner les particules
    colors = ['green', 'orange', 'red']
    labels = ['Particule (no intersection)', 'Particule (small intersection)', 'Particule (large intersection)']
    
    for i, pos in enumerate(particle_positions):
        ax.scatter(*pos, color=colors[i], s=100, label=labels[i])
        add_wireframe_sphere(ax, pos, particle_radius, colors[i])
    
    # Ajouter une colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(r'Precomputed intersection volume $V_B$')
    
    # Configuration de l'axe
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)
    ax.legend()
    ax.view_init(elev=20, azim=-45)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    plt.savefig('Pictures/CH4_splishsplash/volume_maps_plot_3D.pdf')
    
    return fig, ax


def create_2d_visualization(grid_size_2d, visualization_resolution, particle_positions, sphere_center, sphere_radius, particle_radius):
    """Crée la visualisation 2D (coupe) de la grille et des particules"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Créer la grille pour les calculs et visualisation pcolormesh
    x2d_vis = np.linspace(-2, 2, visualization_resolution)
    y2d_vis = np.linspace(-2, 2, visualization_resolution)
    X2d_vis, Y2d_vis = np.meshgrid(x2d_vis, y2d_vis)
    Z2d_vis = np.zeros_like(X2d_vis)  # Coupe à z=0
    
    # Calculer les volumes d'intersection pour la visualisation
    grid_points_2d_vis = np.vstack([X2d_vis.flatten(), Y2d_vis.flatten(), Z2d_vis.flatten()]).T
    volumes_2d_vis = compute_volume_intersection(grid_points_2d_vis, sphere_center, sphere_radius, particle_radius)
    volumes_2d_vis = volumes_2d_vis.reshape(X2d_vis.shape)
    
    # Tracer la carte de couleur
    im = ax.pcolormesh(X2d_vis, Y2d_vis, volumes_2d_vis, cmap='inferno', vmin=0, vmax=0.8)
    
    # Créer la grille pour les nœuds (même taille que la grille de calcul originale)
    x2d_nodes = np.linspace(-2, 2, grid_size_2d)
    y2d_nodes = np.linspace(-2, 2, grid_size_2d)
    X2d_nodes, Y2d_nodes = np.meshgrid(x2d_nodes, y2d_nodes, indexing='ij')
    Z2d_nodes = np.zeros_like(X2d_nodes)  # Coupe à z=0
    
    # Calculer les volumes d'intersection pour les nœuds
    grid_points_2d_nodes = np.vstack([X2d_nodes.flatten(), Y2d_nodes.flatten(), Z2d_nodes.flatten()]).T
    volumes_2d_nodes = compute_volume_intersection(grid_points_2d_nodes, sphere_center, sphere_radius, particle_radius)
    
    # Afficher les nœuds avec couleur correspondant à leur valeur d'intersection
    node_scatter = ax.scatter(X2d_nodes.flatten(), Y2d_nodes.flatten(), 
                             c=volumes_2d_nodes, cmap='inferno', 
                             s=50, alpha=1.0, label='Nodes', 
                             marker='o', edgecolors='white', linewidths=0.5)
    
    # Ajouter le cercle représentant la frontière
    circle = plt.Circle(sphere_center[:2], sphere_radius, fill=False, color='blue', linewidth=2)
    ax.add_artist(circle)
    
    # Dessiner les particules (projection 2D)
    particle_positions_2d = [np.array([pos[0], pos[1]]) for pos in particle_positions]
    colors = ['green', 'orange', 'red']
    labels = ['Particule (no intersection)', 'Particule (small intersection)', 'Particule (large intersection)']
    
    for i, pos in enumerate(particle_positions_2d):
        ax.scatter(*pos, color=colors[i], s=100, label=labels[i])
        particle_circle = plt.Circle(pos, particle_radius, fill=False, color=colors[i], alpha=0.7, linestyle='--')
        ax.add_artist(particle_circle)
        
        # Visualiser l'intersection
        intersection = calculate_circle_intersection(sphere_center[:2], sphere_radius, pos, particle_radius)
        if intersection:
            # Pour la particule
            intersection_points = np.array(intersection)
            angles = np.arctan2(intersection_points[:, 1] - pos[1], intersection_points[:, 0] - pos[0])
            start_angle = np.degrees(angles[0])
            end_angle = np.degrees(angles[1])
            
            if end_angle < start_angle:
                end_angle += 360
            
            wedge = Wedge(pos, particle_radius, start_angle, end_angle, width=0, color=colors[i], alpha=0.5)
            ax.add_artist(wedge)
            
            # Pour la frontière
            boundary_center = sphere_center[:2]
            angles_boundary = np.arctan2(intersection_points[:, 1] - boundary_center[1], 
                                        intersection_points[:, 0] - boundary_center[0])
            start_angle_boundary = np.degrees(angles_boundary[1])
            end_angle_boundary = np.degrees(angles_boundary[0])
            
            if end_angle_boundary < start_angle_boundary:
                end_angle_boundary += 360
            
            wedge_boundary = Wedge(boundary_center, sphere_radius, start_angle_boundary, end_angle_boundary, 
                                  width=0, color=colors[i], alpha=0.5)
            ax.add_artist(wedge_boundary)
    
    # Ajouter une colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Precomputed intersection volume $V_B$')
    
    # Configuration de l'axe
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig('Pictures/CH4_splishsplash/volume_maps_plot_2D.pdf', bbox_inches='tight', pad_inches=0.1)
    
    return fig, ax


def main():
    # Paramètres de base
    grid_resolution = 4  # Résolution de la grille principale (augmentée pour mieux voir)
    vis_resolution = 500  # Résolution de visualisation pour pcolormesh (réduite pour performance)
    sphere_radius = 1.0
    particle_radius = 0.5
    
    # Position de la sphère principale
    sphere_center = np.array([0, 0, 0])
        
    # Positions des particules
    particle_positions = [
        np.array([-1.4, -1.5, -1.5]),  # Loin (sans intersection)
        np.array([1.3, 0.3, 0]),       # Petite intersection
        np.array([0.9, 0.5, 0])        # Intersection moyenne
    ]
    
    # Créer les visualisations
    fig3d, ax3d = create_3d_visualization(4, particle_positions, 
                                         sphere_center, sphere_radius, particle_radius)
    
    fig2d, ax2d = create_2d_visualization(9, vis_resolution, particle_positions, 
                                         sphere_center, sphere_radius, particle_radius)
    
    plt.tight_layout()
    plt.show()
    
    return fig3d, fig2d


if __name__ == "__main__":
    main()