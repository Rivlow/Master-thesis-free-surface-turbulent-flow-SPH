import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from scipy.optimize import minimize


sys.path.append("c:\\Users\\lucas\\Unif\\TFE\\Code")
from sph_container.own_python.exporter.Tools import *
from sph_container.own_python.exporter.plot_vtk import *

import vtk
import numpy as np
from shapely.geometry import Point, LineString, box
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from scipy.integrate import simpson

def compute_u_ghe(y_line, U_0, delta, gamma):
    y_half = np.linspace(0, 0.5, len(y_line)//2)  # semi line y
    
    U_L = U_0 * 4 * y_half * (1 - y_half)  # laminar
    U_T = U_0 * (1 - np.exp(1 - np.exp(y_half / delta)))  # turublent
    U_GHE_half = gamma * U_T + (1 - gamma) * U_L  # hybrid model
    
  
    y_full = np.concatenate((-y_half[::-1], y_half))
    U_full = np.concatenate((U_GHE_half, U_GHE_half[::-1])) 
    U_L_full = np.concatenate((U_L, U_L[::-1]))
    U_T_full = np.concatenate((U_T, U_T[::-1]))

        
    return U_full, U_L_full, U_T_full


def compute_velocity_slice(vtk_data, y_line, x_line, h):
    
    points = vtk_data.points
    velocities = vtk_data['velocity']
    
    u = np.zeros((len(y_line), velocities.shape[1]))
    
    for i in range(len(y_line)):
        y_pos = y_line[i]
        x_pos = x_line[i]
        
        dx = points[:, 0] - x_pos
        dy = points[:, 1] - y_pos
        dist = np.sqrt(dx**2 + dy**2)
        
        mask = dist <= h
        
        if np.any(mask):

            weights = np.array([W(r, h) for r in dist[mask]])
            total_weight = np.sum(weights)
            
            if total_weight > 1e-8:  # instability threshold
                weighted_velocity = np.sum(weights[:, np.newaxis] * velocities[mask], axis=0) / total_weight
                u[i] = weighted_velocity
            else:
                u[i] = np.zeros(velocities.shape[1])

    return u

def compute_mean_velocity_profile(vtk_data_list, 
                                  y_line, x_line, 
                                  h, sample_rate=1.0):
    

    # Particles are moving slowly, don't look at each timestep
    n_files = len(vtk_data_list)
    n_samples = max(1, int(n_files * sample_rate))
    
    if sample_rate < 1.0:
        indices = np.linspace(0, n_files-1, n_samples, dtype=int)
        sampled_vtk = [vtk_data_list[i] for i in indices]
    else:
        sampled_vtk = vtk_data_list
    
    u_all_slice = [compute_velocity_slice(vtk, y_line, x_line, h) for vtk in sampled_vtk]
    
    
    return np.mean(u_all_slice, axis=0)

def compute_centerline_velocity(vtk_data_list, x_start, x_end, h, U_carac, sample_rate=1.0):
    
    # Échantillonnage des fichiers VTK
    n_files = len(vtk_data_list)
    n_samples = max(1, int(n_files * sample_rate))
    
    if sample_rate < 1.0:
        indices = np.linspace(0, n_files-1, n_samples, dtype=int)
        sampled_vtk = [vtk_data_list[i] for i in indices]
    else:
        sampled_vtk = vtk_data_list
    
    # Création de la ligne centrale
    n_points = int((x_end - x_start) / (h/4))  # Résolution appropriée
    x_line = np.linspace(x_start, x_end, n_points)
    y_center = 0.0  # Position centrale en y
    
    # Initialisation du tableau pour stocker les résultats
    u_center_all = np.zeros((len(sampled_vtk), len(x_line)))
    
    # Calcul pour chaque pas de temps
    for t, vtk in enumerate(sampled_vtk):
        points = vtk.points
        velocities = vtk['velocity']
        
        for i, x_pos in enumerate(x_line):
            # Calcul de la distance entre les particules et le point central
            dx = points[:, 0] - x_pos
            dy = points[:, 1] - y_center
            dist = np.sqrt(dx**2 + dy**2)
            
            # Sélection des particules dans le rayon de lissage
            mask = dist <= h
            
            if np.any(mask):
                # Calcul de la vitesse avec une moyenne pondérée (noyau SPH)
                weights = np.array([W(r, h) for r in dist[mask]])
                total_weight = np.sum(weights)
                
                if total_weight > 1e-8:  # Seuil de stabilité
                    # Extraction des vitesses longitudinales (composante x)
                    u_x = velocities[mask, 0]
                    u_center_all[t, i] = np.sum(weights * u_x) / total_weight
    
    # Moyenne sur tous les pas de temps
    u_center = np.mean(u_center_all, axis=0)

    plt.figure()
    plt.plot(x_line, u_center/U_carac)
    plt.xlabel("x [m]")
    plt.ylabel("u [m/s]")
    plt.title("Centerline velocity profile")
    plt.grid()
    

def analyze_particle_distribution(vtk_data, x_slice, delta_x=0.1, n_bins=50, plot=True):
   
    
    points = vtk_data.points
    x_pos = points[:, 0]
    mask = (x_pos >= x_slice - delta_x/2) & (x_pos <= x_slice + delta_x/2) # [x - dx/2, x + dx/2]
    slice_points = points[mask]
    
    if len(slice_points) == 0:
        print(f"No particles found at  x = {x_slice} +- {delta_x/2}")
        return None, (None, None, None)
    
  
    # Y domain (for bins)
    y_pos = slice_points[:, 1]
    y_min, y_max = np.min(y_pos), np.max(y_pos)
    y_range = y_max - y_min
    bins = np.linspace(y_min - 0.05*y_range, y_max + 0.05*y_range, n_bins)
    
    # Histogram
    counts, bin_edges = np.histogram(y_pos, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Find potential low density regions
    bin_width = bin_edges[1] - bin_edges[0]
    density_per_bin = counts / bin_width
    mean_density = np.mean(density_per_bin)
    high_density_threshold = 1.5 * mean_density
    low_density_threshold = 0.5 * mean_density
    
    high_density_regions = bin_centers[density_per_bin > high_density_threshold]
    low_density_regions = bin_centers[density_per_bin < low_density_threshold]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(bin_centers, counts, width=bin_width*0.9, alpha=0.7, color='steelblue')
        
        if len(high_density_regions) > 0:
            for y_pos in high_density_regions:
                ax.axvline(x=y_pos, color='red', linestyle='--', alpha=0.5)
        
        if len(low_density_regions) > 0:
            for y_pos in low_density_regions:
                ax.axvline(x=y_pos, color='orange', linestyle='--', alpha=0.5)
                
        ax.set_xlabel('Position y')
        ax.set_ylabel('Number of particules')
        ax.set_title(f'Particle distribution at x = {x_slice} +- {delta_x/2}')
        ax.grid(True, alpha=0.3)
        
        ax.legend()
        plt.tight_layout()
        

def computeRe(nu, D, U_0):
    Re = int((U_0*D)/nu)
    print(f"Maximal velocity = {U_0}")
    print(f'Reynolds number Re = {Re}')

    if Re < 2300:
        print('Laminar flow')
    else:
        print('Turbulent flow')




def find_particles_in_rectangle(points, x_min, y_min, x_max, y_max):

    rectangle = box(x_min, y_min, x_max, y_max)
    inside_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        shapely_point = Point(point[0], point[1])
        inside_mask[i] = rectangle.contains(shapely_point)
    
    return inside_mask, rectangle

def project_particles(vtk_data, mask, rectangle):

    # Particles inside the rectangle
    points = vtk_data.points
    inside_points = points[mask]
    
    min_x, min_y, max_x, maxy = rectangle.bounds
    middle_x = (min_x + max_x) / 2
    vertical_line = LineString([(middle_x, min_y), (middle_x, maxy)])
    
    # Project (orthogonaly) the particles on the vertical line
    projected_points = np.zeros_like(inside_points)
    for i, point_coords in enumerate(inside_points):

        shapely_point = Point(point_coords[0], point_coords[1])
        
        _, projected = nearest_points(shapely_point, vertical_line)
        
        projected_points[i, 0] = projected.x
        projected_points[i, 1] = projected.y

        # Keep the z coordinate if it exists
        if inside_points.shape[1] > 2:
            projected_points[i, 2] = point_coords[2] 
    
    projected_attributes = {}
    point_arrays = list(vtk_data.point_data.keys())
    
    for key in point_arrays:
        values = vtk_data[key]
        projected_attributes[key] = values[mask]
    
    # Sort the projected points by y coordinate
    if len(projected_points) > 0:
        sort_indices = np.argsort(projected_points[:, 1])
        projected_points = projected_points[sort_indices]
        
        for key in projected_attributes:
            projected_attributes[key] = projected_attributes[key][sort_indices]
    
    return projected_points, projected_attributes, vertical_line


def shapely_single_slice(fully_dev_vtk, 
                         y_min, y_max, 
                         x_pos, 
                         slice_width,
                         attribute):


    attr_all = []
    y_all = []

    x_min = x_pos - slice_width/2
    x_max = x_pos + slice_width/2

    for i, single_vtk in enumerate(fully_dev_vtk):
        
        if i == 0:
            continue

        inside_mask, rectangle = find_particles_in_rectangle(single_vtk.points, x_min, y_min, x_max, y_max)

        projected_points, projected_attributes, vertical_line = project_particles(
            single_vtk, inside_mask, rectangle)
        
        if attribute=="velocity":
            attr_all.append(projected_attributes['velocity'][:, 0])
        elif attribute=="density":
            attr_all.append(projected_attributes['density'])
        y_all.append(projected_points[:, 1])

    for i in range(len(attr_all)):
        plt.scatter(y_all[i], attr_all[i], s=5)
        
    plt.legend()
    plt.savefig(f'Pictures/turbulent/single_slice_{attribute}_x_{x_pos}.png')
    

def shapely_multiple_slices(vtk_data, x_start, x_end, num_slices, y_min, y_max, attribute, slice_width):
    
    x_positions = np.linspace(x_start, x_end, num_slices)
    attr_all = []
    y_all = []
    
    plt.figure(figsize=(12, 8))
    
    for x_pos in x_positions:

        x_min = x_pos - slice_width/2
        x_max = x_pos + slice_width/2
        
        inside_mask, rectangle = find_particles_in_rectangle(vtk_data.points, x_min, y_min, x_max, y_max)        
        projected_points, projected_attributes, vertical_line = project_particles(
            vtk_data, inside_mask, rectangle)
        
        if attribute == "velocity":
            attr = projected_attributes['velocity'][:, 0]
        elif attribute == "density":
            attr = projected_attributes['density']
        
        attr_all.append(attr)
        
        if len(projected_points) > 0:
            plt.scatter(projected_points[:, 1], attr, s=3, 
                      label=f"x = {x_pos:.2f}", alpha=0.7)
    
    plt.ylabel('Velocity')
    plt.xlabel('y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Pictures/turbulent/mult_slices_{attribute}.png')

    return attr_all, y_all

def visualize_results(vtk_data, inside_mask, projected_points, rectangle, vertical_line):

    plt.figure(figsize=(10, 8))
    
    points = vtk_data.points
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5, label='All particles')
    inside_points = points[inside_mask]
    plt.scatter(inside_points[:, 0], inside_points[:, 1], s=3, c='red', label='Particles in rectangle')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=3, c='green', label='Projections')
    
    x, y = rectangle.exterior.xy
    plt.plot(x, y, 'b-', label='Rectangle')
    x, y = vertical_line.xy
    plt.plot(x, y, 'g-', linewidth=2, label='Vertical line')
    plt.axis('equal')
    plt.xlim(0.8*rectangle.bounds[0] , 1.2*rectangle.bounds[2])
    plt.ylim(rectangle.bounds[1] - 0.3 , 1.2*rectangle.bounds[3])
    plt.grid(True)
    plt.savefig('Pictures/turbulent/shapely_visualization.png')


def integrate_slice(Q_init, u_all, y_all):
    integrals = []
    
    for u, y in zip(u_all, y_all):
        u = np.array(u)
        y = np.array(y)
        
        # Vérifier que les points sont triés par y croissant
        if not np.all(np.diff(y) > 0):
            sort_idx = np.argsort(y)
            y = y[sort_idx]
            u = u[sort_idx]
            
        # Intégration avec Simpson (meilleur choix pour données discrètes)
        integral_value = simpson(u, x=y)
        print(f"Intégrale pour la slice : {integral_value}")
        integrals.append(integral_value)
    
    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    plt.hlines(np.ones(len(integrals)) * Q_init, 0, len(integrals), color='red', linestyle='--', label='Q_init')
    plt.bar(range(len(integrals)), integrals)
    plt.xlabel('Numéro de slice')
    plt.ylabel('Intégrale ∫u(y)dy')
    plt.grid(True)
    plt.savefig('Pictures/turbulent/slice_integrals.png')

    
    return integrals

