import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Configure project path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
from python.validation_test.turbulent_flow_pipe import * 
from python.exporter.Tools import *

SMALL_SIZE = 10
MEDIUM_SIZE = 14
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

def create_line(start, end, r, coef):

    dist = np.sqrt((end[1] - start[1])**2 + (end[0] - start[0])**2)
    nb_part = coef*int(dist/(r))


    # y_f - y_i = m*(x_f - x_i)
    if end[0] != start[0]:
        slope = (end[1] - start[1])/(end[0] - start[0])
        x = np.linspace(start[0], end[0], nb_part)
        y = slope*(x - start[0]) + start[1]
    # vertical line
    else:
        y = np.linspace(start[1], end[1], nb_part)
        x = start[0]*np.ones(len(y))

    return y, x


def visualize_particles(vtk_data, dimension='3D', velocity_component='magnitude', show=True, plot_backend='pyvista'):

    # Calculate velocity magnitude or component
    if 'velocity' in vtk_data.array_names and velocity_component == 'magnitude':
        velocity = vtk_data['velocity']
        magnitude = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2 + 
                           (velocity[:, 2]**2 if velocity.shape[1] > 2 else 0))
        vtk_data['velocity_magnitude'] = magnitude
        scalar = 'velocity_magnitude'
        scalar_values = magnitude
    elif velocity_component in ['x', 'y', 'z']:
        idx = {'x': 0, 'y': 1, 'z': 2}[velocity_component]
        vtk_data[f'velocity_{velocity_component}'] = vtk_data['velocity'][:, idx]
        scalar = f'velocity_{velocity_component}'
        scalar_values = vtk_data['velocity'][:, idx]
    else:
        scalar = 'velocity'
        scalar_values = np.linalg.norm(vtk_data['velocity'], axis=1)
    
    if plot_backend.lower() == 'pyvista':
        # PyVista visualization (original implementation)
        plotter = pv.Plotter()
        
        if dimension == '2D':
            # In 2D, ignore Z dimension
            plotter.view_xy()
            plotter.enable_parallel_projection()
        
        # Add points with colorbar
        plotter.add_points(vtk_data, 
                          render_points_as_spheres=True,
                          point_size=10,
                          scalars=scalar,
                          cmap='viridis')
        
        plotter.show_axes()
        plotter.show_grid()
        
        if show:
            plotter.show()
        
        return plotter
    
    elif plot_backend.lower() == 'pyplot':
        # Matplotlib pyplot visualization
        points = vtk_data.points
        
        if dimension == '3D':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Scatter plot with colors based on scalar values
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=scalar_values, cmap='viridis', 
                               s=30, alpha=0.8)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Particles Colored by Velocity {velocity_component}')
            
        else:  # 2D

            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot with colors based on scalar values
            scatter = ax.scatter(points[:, 0], points[:, 1], 
                               c=scalar_values, cmap='viridis', 
                               s=30, alpha=0.8)
    
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Particles Colored by Velocity {velocity_component}')
            ax.grid(True)
            ax.set_aspect('equal')
        
        plt.colorbar(scatter, ax=ax, label=f'Velocity {velocity_component}')
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig
    
    else:
        raise ValueError(f"Unsupported plot_backend: {plot_backend}. Use 'pyvista' or 'pyplot'.")

def compute_u_sph(vtk_data, y, x, smoothing_radius, kernel='cubic'):
    
    # Get particle positions and velocities
    points = vtk_data.points
    velocities = vtk_data['velocity']
    u = np.zeros((len(y), velocities.shape[1]))
    
    # Define kernel functions
    kernel_functions = {'gaussian': lambda r, h: np.exp(-(r/h)**2),
                        'cubic': lambda r, h: (1 - r/h)**3 if r/h < 1 else 0}
    
    kernel_func = kernel_functions.get(kernel, kernel_functions['cubic'])
    
    # Track valid sampling points
    valid_indices = []
   
    # Find particles within the neighbourhood
    for i in range(len(y)):

        y_pos = y[i]
        x_pos = x[i]

        dx = points[:, 0] - x_pos
        dy = points[:, 1] - y_pos
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist <= smoothing_radius
        
        # If inside neighbourhood
        if np.any(mask):

            weights = np.array([kernel_func(r, smoothing_radius) for r in dist[mask]])
            total_weight = np.sum(weights)
            
            if total_weight > 1e-6: # avoid numerical instabilities
                weighted_velocity = np.sum(weights[:, np.newaxis] * velocities[mask], axis=0) / total_weight
                u[i] = weighted_velocity
                valid_indices.append(i)
    
    if valid_indices:
        return y[valid_indices], u[valid_indices]
    else:
        print('Warning: any valid positions found')
        return np.array([]), np.array([])
        

def plot_velocity_profile(y_pos, u_sph, u_th, sym, component='magnitude', latex=False):
    if len(y_pos) == 0:
        return None  # No valid data to plot
    
    plt.figure(figsize=(10, 6))
    
    # Extract the velocity data based on component
    if component == 'magnitude':
        velocity = np.linalg.norm(u_sph, axis=1) if u_sph.ndim > 1 else u_sph
    elif component == 'x':
        velocity = u_sph[:, 0] if u_sph.ndim > 1 else u_sph
    
    # Plot u_sph directly without forcing symmetry
    plt.plot(y_pos, velocity, 'o-', label=r'$u_{SPH}(x)$')
    
    # Only create symmetric profile for u_th (theoretical/experimental)
    if sym:
        # Step 1: Create a mirrored set of points
        mirror_y = -y_pos
        
        # Step 2: Combine original and mirrored points
        full_y = np.concatenate([mirror_y, y_pos])
        full_u_th = np.concatenate([u_th, u_th])
        
        # Step 3: Sort and remove duplicates
        idx = np.argsort(full_y)
        full_y = full_y[idx]
        full_u_th = full_u_th[idx]
        
        # Remove duplicates (keeping only unique positions)
        if len(full_y) > 1:
            unique_idx = np.concatenate([np.array([True]), np.abs(np.diff(full_y)) > 1e-10])
            full_y = full_y[unique_idx]
            full_u_th = full_u_th[unique_idx]
        
        # Plot the symmetric theoretical profile
        plt.plot(full_y, full_u_th, label=r'$u_{theoretical}(x)$')
    else:
        # Plot u_th directly without symmetry
        plt.plot(y_pos, u_th, label=r'$u_{theoretical}(x)$')
    
    isLatex(latex)
    plt.xlabel(r'Pipe diameter D')
    plt.ylabel(r'Velocity profile $\frac{u(y)}{u_{max}}$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def find_periodicity(all_vtk, y_line, x_line, neighbourhood, kernel='cubic'):

    ref_vtk = all_vtk[0]
    y, u_sph_ref = compute_u_sph(ref_vtk, y_line, x_line, neighbourhood, kernel)

    # We iterate over all the vtk files until we find a similar velocity profile
    for i, vtk in enumerate(all_vtk[1:]):
        y, u_sph_i = compute_u_sph(vtk, y_line, x_line, neighbourhood, kernel)
        error = np.linalg.norm(u_sph_ref - u_sph_i)
        if error < 1e-3:
            periodicity = vtk.time
            i_period = i
            break

    return periodicity, i_period

def compute_mean_velocity_profile(all_vtk, y_line, x_line, i_period, neighbourhood, kernel='cubic'):

    # Compute each velocity profile
    u_sph_all = []
    y_pos_all = []
    
    for vtk in all_vtk:
        y, u = compute_u_sph(vtk, y_line, x_line, neighbourhood, kernel)
        u_sph_all.append(u)
        y_pos_all.append(y)
    
    u_sph_all = np.array(u_sph_all)

    # Now we can compute the mean velocity profile
    u_sph_mean = np.mean(u_sph_all[:i_period], axis=0)

    return y, u_sph_mean

