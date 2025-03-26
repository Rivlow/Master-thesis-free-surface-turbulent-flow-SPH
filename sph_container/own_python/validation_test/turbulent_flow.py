import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

from own_python.exporter.Tools import *
from own_python.exporter.plot_vtk import *

def compute_u_th(y_line, U_0, delta, gamma):

   
    y_half = np.linspace(0, 1, len(y_line) // 2)  # semi line y

    U_L = U_0 * 4 * y_half * (1 - y_half)  # laminar
    U_T = U_0 * (1 - np.exp(1 - np.exp(y_half / delta)))  # turublent
    U_GHE_half = gamma * U_T + (1 - gamma) * U_L  # hybrid model

    y_full = np.concatenate((-y_half[::-1], y_half))  
    U_full = np.concatenate((U_GHE_half[::-1], U_GHE_half)) 

    return y_full, U_full

def calculate_velocity_gradient(particle_position, particle_velocity, particle_density, 
                               neighboring_particles, h):
    
    grad_v = np.zeros((2, 2))
    
    for pos_j, vel_j, mass_j, dens_j in neighboring_particles:
        r_ij_vec = np.array([pos_j[0] - particle_position[0], pos_j[1] - particle_position[1]])
        r_ij = np.linalg.norm(r_ij_vec)
        
        if r_ij < 1.0e-9 or r_ij > h:
            continue
        
        dir_ij = r_ij_vec / r_ij
        v_ij = np.array([vel_j[0] - particle_velocity[0], vel_j[1] - particle_velocity[1]])
    
        grad_w_ij = gradW(r_ij, dir_ij, h)
        
        for a in range(2):
            for b in range(2):
                grad_v[a, b] += mass_j * v_ij[a] * grad_w_ij[b] / dens_j
     
    
    return grad_v

def calculate_wall_shear_stress(grad_v, wall_normal, viscosity):

    wall_tangent = np.array([-wall_normal[1], wall_normal[0]])
    
    du_tan_dn = 0
    for a in range(2):
        for b in range(2):
            du_tan_dn += wall_tangent[a] * grad_v[a, b] * wall_normal[b]
    
    return viscosity * du_tan_dn



def calculate_u_plus(vtk_data, wall_positions, nu, wall='bottom'):
    
    points = vtk_data.points
    velocities = vtk_data.point_data['velocity']
    
    # Get u* using existing function
    y_plus = compute_y_plus(vtk_data, wall_positions, nu)
    u_star_top, u_star_bot = compute
    
    # Select the appropriate wall
    if wall == 'bottom':
        u_star = u_star_bot
        wall_y = wall_positions[0]
    else:
        u_star = u_star_top
        wall_y = wall_positions[1]
    
    # Calculate distances from wall
    y_coords = points[:, 1]
    if wall == 'bottom':
        dist_wall = y_coords - wall_y
        sort_idx = np.argsort(dist_wall)
    else:
        dist_wall = wall_y - y_coords
        sort_idx = np.argsort(dist_wall)
    
    # Keep only positive distances
    positive_mask = dist_wall > 0
    dist_wall = dist_wall[positive_mask]
    sort_idx = sort_idx[positive_mask]
    
    # Calculate y+
    y_plus = dist_wall * u_star / nu
    
    # Calculate u+
    u_x = velocities[positive_mask, 0]  # x-component
    u_plus = u_x / (u_star + 1e-10)  # Avoid division by zero
    
    return y_plus, u_plus

def compute_y_plus(u_star, y_line, nu):
    
    dist = abs(max(y_line) - min(y_line))
    y_wall = y_line[y_line < 0.1*dist]
    
    return u_star * y_wall / nu



def plot_wall_law(vtk_data, wall_positions, nu, wall='bottom', rho=1000):
   
    points = vtk_data.points
    velocities = vtk_data.point_data['velocity']
    
    # Get y+ values using the existing functions
    y_plus_bot, y_plus_top, u_star_bot, u_star_top = calculate_y_plus(vtk_data, wall_positions, nu)
    
    # Choose which wall to analyze
    if wall == 'bottom':
        y_plus = y_plus_bot
        u_star = u_star_bot
        wall_y = wall_positions[0]
        wall_normal = np.array([0, 1])  # Normal points up from bottom wall
    else:  # top wall
        y_plus = y_plus_top
        u_star = u_star_top
        wall_y = wall_positions[1]
        wall_normal = np.array([0, -1])  # Normal points down from top wall
    
    # Find particles near the selected wall
    y_coords = points[:, 1]
    if wall == 'bottom':
        dist_wall = y_coords - wall_y
    else:
        dist_wall = wall_y - y_coords
    
    # Define threshold for near-wall particles
    wall_threshold = 0.1 * (wall_positions[1] - wall_positions[0])
    near_wall_mask = dist_wall < wall_threshold
    
    near_wall_points = points[near_wall_mask]
    near_wall_velocities = velocities[near_wall_mask]
    
    # Calculate u+ = u/u*
    # Sort by distance from wall
    if wall == 'bottom':
        sort_idx = np.argsort(near_wall_points[:, 1])
        sorted_dist = near_wall_points[sort_idx, 1] - wall_y
    else:
        sort_idx = np.argsort(-near_wall_points[:, 1])
        sorted_dist = wall_y - near_wall_points[sort_idx, 1]
    
    sorted_vel = near_wall_velocities[sort_idx, 0]  # x-component of velocity
    u_plus = sorted_vel / (u_star + 1e-10)  # u+ = u/u*
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot measurement data
    ax.scatter(y_plus, u_plus, label='SPH simulation', alpha=0.7)
    
    # Plot theoretical wall laws
    y_plus_theory = np.logspace(-1, 3, 1000)
    
    # Viscous sublayer: u+ = y+
    viscous_sublayer = y_plus_theory.copy()
    ax.plot(y_plus_theory[y_plus_theory < 5], 
            viscous_sublayer[y_plus_theory < 5], 
            'r--', label='Viscous sublayer: $u^+ = y^+$')
    
    # Log law: u+ = (1/κ)*ln(y+) + B
    kappa = 0.41  # von Karman constant
    B = 5.5       # Integration constant
    log_law = (1/kappa) * np.log(y_plus_theory) + B
    ax.plot(y_plus_theory[y_plus_theory > 30], 
            log_law[y_plus_theory > 30], 
            'g--', label='Log law: $u^+ = \\frac{1}{\\kappa}\\ln(y^+) + B$')
    
    # Spalding law (covers all regions)
    k = 0.41      # von Karman constant
    B = 5.5       # Integration constant
    E = np.exp(-k*B)
    y_plus_spalding = np.linspace(0.1, 1000, 1000)
    u_plus_spalding = np.zeros_like(y_plus_spalding)
    
    for i, yp in enumerate(y_plus_spalding):
        u = 0.1    # Initial guess
        for _ in range(50):  # Simple iterative solver
            f = u + E * (np.exp(k*u) - 1 - k*u - (k*u)**2/2 - (k*u)**3/6) - yp
            df = 1 + E * (k*np.exp(k*u) - k - k**2*u - (k*u)**2*k/2)
            u = u - f/df
            if abs(f) < 1e-10:
                break
        u_plus_spalding[i] = u
    
    ax.plot(y_plus_spalding, u_plus_spalding, 'k-', label='Spalding law')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('$y^+$')
    ax.set_ylabel('$u^+$')
    ax.set_title(f'Wall Law - {wall.capitalize()} Wall')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Highlight different regions
    ax.axvspan(0, 5, alpha=0.1, color='red', label='Viscous sublayer')
    ax.axvspan(5, 30, alpha=0.1, color='yellow', label='Buffer layer')
    ax.axvspan(30, 300, alpha=0.1, color='green', label='Log-law region')
    
    plt.tight_layout()
    plt.show()
    
    return fig
