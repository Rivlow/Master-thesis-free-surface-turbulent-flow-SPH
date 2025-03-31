import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import glob 

sys.path.append("c:\\Users\\lucas\\Unif\\TFE\\Code")
from sph_container.own_python.validation_test.turbulent_flow import *

# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 10, 14, 18
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=MEDIUM_SIZE)  
plt.rc('ytick', labelsize=MEDIUM_SIZE)   
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE)


def create_line(start, end, r, coef=1):

    dist = np.sqrt((end[1] - start[1])**2 + (end[0] - start[0])**2)
    nb_part = coef * int(dist / (2*r))

    # Non vertical line
    if end[0] != start[0]: 
        slope = (end[1] - start[1]) / (end[0] - start[0])
        x = np.linspace(start[0], end[0], nb_part)
        y = slope * (x - start[0]) + start[1]
    # Vertical line
    else:  
        y = np.linspace(start[1], end[1], nb_part)
        x = start[0] * np.ones(len(y))
    
    return y, x

def visualize_particles(vtk_data, velocity_component='magnitude', show=True):
    
    points = vtk_data.points
    data = vtk_data.point_data
    
    v = data['velocity']
    v_x = v[:,0]
    v_mag = np.linalg.norm(v, axis=1)

    if velocity_component=="magnitude":
        scalar_values = v_mag
    elif velocity_component=="x":
        scalar_values = v_x

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(points[:, 0], points[:, 1], 
                        c=scalar_values, cmap='viridis', 
                        s=30, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')

    plt.colorbar(scatter, ax=ax, label=f'Velocity {velocity_component}')
    
    if show:
        plt.tight_layout()
        plt.show()

    return fig

def plot_velocity_profile(y_line, u_numerical, U_0, delta, gamma):
    
    u_ghe, u_l, u_t = compute_u_th(y_line, U_0, delta, gamma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_line, u_numerical, 'o', label='SPH')
    plt.plot(y_line, u_ghe, '-', label=f'GHE (δ={delta:.4f}, γ={gamma:.4f})')
    #plt.plot(y_line, u_l, '--', label='Laminaire')
    #plt.plot(y_line, u_t, '-.', label='Turbulent')
    plt.xlabel('y/h')
    plt.ylabel('u/U_0')
    plt.title('Velocity profile')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
