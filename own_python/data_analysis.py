import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configuration du style des graphiques
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 10, 16, 18
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=MEDIUM_SIZE)  
plt.rc('ytick', labelsize=MEDIUM_SIZE)   
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE)

# Configure project path
sys.path.append((os.getcwd()))
from own_python.write_scenes.Tools_scenes import *
from validation_test.turbulent_flow import *
from exporter.plot_vtk import *
from exporter.Tools import *
from exporter.Transfer_data import *
from exporter.shapely_tools import *


def main():

    vtk_folder="output_host/channel_curve_2D/DFSPH/5_m_s/angle_15/no_turbulence"

    nu = 1e-6
    rho = 1e3
    r = 0.02
    h = 4*r
    particle = 2*r

    mm = 1e-3

    # Load vtk files    
    all_vtk = load_vtk_files(vtk_folder)
    #fig = visualize_particles(all_vtk[-1], velocity_component='x', show=False)
    
    fully_dev_idx = 590
    fully_dev_vtk = all_vtk[fully_dev_idx:]
    last_vtk = fully_dev_vtk[-1]

    x_min, y_min = 10, -1.615
    x_max, y_max = 10+ particle*5, 1.615
    dist_y = np.sqrt(y_max**2 - y_min**2)
    U_0 = 5
    Q_init = U_0*dist_y

    u_all = []
    y_all = []

    inside_mask, rectangle = find_particles_in_rectangle(last_vtk.points, x_min, y_min, x_max, y_max)

    projected_points, projected_attributes, vertical_line = project_particles(
            last_vtk, inside_mask, rectangle)
        
    #visualize_results(last_vtk, inside_mask, projected_points, rectangle, vertical_line)

    
    shapely_single_slice(fully_dev_vtk, 
                         y_min, y_max, 
                         x_pos=10,
                         slice_width=particle*5,
                         attribute="density")
    
    '''
    attr_all, y_all = shapely_multiple_slices(fully_dev_vtk[-1],
                                           x_start=25, x_end=45,
                                           num_slices=50,
                                           y_min=-1.615, y_max=1.615,
                                           attribute="density",
                                           slice_width=(2*r)*5)
    '''
    integrate_slice(Q_init, u_all, y_all)
    plt.show()

    

    
    '''
    # Sampling line
    x_slice = 15
    y_max = 1.615
    start = [x_slice, -y_max]
    end = [x_slice, y_max]
    y_line, x_line = create_line(start, end, r, coef=1)
    D = abs(max(y_line) - min(y_line))
    w = D/2

    #---------------------#
    # Turbulence analysis #
    #---------------------#

    # 1. Compute and plot both analytical and numerical velocity profiles
    #computeRe(nu, D, U_0=5)

    u_mean_sph = compute_mean_velocity_profile(fully_dev_vtk, 
                                                y_line, x_line, 
                                                h, sample_rate=0.5)
    U_carac = 6.60
    plot_velocity_profile(y_line, u_mean_sph[:, 0]/U_carac, U_0=1, delta=0.22, gamma=0.86)


    compute_centerline_velocity(fully_dev_vtk, 0.5, 50, h, U_carac, sample_rate=0.5)
    #analyze_particle_distribution(fully_dev_vtk[-1], x_slice, delta_x=10*r, n_bins=100, plot=True)
    '''
    '''
    U_carac = 6.60
    compute_centerline_velocity(fully_dev_vtk, 0.5, 50, h, U_carac, sample_rate=0.5)
    # Analyse different slice over channel to see if fully developed flow is reached
    x_multiple_slices = [5*np.ones(len(y_line)), 10*np.ones(len(y_line)), 15*np.ones(len(y_line)), 20*np.ones(len(y_line)), 25*np.ones(len(y_line))]
    u_mean_sph_multiple_slices = [compute_mean_velocity_profile(fully_dev_vtk, 
                                                                y_line, x_solo_slice, h, 
                                                                sample_rate=0.5) for x_solo_slice in x_multiple_slices]
    
    
    plt.figure()
    for i, u_mean_sph_slice in enumerate(u_mean_sph_multiple_slices):
        plt.plot(y_line, u_mean_sph_slice[:, 0], label=f"x = {x_multiple_slices[i][0]}")
    plt.legend()
    plt.show()
    
    '''
    
    
if __name__ == "__main__":
    main()