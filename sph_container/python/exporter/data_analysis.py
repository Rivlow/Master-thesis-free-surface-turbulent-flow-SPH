import pyvista as pv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configure project path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
from python.validation_test.turbulent_flow_pipe import * 
from python.exporter.Tools import *
from plot_vtk import *


def main():

    # Parameters
    r = 0.005
    nu = 1e-6
    neighbourhood = 4*r  # influence neighborhood
    
    # Load VTK files (only fluid particles)
    vtk_folder = "output_host/channel_curve_2D/DFSPH/5_m_s/angle_35"
    all_vtk = load_vtk_files(vtk_folder)
    
    # Plot single vtk frame
    vtk = all_vtk[-1]
    visualize_particles(vtk, dimension='2D', plot_backend='pyplot', show=True)

    # Assuming the fluid create some periodic vortex, we need to find first the periodicity of the flow
    start = [3, -0.49-2*r]
    end = [3, 0.49+2*r]
    y_line, x_line = create_line(start, end, r, coef=1)

    x_slice = 3
    periodicity, i_period = find_periodicity(all_vtk, y_line, x_line, neighbourhood, kernel='cubic')

    # Compute mean veloticty profile
    y, u_sph_mean = compute_mean_velocity_profile(all_vtk, x_slice, i_period, neighbourhood, kernel='cubic')

    # Flow information
    D = np.sqrt((end[1] - start[1])**2)
    computeRe(nu, D, 5.6)

    #u_th = compute_u_th(np.linspace(0, 0.5, len(y)), np.max(u_sph), delta=0.010, gamma=0.29) # ''''good'''' with U_0 = 5 and -y, y = -0.5, 0.5
    #u_th_norm = u_th
    #plot_velocity_profile(np.linspace(0, 0.5, len(y)), u_sph_norm, u_th_norm, sym=True, component='x', latex=False)

    plt.show()

if __name__ == "__main__":
    main()