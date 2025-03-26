import pyvista as pv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Configure project path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
from sph_container.own_python.validation_test.turbulent_flow import *
from exporter.plot_vtk import *
from exporter.Tools import *
from exporter.Transfer_data import *

def main():

    vtk_folder="output_host/channel_curve_2D/DFSPH/5_m_s/angle_15"

    nu = 0.5e-4
    rho = 1e3
    r = 0.02
    h = 4*r
    
    all_vtk = load_vtk_files(vtk_folder)
    fig = visualize_particles(all_vtk[-1], velocity_component='x', show=False)
    
    fully_dev_idx = 1300
    fully_dev_vtk = all_vtk[fully_dev_idx:]
    
    # Sampling line
    x_slice = 25
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
    u_mean_sph = compute_mean_velocity_profile(fully_dev_vtk, 
                                                y_line, x_line, 
                                                h, sample_rate=0.5)
    
    u_th = compute_u_th(y_line, U_0=5, delta=0.01, gamma=0.71)
    plot_velocity_profile(y_line, u_mean_sph , u_th)
    
    
    # 2. Compute and plot the y+ distribution and wall law
    last_vtk = fully_dev_vtk[-1]
    wall_positions = (-y_max, y_max)  # Positions des parois (inférieure, supérieure)

    # Analyser la loi de paroi (pour la paroi inférieure)
    # Extraire la composante horizontale de la vitesse
    velocities = last_vtk.point_data['velocity']
    u_profile = velocities[:, 0]
    
    # Calculer u+ = u/u*
    grad_v = calculate_velocity_gradient(particle_position, particle_velocity, 
                                        particle_density, neighboring_particles, h)
    
    tau_w = calculate_wall_shear_stress(grad_v, wall_normal, viscosity)
    u_star = np.sqrt(abs(tau_w) / rho)
    y_plus  = compute_y_plus(u_star, y_line, nu)
    eta = y_line/w

    y_near_wall = y_line[y_line < 0.1*D]
    y_plus = u_star*y_near_wall/nu
    
    # Tracer la loi de paroi
    plot_wall_law(y_plus_bottom, u_plus)
    plt.savefig("loi_paroi.png", dpi=300)
    plt.show()
    
    
if __name__ == "__main__":
    main()