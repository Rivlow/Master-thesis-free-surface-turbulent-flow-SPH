import numpy as np
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
import os
import glob

def computeRe(nu, D, U_0):
    Re = int((U_0*D)/nu)
    print(f"Maximal velocity = {U_0}")
    print(f'Reynolds number Re = {Re}')

    print('max desired = ', 970000*nu/D)

    if Re < 2300:
        print('Laminar flow')
    else:
        print('Turbulent flow')

def print_section_ratio(D_init, D_fin, U_0):

    ratio = 0.88*D_fin/D_init # 0.88 = magic
    print(f"aspect ratio D_final/D_init:  {ratio}")
    print(f"Estimated velocity at final section: {U_0/ratio}")


def setup_latex(use_latex=False):

    if use_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')

def load_vtk_files(folder_path):
    # Utilisation de glob pour ne prendre que les fichiers commençant par "ParticleFluid"
    vtk_files = sorted(glob.glob(os.path.join(folder_path, "ParticleData_Fluid*.vtk")))
    all_vtk = []
    
    for file in vtk_files:
        try:
            vtk_data = pv.read(file)
            all_vtk.append(vtk_data)
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {e}")
    
    print(f"Chargement de {len(all_vtk)} fichiers VTK réussi")

    if not all_vtk:
        print("Aucun fichier VTK trouvé ou erreur de chargement")
        return all_vtk
    
    # Informations sur le premier fichier chargé
    first_vtk = all_vtk[-1]
    point_arrays = list(first_vtk.point_data.keys())
    cell_arrays = list(first_vtk.cell_data.keys())
    field_arrays = list(first_vtk.field_data.keys())
    
    print(" Données disponibles dans les fichiers VTK sélectionnés:")
    print(f"- Données aux points: {point_arrays}")
    print(f"- Données aux cellules: {cell_arrays}")
    print(f"- Données globales: {field_arrays}")
    
    return all_vtk

def W(r, h):

    q = r / h
    pi = np.pi
    k = 40.0 / (7.0 * pi * h**2)
    
    if q <= 1.0:
        if q <= 0.5:
            return k * (6.0 * q**3 - 6.0 * q**2 + 1.0)
        else:
            return k * (2.0 * (1.0 - q)**3)
    else:
        return 0.0

def gradW(r, dir_vector, h):

    if r > h or r < 1.0e-9:
        return np.zeros_like(dir_vector)
    
    q = r / h
    pi = np.pi
    l = 240.0 / (7.0 * pi * h**2)
    
    if q <= 0.5:
        dwdr = l * q * (3.0 * q - 2.0) / h
    else:
        factor = 1.0 - q
        dwdr = l * (-factor * factor) / h
    
    return dwdr * dir_vector
   

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
    


