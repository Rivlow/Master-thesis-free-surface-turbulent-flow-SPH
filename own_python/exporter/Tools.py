import numpy as np
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
import os
import glob


def setup_latex(use_latex=False):

    if use_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')

def load_vtk_files(folder_path):
    
    vtk_files = sorted(glob.glob(os.path.join(folder_path, "ParticleData_Fluid*.vtk")))
    all_vtk = []

    for file in vtk_files:
        try:
            vtk_data = pv.read(file)
            all_vtk.append(vtk_data)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(f"Successfully loaded {len(all_vtk)} VTK files")

    if not all_vtk:
        print("No VTK files found or failed to load")
        return all_vtk

    first_vtk = all_vtk[-1]
    point_arrays = list(first_vtk.point_data.keys())
    cell_arrays = list(first_vtk.cell_data.keys())
    field_arrays = list(first_vtk.field_data.keys())

    print("Available data in the selected VTK file:")
    print(f"- Data points: {point_arrays}")

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
   

