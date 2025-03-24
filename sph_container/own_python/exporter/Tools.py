import numpy as np
from pathlib import Path
import pyvista as pv

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

def load_vtk_files(folder_path):
    
    folder = Path(folder_path)
    vtk_paths = folder.glob('ParticleData_Fluid*.vtk') # get all .vtk files
    
    # Sort numerically by extracting the number from the filename
    def get_file_number(path):
        try:
            filename = path.stem  # Name without extension
            number_part = filename.split('_')[-1]
            return int(number_part)
        except (ValueError, IndexError):
            return 0  # Fallback
    
    # Sort by extracted number
    vtk_paths = sorted(vtk_paths, key=get_file_number)
    
    all_vtk = []
    for path in vtk_paths:
        try:
            mesh = pv.read(path)
            all_vtk.append(mesh)
        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    if not all_vtk:
        print("No VTK files found.")
        return
    
    return all_vtk

