import os
import shutil
from pathlib import Path
import glob
import pyvista as pv

def transfer_vtk_files(source_path, destination_path):
    source_dir = Path(source_path)
    output_dir = Path(destination_path)
    
    try:
        # Si le répertoire de destination existe déjà, supprimer tout son contenu
        if output_dir.exists():
            print(f"Destination directory exists. Removing all contents from {output_dir}")
            shutil.rmtree(output_dir)
        
        # Créer le répertoire de destination
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created empty destination directory at {output_dir}")
        
        # Créer un répertoire temporaire pour conserver la même structure
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # Chercher tous les fichiers .vtk dans le répertoire source et les sous-répertoires
        vtk_count = 0
        for vtk_file in source_dir.glob('**/*.vtk'):
            # Calculer le chemin relatif pour préserver la structure
            rel_path = vtk_file.relative_to(source_dir)
            temp_file = temp_dir / rel_path
            
            # S'assurer que le répertoire parent existe
            temp_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Copier d'abord vers le répertoire temporaire
            shutil.copy2(vtk_file, temp_file)
            vtk_count += 1
        
        print(f"Copied {vtk_count} .vtk files to temporary directory")
        
        # Déplacer les fichiers .vtk vers la destination finale
        moved_count = 0
        for vtk_file in temp_dir.glob('**/*.vtk'):
            shutil.move(str(vtk_file), output_dir)
            moved_count += 1
        
        print(f"Successfully moved {moved_count} .vtk files to {output_dir}")
        
    except Exception as e:
        print(f"Error while copying: {e}")
    finally:
        # Nettoyer les fichiers temporaires
        if 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory")


def load_vtk_files(folder_path, print_files):

    pattern = os.path.join(folder_path, "ParticleData_Fluid_*.vtk")
    vtk_files = glob.glob(pattern)
    
    # Trier numériquement par le numéro de fichier (extraction de l'entier dans le nom)
    def extract_number(filename):
        try:
            # Extrait le numéro entre "ParticleData_Fluid_" et ".vtk"
            base_name = os.path.basename(filename)
            number_str = base_name.replace("ParticleData_Fluid_", "").replace(".vtk", "")
            return int(number_str)
        except:
            return 0  # Valeur par défaut en cas d'erreur
    
    vtk_files = sorted(vtk_files, key=extract_number)
    
    # Afficher les fichiers triés
    print(f"Found {len(vtk_files)} files")
    if print_files:
        for file in vtk_files:
            print(f" - {os.path.basename(file)} (#{extract_number(file)})")
    
    # Vérifier également si d'autres fichiers .vtk existent
    all_vtk_files = glob.glob(os.path.join(folder_path, "*.vtk"))
    other_vtk_files = [f for f in all_vtk_files if f not in vtk_files]
    
    print(f"Other .vtk files in directory ({len(other_vtk_files)})")
    if print_files:
        for file in other_vtk_files[:10]:  # Limiter à 10 pour éviter de surcharger la sortie
            print(f" - {os.path.basename(file)}")
        
        if len(other_vtk_files) > 10:
            print(f"   ...and {len(other_vtk_files) - 10} more")
        
    # Continuer avec le reste de la fonction
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
    
    if len(all_vtk) > 0:
        first_vtk = all_vtk[0]
        point_arrays = list(first_vtk.point_data.keys())
        print("Available data in VTK files:")
        print(f"- Data points: {point_arrays}")
    
    return all_vtk
    

if __name__ == "__main__":
    
    source_path = "SPlisHSPlasH/bin/output/free_surface_pressure_2D/vtk"
    destination_path = "output/free_surface/BC/at_outlet/r_001/maybe"
    
    transfer_vtk_files(source_path, destination_path)
    