import docker
import os
import tarfile
import shutil
from pathlib import Path

def copy_from_container(container_id, container_path, local_path):
  
    client = docker.from_env()
    output_dir = Path(local_path)
    temp_dir = output_dir / 'temp'
    tar_path = temp_dir / 'data.tar'
    
    try:
        # Prepare directories
        output_dir.mkdir(exist_ok=True, parents=True)
        temp_dir.mkdir(exist_ok=True)
        
        # Extract container data
        with open(tar_path, 'wb') as f:
            for chunk in client.containers.get(container_id).get_archive(container_path)[0]:
                f.write(chunk)
        
        # Extract archive and move .vtk files
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=temp_dir)
        
        # Move only .vtk files to destination
        for vtk_file in temp_dir.glob('**/*.vtk'):
            shutil.move(str(vtk_file), output_dir)
            
        print(f".vtk files successfully copied to {output_dir}")
    
    except Exception as e:
        print(f"Error while copying : {e}")
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":

    container_id = '6f08d4bc7201923b19e92b8f5d31c84093447dcb0fbe6195eb4bf4ce6102d823'
    container_path = "/opt/sph_container/Code/bin/output/channel_curve_2D/vtk"
    local_path = "output_host/channel_curve_2D/DFSPH/5_m_s/angle_15"
    
    copy_from_container(container_id, container_path, local_path)