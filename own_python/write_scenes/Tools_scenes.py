import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#---------------------------------#
#   General functional function   #
#---------------------------------#

def clean_files(path):
   
    try:
       
        if not os.path.exists(path):
            print(f"Path '{path}' does not exist.")
            return False
            
        if not os.path.isdir(path):
            print(f"'{path}' is not a folder.")
            return False
            
      
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            
          
            if os.path.isfile(item_path) or os.path.islink(item_path):
               
                os.unlink(item_path)
            elif os.path.isdir(item_path):

                shutil.rmtree(item_path)
                
        print(f"Cleaning folder '{path}' successfully.")
        return True
        
    except Exception as e:
        print(f"Error while cleaning folder: {str(e)}")
        return False
    

#----------------------#
#   UTILITY FUNCTIONS  #
#----------------------#

def print_simulation_info(emitter, bodies):
    """Display simulation configuration information."""
    # Emitter information
    print("\n---------- Emitter Configuration ----------")
    print(f"Position: {emitter['translation']}")
    print(f"Physical dimensions: width = {emitter['physical_width']}m, height = {emitter['physical_height']}m")
    print(f"Particles: width = {emitter['width']}, height = {emitter['height']}")
    print(f"Velocity: {emitter['velocity']} m/s")

    # Rigid bodies information
    print("\n---------- Rigid Bodies Configuration ----------")
    print(f"Number of rigid bodies: {len(bodies)}")
    
    body_names = ["Bottom Wall", "Top Wall", "Bottom-Left Obstacle", "Top-Left Obstacle"]
    for i, body in enumerate(bodies):
        print(f"{body_names[i]}:")
        print(f"  Position: {body['translation']}")
        print(f"  Rotation angle: {body['rotationAngle']} rad ({np.degrees(body['rotationAngle']):.1f}°)")
        print(f"  Scale: {body['scale']}")
        print("")

def print_simulation_info(bodies, Lx_calculated, nb_elem):
    """Display simulation configuration information."""
    # Rigid bodies information
    print("\n---------- Rigid Bodies Configuration ----------")
    print(f"Number of rigid bodies: {len(bodies)}")
    print(f"Calculated rectangle length (Lx): {Lx_calculated:.3f} m")
    
    for i, body in enumerate(bodies):
        if i < len(bodies)//2:
            wall_type = "Bottom Wall"
            idx = i
        else:
            wall_type = "Top Wall"
            idx = i - len(bodies)//2
            
        if idx < nb_elem:
            wall_section = "Narrowing"
        else:
            wall_section = "Straight"
            
        print(f"{wall_type} {idx} ({wall_section}):")
        print(f"  Position: {body['translation']}")
        print(f"  Rotation angle: {body['rotationAngle']} rad ({np.degrees(body['rotationAngle']):.1f}°)")
        print(f"  Scale: {body['scale']}")
        print("")
        
#---------------------------------#
#   Functions: channel curve 2D   #
#---------------------------------#


def calculate_lx_for_total_length(Lx_max, nb_elem, alpha_tot):
    
    # Decreasing angles from alpha_tot to 0
    angles = np.linspace(alpha_tot, 0, nb_elem)
    
    # Calculate sum of horizontal projections with Lx=1 (for normalization)
    total_projection = 0
    for angle in angles:
        total_projection += np.cos(angle)
    
    # Calculate Lx to achieve Lx_max
    Lx = Lx_max / total_projection
    print(f"Calculated rectangle length (Lx) for narrowing: {Lx}")
    
    return Lx


def create_narrowing(Lx, Ly, Lz, alpha_tot, nb_elem, x_start, y_init):
    
    # Local function to create a connected wall
    def create_connected_rectangles(Lx, Ly, Lz, alpha_tot, nb_elem, init_x, init_y, is_top_wall=False):
        rectangles = []
        
        # For top wall, invert the sign of angles
        sign = -1 if is_top_wall else 1
        
        # Calculate angles for each rectangle (from alpha_tot to 0)
        angles = np.linspace(sign * alpha_tot, 0, nb_elem)
        
        # Initial position
        center_x, center_y = init_x, init_y
        
        # First rectangle
        rectangles.append({
            "translation": [center_x, center_y, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": angles[0],
            "scale": [Lx, Ly, Lz]
        })
        
        # For each subsequent rectangle
        for i in range(1, len(angles)):
            prev_angle = angles[i-1]
            current_angle = angles[i]
            
            if not is_top_wall:
                # BOTTOM WALL - connect upper corners
                # Position of the upper right corner of the previous rectangle
                prev_corner_x = center_x + (Lx/2) * np.cos(prev_angle) - (Ly/2) * np.sin(prev_angle)
                prev_corner_y = center_y + (Lx/2) * np.sin(prev_angle) + (Ly/2) * np.cos(prev_angle)
                
                # Position of the upper left corner of the new rectangle relative to its center
                curr_corner_dx = -(Lx/2) * np.cos(current_angle) - (Ly/2) * np.sin(current_angle)
                curr_corner_dy = -(Lx/2) * np.sin(current_angle) + (Ly/2) * np.cos(current_angle)
            else:
                # TOP WALL - connect lower corners
                # Position of the lower right corner of the previous rectangle
                prev_corner_x = center_x + (Lx/2) * np.cos(prev_angle) + (Ly/2) * np.sin(prev_angle)
                prev_corner_y = center_y + (Lx/2) * np.sin(prev_angle) - (Ly/2) * np.cos(prev_angle)
                
                # Position of the lower left corner of the new rectangle relative to its center
                curr_corner_dx = -(Lx/2) * np.cos(current_angle) + (Ly/2) * np.sin(current_angle)
                curr_corner_dy = -(Lx/2) * np.sin(current_angle) - (Ly/2) * np.cos(current_angle)
            
            # Calculate the center of the current rectangle so corners touch
            center_x = prev_corner_x - curr_corner_dx
            center_y = prev_corner_y - curr_corner_dy
            
            rectangles.append({
                "translation": [center_x, center_y, 0],
                "rotationAxis": [0, 0, 1],
                "rotationAngle": current_angle,
                "scale": [Lx, Ly, Lz]
            })
        
        return rectangles
    
    # Generate rectangle configurations for bottom wall
    bottom_wall = create_connected_rectangles(Lx, Ly, Lz, alpha_tot, nb_elem, x_start, -y_init, is_top_wall=False)
    
    # Generate rectangle configurations for top wall
    top_wall = create_connected_rectangles(Lx, Ly, Lz, alpha_tot, nb_elem, x_start, y_init, is_top_wall=True)
    
    return bottom_wall, top_wall


def create_straight_channel(Lx_straight, Ly, Lz, narrowing_bottom, narrowing_top, num_segments=3):

    bottom_straight = []
    top_straight = []
    
    # Get the last rectangle of the narrowing
    last_bottom = narrowing_bottom[-1]
    last_top = narrowing_top[-1]
    
    # Position of the upper right corner of the last bottom rectangle
    last_bottom_x = last_bottom["translation"][0]
    last_bottom_y = last_bottom["translation"][1]
    bottom_right_x = last_bottom_x + (Lx_straight/2)
    bottom_right_y = last_bottom_y + (Ly/2)
    
    # Position of the lower right corner of the last top rectangle
    last_top_x = last_top["translation"][0]
    last_top_y = last_top["translation"][1]
    top_right_x = last_top_x + (Lx_straight/2)
    top_right_y = last_top_y - (Ly/2)
    
    # Create horizontal segments
    for i in range(num_segments):
        # Position of the bottom segment
        bottom_x = bottom_right_x + (i * Lx_straight)
        bottom_y = last_bottom_y
        
        # Position of the top segment
        top_x = top_right_x + (i * Lx_straight)
        top_y = last_top_y
        
        bottom_straight.append({
            "translation": [bottom_x, bottom_y, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0,  # Horizontal
            "scale": [Lx_straight, Ly, Lz]
        })
        
        top_straight.append({
            "translation": [top_x, top_y, 0],
            "rotationAxis": [0, 0, 1],
            "rotationAngle": 0,  # Horizontal
            "scale": [Lx_straight, Ly, Lz]
        })
    
    return bottom_straight, top_straight


def add_rectangles(common_params, bottom_narrowing, top_narrowing, bottom_straight, top_straight):
    
    rigid_bodies = []
    
    # Bottom narrowing walls 
    for rect in bottom_narrowing:
        rigid_bodies.append({
            **common_params,
            **rect,
            "color": [0.1, 0.4, 0.6, 1.0] 
        })
    
    # Top narrowing walls 
    for rect in top_narrowing:
        rigid_bodies.append({
            **common_params,
            **rect,
            "color": [0.1, 0.4, 0.6, 1.0]
        })
    
    # Bottom horizontal walls 
    for rect in bottom_straight:
        rigid_bodies.append({
            **common_params,
            **rect,
            "color": [0.1, 0.4, 0.6, 1.0]
        })
    
    # Top horizontal walls 
    for rect in top_straight:
        rigid_bodies.append({
            **common_params,
            **rect
        })
    
    return rigid_bodies

#---------------------------------#
#   Functions: channel horiz 2D   #
#---------------------------------#

def create_horiz_channel(Lx, Ly, Lz, diameter, x_start, y_start):

    rectangles = []

    x_center = x_start + (Lx/2)
    y_center_top = y_start + (diameter/2)
    y_center_bot = y_start - (diameter/2)

    # Top rectangle
    rectangles.append({
        "translation": [x_center, y_center_top, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [Lx, Ly, Lz]
    })

    # Bot rectangle
    rectangles.append({
        "translation": [x_center, y_center_bot, 0],
        "rotationAxis": [0, 0, 1],
        "rotationAngle": 0,
        "scale": [Lx, Ly, Lz]
    })

    return rectangles[0], rectangles[1]


#---------------------------------#
#    Functions: Free surface 0    #
#---------------------------------#

def calculate_scene_bounds(rigid_bodies, margin=0.1):
    
    # Initialize with extreme values
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    for body in rigid_bodies:
        trans = body["translation"]
        geometry_type = body["geometryFile"].split('/')[-1]
        
        if "UnitBox" in geometry_type:
            # Box extents
            half_scale = [s/2 for s in body["scale"]]
            body_min = [trans[0] - half_scale[0], trans[1] - half_scale[1], trans[2] - half_scale[2]]
            body_max = [trans[0] + half_scale[0], trans[1] + half_scale[1], trans[2] + half_scale[2]]
        
        elif "sphere" in geometry_type:
            # Sphere extents (assuming uniform scaling)
            radius = body["scale"][0]
            body_min = [trans[0] - radius, trans[1] - radius, trans[2] - radius]
            body_max = [trans[0] + radius, trans[1] + radius, trans[2] + radius]
        
        else:
            # Unknown geometry - skip
            continue
        
        # Update global min/max
        min_x = min(min_x, body_min[0])
        min_y = min(min_y, body_min[1])
        min_z = min(min_z, body_min[2])
        max_x = max(max_x, body_max[0])
        max_y = max(max_y, body_max[1])
        max_z = max(max_z, body_max[2])
    
    # Add margin
    min_coords = [min_x - margin, min_y - margin, min_z - margin]
    max_coords = [max_x + margin, max_y + margin, max_z + margin]
    
    return min_coords, max_coords
