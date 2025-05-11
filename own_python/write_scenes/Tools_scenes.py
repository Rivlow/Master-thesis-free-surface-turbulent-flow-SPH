import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

#---------------------------------#
#   General functional function   #
#---------------------------------#

def clean_files(directory):
	import shutil
	if os.path.exists(directory):
		shutil.rmtree(directory)
	os.makedirs(directory, exist_ok=True)
	
def calculate_emitter_particle_positions(emitter_pos, width, height, particle_radius):
	# Calculate particle diameter
	diam = 2.0 * particle_radius
	
	# Calculate starting offsets (like in the C++ code)
	startX = -0.5 * (width - 1) * diam
	startZ = -0.5 * (height - 1) * diam
	
	# Get rotation axes (simplified for default case)
	axisHeight = np.array([0, 1, 0])  # Y axis
	axisWidth = np.array([1, 0, 0])   # X axis
	
	# Calculate positions
	positions = []
	for j in range(height):
		for i in range(width):
			pos = emitter_pos + (i*diam + startX)*axisWidth + (j*diam + startZ)*axisHeight
			positions.append(pos)
	pos = positions[1:-1]
	y_pos = np.array([val[1] for val  in pos])
	
	return y_pos

def write_summary(summary_path, data, additional_params=None):
	"""
	Writes a file summarizing all simulation parameters using the existing data dictionary.
	
	Args:
		summary_path (str): Path where to save the summary file
		data (dict): The main data dictionary used for the simulation
		additional_params (dict, optional): Any additional parameters not in data
	"""
	# Create directory if needed
	os.makedirs(os.path.dirname(summary_path), exist_ok=True)
	
	with open(summary_path, 'w') as f:
		f.write("=== SIMULATION PARAMETERS SUMMARY ===\n\n")
		
		# First write the main simulation data
		for section, params in data.items():
			f.write(f"=== {section} ===\n")
			
			# Handle different types of sections
			if isinstance(params, list):
				# For list sections like Materials, RigidBodies, etc.
				for i, item in enumerate(params):
					f.write(f"--- Item {i+1} ---\n")
					for param_name, param_value in item.items():
						param_str = str(param_value).replace('\n', ' ')
						f.write(f"{param_name}: {param_str}\n")
					f.write("\n")
			else:
				# For dict sections like Configuration
				for param_name, param_value in params.items():
					param_str = str(param_value).replace('\n', ' ')
					f.write(f"{param_name}: {param_str}\n")
			f.write("\n")
		
		# Add any additional parameters not in the main data structure
		if additional_params:
			f.write("=== Additional Parameters ===\n")
			for section, params in additional_params.items():
				f.write(f"--- {section} ---\n")
				for param_name, param_value in params.items():
					param_str = str(param_value).replace('\n', ' ')
					f.write(f"{param_name}: {param_str}\n")
				f.write("\n")
		
		f.write("=== END OF SUMMARY ===\n")
	
	print(f"Parameters summary written to '{summary_path}'")

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
#    Functions: Free surface      #
#---------------------------------#

# Parabola function
def parabole(x):
    z = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < 8:
            z[i] = 0
        elif x[i] > 12:
            z[i] = 0
        else:
            z[i] = 0.2 - 0.05*(x[i]-10)**2
    return z

# Function to calculate scene bounds
def calculate_scene_bounds(rigid_bodies, margin=0.01):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    for body in rigid_bodies:
        translation = body.get("translation", [0, 0, 0])
        scale = body.get("scale", [1, 1, 1])
        
        # Calculate corners of the body's bounding box
        half_scale = [s/2 for s in scale]
        min_corner = [translation[i] - half_scale[i] for i in range(3)]
        max_corner = [translation[i] + half_scale[i] for i in range(3)]
        
        # Update overall bounds
        min_x = min(min_x, min_corner[0])
        min_y = min(min_y, min_corner[1])
        min_z = min(min_z, min_corner[2])
        max_x = max(max_x, max_corner[0])
        max_y = max(max_y, max_corner[1])
        max_z = max(max_z, max_corner[2])
    
    # Add margin
    min_x -= margin
    min_y -= margin
    min_z -= margin
    max_x += margin
    max_y += margin
    max_z += margin
    
    return [min_x, min_y, min_z], [max_x, max_y, max_z]

#-----------------------------------------#
#      Functions: Full init/ Bridge       #
#-----------------------------------------#

def isFluvial(fluv):
    if fluv:
        return 0.5
    else:
        return 1.5
    
def create_bridge(RigidBodies,
                  Lx_dom, Ly_dom, Lz_dom,
                  Lx_roof, Ly_roof, Lz_roof, trans_roof,
                  Lx_foot, Ly_foot, Lz_foot, trans_left_foot, trans_mid_foot, trans_right_foot):
      
	Bridge_foot_left = {
		"id": 2,
		"geometryFile": "../models/UnitBox.obj",
		"translation": trans_left_foot,
		"scale": [Lx_foot, Ly_foot, Lz_foot],
		"rotationAxis": [1, 0, 0],
		"rotationAngle": 0,
		"collisionObjectType": 2,
		"collisionObjectScale": [Lx_foot, Ly_foot, Lz_foot],
		"color": [0.1, 0.4, 0.6, 1.0],
		"isDynamic": False,
		"isWall": False,
		"mapInvert": False,
		"invertSDF": False,
		"mapThickness": 0.0,
		"mapResolution": [40, 40, 40],
		"samplingMode": 1,
		"friction": 0
	}

	Bridge_foot_mid = {
		"id": 3,
		"geometryFile": "../models/UnitBox.obj",
		"translation": trans_mid_foot,
		"scale": [Lx_foot, Ly_foot, Lz_foot],
		"rotationAxis": [1, 0, 0],
		"rotationAngle": 0,
		"collisionObjectType": 2,
		"collisionObjectScale": [Lx_foot, Ly_foot, Lz_foot],
		"color": [0.1, 0.4, 0.6, 1.0],
		"isDynamic": False,
		"isWall": False,
		"mapInvert": False,
		"invertSDF": False,
		"mapThickness": 0.0,
		"mapResolution": [40, 40, 40],
		"samplingMode": 1,
		"friction": 0
	}
	Bridge_foot_right = {
		"id": 4,
		"geometryFile": "../models/UnitBox.obj",
		"translation": trans_right_foot,
		"scale": [Lx_foot, Ly_foot, Lz_foot],
		"rotationAxis": [1, 0, 0],
		"rotationAngle": 0,
		"collisionObjectType": 2,
		"collisionObjectScale": [Lx_foot, Ly_foot, Lz_foot],
		"color": [0.1, 0.4, 0.6, 1.0],
		"isDynamic": False,
		"isWall": False,
		"mapInvert": False,
		"invertSDF": False,
		"mapThickness": 0.0,
		"mapResolution": [40, 40, 40],
		"samplingMode": 1,
		"friction": 0
	}
      
	Bridge_roof = {
		"id": 1,
		"geometryFile": "../models/UnitBox.obj",
		"translation": trans_roof,
		"scale": [Lx_roof, Ly_roof, Lz_roof],
		"rotationAxis": [1, 0, 0],
		"rotationAngle": 0,
		"collisionObjectType": 2,
		"collisionObjectScale": [Lx_roof, Ly_roof, Lz_roof],
		"color": [0.1, 0.4, 0.6, 1.0],
		"isDynamic": False,
		"isWall": False,
		"mapInvert": False,
		"invertSDF": False,
		"mapThickness": 0.0,
		"mapResolution": [60, 60, 60],
		"samplingMode": 1,
		"friction": 0
	}
     
	RigidBodies.append(Bridge_roof)
	RigidBodies.append(Bridge_foot_left)
	RigidBodies.append(Bridge_foot_mid)
	RigidBodies.append(Bridge_foot_right)
	
	return RigidBodies

def create_wood_distribution(RigidBodies, wood_distribution, next_id,
                             Lx_emit, Ly_emit, Lz_emit, trans_emit,
							 wood_density, restitution, friction,
							 placement_area_x_min, placement_area_x_max,
							 placement_area_z_min, placement_area_z_max,
							 placement_area_y,):
     
	
	placed_cylinders = []
	cylinders_placed = 0
	cylinders_skipped = 0
	

	for wood_type in wood_distribution:
		for i in range(wood_type["count"]):

			cylinder_length = wood_type["L"]
			cylinder_diameter = wood_type["D"]
			
			# Trouver une position valide pour ce cylindre
			result = generate_valid_position(
				placed_cylinders,
				cylinder_length,
				cylinder_diameter,
				[placement_area_x_min, placement_area_x_max],
				[placement_area_z_min, placement_area_z_max],
				placement_area_y,
				Lx_emit, Ly_emit, Lz_emit,
                trans_emit)
				
			
			# Si aucune position valide n'a été trouvée après max_tries
			if result is None:
				cylinders_skipped += 1
				continue
			
			position, rotation_axis, rotation_angle = result
			
			# Création de l'objet cylindre
			cylinder = {
				"id": next_id,
				"geometryFile": "../models/cylinder.obj",
				"translation": position,
				"scale": [cylinder_diameter/2, cylinder_length, cylinder_diameter/2],
				"rotationAxis": rotation_axis,
				"rotationAngle": rotation_angle,
				"collisionObjectType": 3,
				"collisionObjectScale": [cylinder_diameter/2, cylinder_length, cylinder_diameter/2],
				"color": [0.3, 0.5, 0.8, 1.0],
				"isDynamic": True,
				"density": wood_density,
				"velocity": [0, 0, 0],
				"restitution": restitution,
				"friction": friction,
				"mapInvert": False,
				"mapThickness": 0.0,
				"mapResolution": [60, 60, 60],
				"resolutionSDF": [60, 60, 60]
			}
			
			# Enregistrer ce cylindre pour les vérifications futures
			placed_cylinders.append({
				'pos': position,
				'dim': [cylinder_diameter/2, cylinder_length, cylinder_diameter/2],
				'axis': rotation_axis,
				'angle': rotation_angle
			})
			
			# Ajouter le cylindre à la liste des corps rigides
			RigidBodies.append(cylinder)
			next_id += 1
			cylinders_placed += 1

	print(f"Cylindres placés: {cylinders_placed}, Cylindres ignorés: {cylinders_skipped}")
	return RigidBodies

    
    

    
# Fonction pour vérifier si deux cylindres se chevauchent
def cylinders_overlap(pos1, dim1, axis1, angle1, pos2, dim2, axis2, angle2):
	"""
	Vérifie si deux cylindres se chevauchent en utilisant une approximation de boîte englobante.
	
	Args:
		pos1, pos2: positions [x, y, z] des centres des cylindres
		dim1, dim2: dimensions [rayon, longueur, rayon] des cylindres
		axis1, axis2: axes de rotation des cylindres
		angle1, angle2: angles de rotation des cylindres
	
	Returns:
		bool: True si les cylindres se chevauchent, False sinon
	"""
	# Calcul de la distance minimale entre les centres pour éviter le chevauchement
	min_dist_x = (dim1[1] + dim2[1]) / 2 * 1.1  # 10% de marge de sécurité en longueur
	min_dist_y = (dim1[0]*2 + dim2[0]*2) / 2 * 1.1  # 10% de marge en hauteur (diamètre)
	min_dist_z = (dim1[2]*2 + dim2[2]*2) / 2 * 1.1  # 10% de marge en largeur (diamètre)
	
	# Calcul de la distance réelle entre les centres
	dx = abs(pos1[0] - pos2[0])
	dy = abs(pos1[1] - pos2[1])
	dz = abs(pos1[2] - pos2[2])
	
	# Vérifier si les cylindres se chevauchent
	return dx < min_dist_x and dy < min_dist_y and dz < min_dist_z

# Fonction pour vérifier si un cylindre est en collision avec l'émetteur
def cylinder_collides_with_emitter(position, dimensions, trans_emit, Lx_emit,Ly_emit,Lz_emit):
	"""
	Vérifie si un cylindre est en collision avec l'émetteur.
	
	Args:
		position: position [x, y, z] du centre du cylindre
		dimensions: dimensions [rayon, longueur, rayon] du cylindre
	
	Returns:
		bool: True si le cylindre est en collision avec l'émetteur
	"""
	# Position et dimensions de l'émetteur
	emitter_position = trans_emit
	emitter_width = Lz_emit
	emitter_height = Ly_emit
	emitter_thickness = Lx_emit
	
	# Boîte englobante de l'émetteur (avec marge de sécurité)
	emitter_min_x = emitter_position[0] - emitter_thickness/2 - 0.3  # Marge de sécurité de 30cm
	emitter_max_x = emitter_position[0] + emitter_thickness/2 + 0.3
	emitter_min_y = emitter_position[1] - emitter_height/2 - 0.3
	emitter_max_y = emitter_position[1] + emitter_height/2 + 0.3
	emitter_min_z = emitter_position[2] - emitter_width/2 - 0.3
	emitter_max_z = emitter_position[2] + emitter_width/2 + 0.3
	
	# Dimensions effectives du cylindre (longueur dans l'axe X)
	cylinder_half_length = dimensions[1] / 2
	cylinder_radius = dimensions[0]
	
	# Vérifier la collision dans chaque dimension
	# En X: distance entre le centre du cylindre et l'émetteur < demi-longueur du cylindre
	x_collision = (position[0] - cylinder_half_length < emitter_max_x) and (position[0] + cylinder_half_length > emitter_min_x)
	# En Y: distance entre le centre du cylindre et l'émetteur < rayon du cylindre
	y_collision = (position[1] - cylinder_radius < emitter_max_y) and (position[1] + cylinder_radius > emitter_min_y)
	# En Z: distance entre le centre du cylindre et l'émetteur < rayon du cylindre
	z_collision = (position[2] - cylinder_radius < emitter_max_z) and (position[2] + cylinder_radius > emitter_min_z)
	
	return x_collision and y_collision and z_collision

# Fonction pour générer une position valide pour un nouveau cylindre
def generate_valid_position(placed_cylinders, cylinder_length, cylinder_diameter, x_range, z_range, y_pos, Lx_emit,Ly_emit,Lz_emit, trans_emit):
	"""
	Génère une position valide pour un nouveau cylindre qui ne chevauche aucun cylindre existant
	ni l'émetteur.
	
	Args:
		placed_cylinders: liste des cylindres déjà placés (position, dimension, rotation)
		cylinder_length, cylinder_diameter: dimensions du nouveau cylindre
		x_range, z_range: plages pour les coordonnées x et z
		y_pos: position de base en y
	
	Returns:
		tuple: (position, axis, angle) ou None si aucune position valide n'est trouvée après max_tries
	"""
	max_tries = 5000
	rotation_axis = [0, 0, 1]  # Tous les cylindres orientés dans le sens de l'écoulement
	rotation_angle = np.pi/2
	
	dimension = [cylinder_diameter/2, cylinder_length, cylinder_diameter/2]  # [rayon, longueur, rayon]
	
	for _ in range(max_tries):
		# Générer une position aléatoire
		pos_x = random.uniform(x_range[0], x_range[1])
		pos_z = random.uniform(z_range[0], z_range[1])
		pos_y = y_pos + random.uniform(-0.05, 0.05)  # Légère variation en hauteur
		
		position = [pos_x, pos_y, pos_z]
		
		# Vérifier collision avec l'émetteur
		if cylinder_collides_with_emitter(position, dimension, trans_emit, Lx_emit,Ly_emit,Lz_emit):
			continue
		
		# Vérifier s'il y a chevauchement avec des cylindres existants
		overlap = False
		for cyl in placed_cylinders:
			if cylinders_overlap(position, dimension, rotation_axis, rotation_angle, 
								cyl['pos'], cyl['dim'], cyl['axis'], cyl['angle']):
				overlap = True
				break
		
		# Si pas de chevauchement, retourner la position valide
		if not overlap:
			return position, rotation_axis, rotation_angle
	
	# Si aucune position valide n'est trouvée après max_tries
	return None