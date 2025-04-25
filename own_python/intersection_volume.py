import numpy as np
import trimesh
import math
import matplotlib.pyplot as plt


def create_cylinder(radius, height, transform=None):
	"""Crée un cylindre avec un rayon et une hauteur spécifiés"""
	cylinder = trimesh.creation.cylinder(radius=radius, height=height)
	if transform is not None:
		cylinder.apply_transform(transform)
	return cylinder

def calculate_cylinder_intersection_volume(radius1, height1, radius2, height2, t_junction=False, visualize=True):

	cylinder1 = create_cylinder(radius=radius1, height=height1)
	cylinder2 = create_cylinder(radius=radius2, height=height2)

	# Rotation de 90 degrés autour de l'axe y pour aligner avec l'axe x
	rotation = trimesh.transformations.rotation_matrix(
		angle=np.pi/2, 
		direction=[0, 1, 0],
		point=[0, 0, 0]
	)

	if t_junction:
		# Pour une jonction en T, nous avons besoin que le cylindre 2 s'arrête exactement 
		# à l'axe central du cylindre 1
		# Calculer le déplacement nécessaire
		displacement = radius1  # Distance du centre du cylindre 1 jusqu'à sa surface
		
		# Combiner rotation et translation
		translation = trimesh.transformations.translation_matrix([0, 0, 0])
		transform = trimesh.transformations.concatenate_matrices(rotation, translation)
	else:
		# Utiliser uniquement la rotation (cylindres se croisant en leur centre)
		transform = rotation

	# Appliquer la transformation
	cylinder2.apply_transform(transform)

	# Calculer l'intersection des deux cylindres
	intersection = trimesh.boolean.intersection([cylinder1, cylinder2])

	# Volume d'intersection
	volume = 0.0
	if intersection is not None:
		volume = intersection.volume

	# Visualiser si demandé
	if visualize:
		visualize_intersection(cylinder1, cylinder2, intersection)

	return volume, cylinder1, cylinder2, intersection

def visualize_intersection(cylinder1, cylinder2, intersection):
	"""Visualise les cylindres et leur intersection"""
	# Créer une scène pour la visualisation
	scene = trimesh.Scene()

	# Ajouter les cylindres à la scène avec des couleurs semi-transparentes
	cylinder1.visual.face_colors = [255, 0, 0, 100]  # Rouge transparent
	cylinder2.visual.face_colors = [0, 0, 255, 100]  # Bleu transparent

	scene.add_geometry(cylinder1)
	scene.add_geometry(cylinder2)

	# Ajouter l'intersection si elle existe
	if intersection is not None:
		intersection.visual.face_colors = [0, 255, 0, 255]  # Vert opaque
		scene.add_geometry(intersection)

	# Afficher la scène
	scene.show()

# Exemple d'utilisation
if __name__ == "__main__":
	cm = 1e-2
	radius1 = 5*cm  # rayon du premier cylindre
	height1 = 25*cm  # hauteur du premier cylindre
	radius2 = 5*cm # rayon du second cylindre
	height2 = 15*cm  # hauteur du second cylindre

	# Calculer et visualiser
	volume, cylinder1, cylinder2, intersection = calculate_cylinder_intersection_volume(
	radius1, height1, radius2, height2, t_junction=True)
	print(f"Volume d'intersection: {volume/2}")

	# Comparaison avec la formule analytique pour vérification
	analytical_volume = 8 * radius1**2 * radius2
	print(f"Volume analytique: {analytical_volume}")

	plt.savefig('intersectionvolume.PDF')