import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
def analyze_3d_flow_projection(vtk_data, slice_value, axis='y', thickness=0.1,
							bounds=None, plot=True, save=False, save_path=None,
							streamline_density=50, cmap='RdBu_r'):
	"""
	Projette les particules d'un volume 3D sur un plan 2D selon un axe spécifié,
	puis visualise la vorticité et les streamlines de vitesse.
	
	Args:
		vtk_data: Données VTK contenant les informations 3D des particules
		slice_value (float): Valeur sur l'axe choisi pour la coupe
		axis (str): Axe de projection ('y' ou 'z')
		thickness (float): Épaisseur de la tranche
		bounds (dict): Limites optionnelles pour les autres axes {x: (min, max), y: (min, max), z: (min, max)}
		plot (bool): Générer les visualisations
		save (bool): Sauvegarder les visualisations
		save_path (str): Chemin pour sauvegarder
		streamline_density (int): Densité des streamlines
		cmap (str): Colormap pour la vorticité
	"""
	
	points = vtk_data.points
	
	# Configuration des axes selon la projection choisie
	if axis == 'y':
		slice_axis = 1
		plot_axes = [0, 2]  # x et z pour le plan de visualisation
		plot_labels = ['x [m]', 'z [m]']
		vorticity_component_idx = 1  # Composante y de la vorticité
		velocity_components_idx = [0, 2]  # Composantes x et z de la vitesse
	elif axis == 'z':
		slice_axis = 2
		plot_axes = [0, 1]  # x et y pour le plan de visualisation
		plot_labels = ['x [m]', 'y [m]']
		vorticity_component_idx = 2  # Composante z de la vorticité
		velocity_components_idx = [0, 1]  # Composantes x et y de la vitesse
	else:
		raise ValueError("L'axe doit être 'y' ou 'z'")
	
	# Calcul des limites pour chaque axe
	axis_limits = {}
	for i, ax_name in enumerate(['x', 'y', 'z']):
		if bounds is not None and ax_name in bounds:
			axis_limits[ax_name] = bounds[ax_name]
		else:
			axis_limits[ax_name] = (np.min(points[:, i]), np.max(points[:, i]))
	
	# Création du masque pour la tranche
	slice_min = slice_value - thickness/2
	slice_max = slice_value + thickness/2
	
	# Initialisation du masque avec la condition sur l'axe de tranche
	inside_mask = (points[:, slice_axis] >= slice_min) & (points[:, slice_axis] <= slice_max)
	
	# Ajout des conditions sur les autres axes
	for i, ax_name in enumerate(['x', 'y', 'z']):
		if i != slice_axis:  # Ignorer l'axe de tranche
			ax_min, ax_max = axis_limits[ax_name]
			inside_mask &= (points[:, i] >= ax_min) & (points[:, i] <= ax_max)
	
	if not np.any(inside_mask):
		print(f"Aucune particule trouvée dans la tranche {axis}={slice_value} ± {thickness/2}")
		return None
	
	# Extraction des particules dans la tranche
	slice_points = points[inside_mask]
	plot_coords = [slice_points[:, ax] for ax in plot_axes]
	
	# Extraction des vitesses
	velocity = vtk_data['velocity'][inside_mask]
	plot_velocities = [velocity[:, ax] for ax in velocity_components_idx]
	velocity_mag = np.sqrt(plot_velocities[0]**2 + plot_velocities[1]**2)
	
	# Extraction de la vorticité (en supposant qu'elle est stockée dans 'angular_velocity')
	vorticity = vtk_data['angular_velocity'][inside_mask]
	vorticity_component = vorticity[:, vorticity_component_idx]

	print(np.shape(vorticity_component))
	print(vorticity_component)


	
	# Création des visualisations
	if plot:
		# 1. Heatmap de vorticité
		plt.figure(figsize=(10, 8))
		
		# Utilisation d'une colormap divergente pour la vorticité
		vmax = np.max(np.abs(vorticity_component))
		norm = colors.Normalize(-vmax, vmax)
		
		scatter = plt.scatter(plot_coords[0], plot_coords[1], 
							c=vorticity_component, cmap=cmap,
							norm=norm, s=30, alpha=0.8)
		
		ax_name = 'Y' if axis == 'y' else 'Z'
		plt.colorbar(scatter, label=f'Composante {ax_name} de la Vorticité [1/s]')
		plt.title(f'Vorticité (composante {ax_name}) à {axis}={slice_value} m')
		plt.xlabel(plot_labels[0])
		plt.ylabel(plot_labels[1])
		plt.axis('equal')
		plt.tight_layout()

		plt.show()
		
		if save and save_path is not None:
			plt.savefig(f'{save_path}_vorticity_{axis}{slice_value}.pdf', dpi=300)
			
		# 2. Visualisation des streamlines
		plt.figure(figsize=(10, 8))
		
		# Méthode de visualisation par segments de ligne (plus efficace pour les données de particules)
		segments = []
		colors_list = []
		
		# Facteur de longueur proportionnel à la vitesse
		length_factor = 0.05 * np.mean(velocity_mag) if np.mean(velocity_mag) > 0 else 0.02
		
		# Création des segments de ligne pour représenter les vecteurs de vitesse
		for i in range(len(plot_coords[0])):
			x, y = plot_coords[0][i], plot_coords[1][i]
			dx, dy = plot_velocities[0][i], plot_velocities[1][i]
			
			# Normalisation de la direction tout en conservant l'influence de la magnitude
			magnitude = np.sqrt(dx**2 + dy**2)
			if magnitude > 0:
				dx_norm = dx / magnitude
				dy_norm = dy / magnitude
				
				line_length = magnitude * length_factor
				
				# Création d'un segment de ligne
				segments.append([(x - dx_norm * line_length, y - dy_norm * line_length),
							(x + dx_norm * line_length, y + dy_norm * line_length)])
				
				# Coloration par vorticité
				colors_list.append(vorticity_component[i])
		
		if segments:
			# Création d'une LineCollection
			lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, alpha=0.8)
			lc.set_array(np.array(colors_list))
			
			plt.gca().add_collection(lc)
			plt.colorbar(lc, label=f'Composante {ax_name} de la Vorticité [1/s]')
			
			# Ajout des positions des particules comme points
			plt.scatter(plot_coords[0], plot_coords[1], s=5, color='gray', alpha=0.3)
			
			plt.title(f'Motifs d\'écoulement et Vorticité à {axis}={slice_value} m')
			plt.xlabel(plot_labels[0])
			plt.ylabel(plot_labels[1])
			plt.axis('equal')
			plt.tight_layout()
			
			if save and save_path is not None:
				plt.savefig(f'{save_path}_flow_patterns_{axis}{slice_value}.pdf', dpi=300)
		
		if not save:
			plt.show()

def visualize_3d_particles_pyvista(vtk_data, color_by='velocity', component=None, 
								clip_plane=None, scalar_range=None, cmap='rainbow',
								point_size=5.0, background_color='white', 
								show=True, save_path=None, screenshot_size=(1920, 1080),
								add_axes=True, add_legend=True):
	"""
	Visualise des particules 3D à l'aide de PyVista avec coloration par vitesse ou vorticité.
	
	Args:
		vtk_data: Données VTK contenant les positions et attributs des particules
		color_by (str): 'velocity' ou 'vorticity' pour colorier les particules
		component (int or str): Composante à utiliser (0, 1, 2 pour x, y, z ou 'magnitude')
							Si None, utilise la magnitude
		clip_plane (dict): Dict pour découper le domaine {
			'normal': (nx, ny, nz),  # Normal du plan
			'origin': (ox, oy, oz)   # Point sur le plan
		}
		scalar_range (tuple): Limites min/max pour l'échelle de couleur (min, max)
		cmap (str): Nom de la colormap à utiliser
		point_size (float): Taille des points à afficher
		background_color (str): Couleur d'arrière-plan
		show (bool): Afficher la visualisation interactive
		save_path (str): Chemin pour sauvegarder le screenshot
		screenshot_size (tuple): Résolution du screenshot (width, height)
		add_axes (bool): Ajouter un widget d'axes
		add_legend (bool): Ajouter une légende pour l'échelle de couleur
		
	Returns:
		plotter: Objet plotter PyVista pour d'éventuelles modifications supplémentaires
	"""
	import numpy as np
	import pyvista as pv
	
	# Configuration de base de PyVista
	pv.set_plot_theme('document')  # Theme clair pour publication
	
	# Création d'un point cloud à partir des positions des particules
	points = vtk_data.points
	cloud = pv.PolyData(points)
	
	# Préparation des scalaires pour la coloration
	if color_by.lower() == 'velocity':
		vectors = vtk_data['velocity']
		if component is None or component == 'magnitude':
			# Magnitude de la vitesse
			scalars = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2 + vectors[:, 2]**2)
			scalar_name = 'Vitesse (magnitude) [m/s]'
		elif isinstance(component, int) and 0 <= component <= 2:
			# Composante spécifique
			scalars = vectors[:, component]
			comp_name = ['x', 'y', 'z'][component]
			scalar_name = f'Vitesse {comp_name} [m/s]'
		else:
			raise ValueError("component doit être 0, 1, 2 (pour x, y, z) ou 'magnitude'")
	
	elif color_by.lower() == 'vorticity':
		# Supposons que la vorticité est stockée comme 'angular_velocity'
		# Adaptez la clé si nécessaire pour votre structure de données
		vectors = vtk_data['angular_velocity']
		if component is None or component == 'magnitude':
			# Magnitude de la vorticité
			scalars = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2 + vectors[:, 2]**2)
			scalar_name = 'Vorticité (magnitude) [1/s]'
		elif isinstance(component, int) and 0 <= component <= 2:
			# Composante spécifique
			scalars = vectors[:, component]
			comp_name = ['x', 'y', 'z'][component]
			scalar_name = f'Vorticité {comp_name} [1/s]'
		else:
			raise ValueError("component doit être 0, 1, 2 (pour x, y, z) ou 'magnitude'")
	
	else:
		raise ValueError("color_by doit être 'velocity' ou 'vorticity'")
	
	# Ajouter les scalaires au point cloud
	cloud.point_data[scalar_name] = scalars
	
	# Ajouter les vecteurs de vitesse complets pour éventuellement afficher des glyphes
	cloud.point_data["Velocity"] = vtk_data['velocity']
	
	# Créer un plotter
	plotter = pv.Plotter(window_size=screenshot_size, off_screen=not show)
	plotter.background_color = background_color
	
	# Ajouter les axes
	if add_axes:
		plotter.add_axes()
		plotter.add_bounding_box()
	
	# Appliquer un plan de coupe si spécifié
	if clip_plane is not None:
		clipped_cloud = cloud.clip(normal=clip_plane['normal'], origin=clip_plane['origin'])
		plotter.add_mesh(clipped_cloud, point_size=point_size, render_points_as_spheres=True,
						scalars=scalar_name, cmap=cmap, clim=scalar_range,
						show_scalar_bar=add_legend)
	else:
		plotter.add_mesh(cloud, point_size=point_size, render_points_as_spheres=True,
						scalars=scalar_name, cmap=cmap, clim=scalar_range,
						show_scalar_bar=add_legend)
	
	# Configurer la barre de couleur
	if add_legend:
		sbar = plotter.scalar_bar
		sbar.title = scalar_name
		sbar.title_font_size = 14
		sbar.label_font_size = 12
		sbar.position_x = 0.8
		sbar.position_y = 0.1
	
	# Sauvegarder la capture d'écran si demandé
	if save_path is not None:
		plotter.screenshot(save_path, window_size=screenshot_size, 
						return_img=False, transparent_background=False)
	
	# Montrer la visualisation interactive
	if show:
		plotter.show()
	
	return plotter
	