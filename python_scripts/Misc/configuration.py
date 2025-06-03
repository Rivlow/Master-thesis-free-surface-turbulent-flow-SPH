import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 26, 26
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=MEDIUM_SIZE)    
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=MEDIUM_SIZE)  
plt.rc('ytick', labelsize=MEDIUM_SIZE)   
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE) 

def configure_latex():
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')


def free_surface_config():

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

	configure_latex()

	# Configuration du graphique
	fig, ax = plt.subplots(1, 1, figsize=(12, 6))



	# Définition du domaine
	x = np.linspace(7, 15, 1000)
	z_obstacle = parabole(x)

	# Fond du canal (plus épais)
	ax.plot([7, 15], [0, 0], 'k-', linewidth=4)

	# Obstacle parabolique (ligne épaisse seulement)
	x_obstacle = x[(x >= 8) & (x <= 12)]
	z_obstacle_plot = parabole(x_obstacle)
	ax.plot(x_obstacle, z_obstacle_plot, color='black', linewidth=5, label="Topography \& channel bed")
	ax.fill_between(x_obstacle, z_obstacle_plot, color='grey', alpha=0.5)

	# Zone de condition limite amont (rectangle avec débit Q)
	x_inlet = 7.5
	width_inlet = 0.4
	height_inlet = 0.25

	# Rectangle pour la condition limite d'entrée
	rect_inlet = plt.Rectangle((x_inlet-width_inlet/2, 0), width_inlet, height_inlet, 
							fill=True, facecolor='lightgreen', alpha=0.3, edgecolor='green', linewidth=2)
	ax.add_patch(rect_inlet)

	# Petites flèches dans la zone amont
	n_arrows_inlet = 4
	for i in range(n_arrows_inlet):
		y_arrow = 0.05 + i * 0.05
		ax.annotate('', xy=(x_inlet + 0.1, y_arrow), xytext=(x_inlet - 0.1, y_arrow),
					arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))

	# Zone de condition limite aval (rectangle avec flèches)
	x_outlet = 14.5
	width_outlet = 0.4
	height_outlet = 0.25

	# Rectangle pour la condition limite aval
	rect_outlet = plt.Rectangle((x_outlet-width_outlet/2, 0), width_outlet, height_outlet, 
							fill=True, facecolor='lightcoral', alpha=0.3, edgecolor='red', linewidth=2)
	ax.add_patch(rect_outlet)

	# Petites flèches dans la zone aval
	n_arrows = 4
	for i in range(n_arrows):
		y_arrow = 0.05 + i * 0.05
		ax.annotate('', xy=(x_outlet + 0.1, y_arrow), xytext=(x_outlet - 0.1, y_arrow),
					arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

	# Label pour la zone aval
	ax.text(13.8, 0.15, '$U_{outlet}$', color='red', fontweight='bold')
	ax.text(7.1, 0.15, '$U_{0}$', color='green', fontweight='bold')


	# Annotation de l'équation de l'obstacle
	ax.text(10, 0.25, r'$Z(x) = 0.2 - 0.05(x-10)^2$', 
			bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
			ha='center')

	# Annotation de la longueur
	ax.annotate('', xy=(7, -0.05), xytext=(15, -0.05),
				arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
	ax.text(11, -0.08, 'L = 8 [m]', ha='center', 
			bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))


	# Configuration des axes et labels
	ax.set_xlim(7, 15)
	ax.set_ylim(-0.1, 0.4)
	ax.set_xlabel(r'Distance x [m]')
	ax.set_xticks(np.arange(7, 15+1, 1))
	ax.set_ylabel(r'Height y [m]')
	ax.set_yticks(np.arange(-0.1, 0.4+0.1, 0.1))
	ax.grid(True, alpha=0.4, ls='--')
	ax.set_aspect('auto')

	plt.legend()

	plt.tight_layout()


	plt.savefig('Pictures/CH6_valid_test/free_surface/free_surf_config.pdf', bbox_inches='tight', dpi=30)
	plt.show()

def turbulent_config():

	def compute_u_ghe(y_line, U_0, delta, gamma):
		"""
		Compute velocity profile using the Generalized Hybrid Equation (GHE) model.
		
		Args:
			y_line (array): Array of y-positions (normalized between -1 and 1)
			U_0 (float): Maximum velocity
			delta (float): Boundary layer thickness parameter
			gamma (float): Weighting parameter between laminar and turbulent profiles
		
		Returns:
			tuple: (Full velocity profile, Laminar profile, Turbulent profile)
		"""
		# Determine size for half profile
		if len(y_line) % 2 == 0:
			size_half = len(y_line) // 2
		else:
			size_half = (len(y_line) + 1) // 2
		
		# Create normalized y coordinates for half profile (0 to 0.5)
		y_half = np.linspace(0, 0.5, size_half)
		
		# Compute velocity profiles
		U_L = U_0 * 4 * y_half * (1 - y_half)  # Laminar profile
		U_T = U_0 * (1 - np.exp(1 - np.exp(y_half / delta)))  # Turbulent profile
		U_GHE_half = gamma * U_T + (1 - gamma) * U_L  # Hybrid model
		
		# Create full symmetric profile
		if len(y_line) % 2 == 0:
			U_full = np.concatenate((U_GHE_half, U_GHE_half[::-1]))
			U_L_full = np.concatenate((U_L, U_L[::-1]))
			U_T_full = np.concatenate((U_T, U_T[::-1]))
		else:
			U_full = np.concatenate((U_GHE_half, U_GHE_half[1:][::-1]))
			U_L_full = np.concatenate((U_L, U_L[1:][::-1]))
			U_T_full = np.concatenate((U_T, U_T[1:][::-1]))
		
		return U_full, U_L_full, U_T_full

	
	# Channel parameters
	L = 50  # Channel length [m]
	D = 3.94  # Channel height [m]
	
	# Velocity profile parameters
	U_max = 4.0  # Maximum velocity
	delta = 0.06  # Boundary layer parameter
	gamma = 0.6  # Weighting parameter
	
	# Create figure and axis
	configure_latex()
	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	
	
	
	# === DRAW CHANNEL WALLS ===
	# Horizontal walls (top and bottom) - centered around y=0
	ax.plot([0, L], [-D/2, -D/2], 'k-', linewidth=4, label='Channel walls')
	ax.plot([0, L], [D/2, D/2], 'k-', linewidth=4)
	
	# Vertical walls (inlet and outlet) - commented out as in original
	#ax.plot([0, 0], [-D/2, D/2], 'k-', linewidth=4)
	#ax.plot([L, L], [-D/2, D/2], 'k-', linewidth=4)
	
	# === INLET ZONE ===
	x_inlet = 0
	width_inlet = 2.5
	height_inlet = D
	
	# Rectangle for inlet boundary condition - centered around y=0
	rect_inlet = plt.Rectangle((x_inlet, -D/2), width_inlet, height_inlet, 
							fill=True, facecolor='lightgreen', alpha=0.3, 
							edgecolor='green', linewidth=2)
	ax.add_patch(rect_inlet)
	
	# Arrows in the inlet zone - centered around y=0
	n_arrows_inlet = 8
	for i in range(n_arrows_inlet):
		y_arrow = -D/2 + D * (i + 1) / (n_arrows_inlet + 1)
		ax.annotate('', xy=(x_inlet + 2, y_arrow), xytext=(x_inlet + 0.4, y_arrow),
				arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
	
	# Label for inlet zone - centered at y=0
	ax.text(3.8, 0, r'$U_{inlet}$', color='green', 
			fontweight='bold', ha='center', va='center')
	
	# === GHE VELOCITY PROFILE ===
	# Create y profile and normalize - centered around 0
	y_profile = np.linspace(-D/2, D/2, 100)
	y_normalized = y_profile / (D/2)  # Normalize between -1 and 1
	
	# Compute GHE velocity profile
	U_ghe, U_L, U_T = compute_u_ghe(y_normalized, U_max, delta, gamma)
	
	# Position of velocity profile (oriented to the right)
	x_profile_base = L - 25
	x_profile = x_profile_base + U_ghe
	
	# Draw velocity profile
	ax.plot(x_profile, y_profile, 'r-', linewidth=3)
	
	# Label u(y) in a box - centered at y=0
	ax.text(L-23, 0, r'$u(y)$', 
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
			ha='center')
	
	# === DIMENSION ANNOTATIONS ===
	# Length annotation
	ax.annotate('', xy=(0, -D/2-0.3), xytext=(L, -D/2-0.3),
			arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
	ax.text(L/2, -D/2-0.6, f'L = {L} [m]', ha='center',
			bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
	
	# Height annotation - centered around y=0
	ax.annotate('', xy=(L+1, -D/2), xytext=(L+1, D/2),
			arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
	ax.text(L-3, 0, f'D = {D} [m]', ha='center', va='center',
			bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
	
	# === AXIS CONFIGURATION ===
	# Light grid
	ax.grid(True, alpha=0.3, ls="--")
	
	# Configure axes and labels
	ax.set_xlim(-2, L+5)
	ax.set_xticks(np.arange(-5, 55+10, 10))

	ax.set_ylim(-2, 2)
	ax.set_yticks(np.arange(-3, 3+1, 1))

	ax.set_xlabel(r'Distance x [m]')
	ax.set_ylabel(r'Diameter y [m]')
	ax.set_aspect('auto')	
	
	# Legend
	ax.legend(loc='upper left')
	
	plt.tight_layout()
	
	plt.savefig('Pictures/CH6_valid_test/turbulent/turbulent_config.pdf', bbox_inches='tight', dpi=300)
	
	plt.show()

def bridge_config():

		# Dimensions du pont (en mm)
	WIDTH_TOTAL = 983  # Largeur totale
	HEIGHT_TOTAL = 211  # Hauteur totale
	DEPTH = 500  # Profondeur
	DECK_THICKNESS = 64  # Épaisseur du tablier
	PIER_WIDTH_2D = 64  # Largeur des piliers
	PIER_SPACING_2D = 461  # Espacement entre piliers

	deck_height = HEIGHT_TOTAL
	total_height = HEIGHT_TOTAL + DECK_THICKNESS  # 275 mm

	# Fonction pour dessiner les lignes de cotation avec flèches
	def draw_dimension_line(ax, x1, y1, x2, y2, text, offset=20, text_offset=10):
		ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
		ax.plot([x1, x1], [y1-10, y1+10], 'k-', linewidth=1)
		ax.plot([x2, x2], [y1-10, y1+10], 'k-', linewidth=1)
		
		arrow_size = 8
		ax.annotate('', xy=(x1, y1), xytext=(x1+arrow_size, y1),
					arrowprops=dict(arrowstyle='<-', color='black', lw=1))
		ax.annotate('', xy=(x2, y1), xytext=(x2-arrow_size, y1),
					arrowprops=dict(arrowstyle='<-', color='black', lw=1))
		
		mid_x = (x1 + x2) / 2
		ax.text(mid_x, y1 + text_offset, text, ha='center', va='bottom', fontsize=10)

	def draw_vertical_dimension(ax, x, y1, y2, text, offset=30):
		ax.plot([x + offset, x + offset], [y1, y2], 'k-', linewidth=1)
		ax.plot([x, x + offset + 10], [y1, y1], 'k-', linewidth=1)
		ax.plot([x, x + offset + 10], [y2, y2], 'k-', linewidth=1)
		
		arrow_size = 8
		ax.annotate('', xy=(x + offset, y1), xytext=(x + offset, y1+arrow_size),
					arrowprops=dict(arrowstyle='<-', color='black', lw=1))
		ax.annotate('', xy=(x + offset, y2), xytext=(x + offset, y2-arrow_size),
					arrowprops=dict(arrowstyle='<-', color='black', lw=1))
		
		mid_y = (y1 + y2) / 2
		ax.text(x + offset + 15, mid_y, text, ha='left', va='center', fontsize=10, rotation=90)

	# =====================================================
	# FIGURE UNIQUE CORRIGÉE AVEC ALIGNEMENT VERTICAL
	# =====================================================
	fig = plt.figure(figsize=(12, 6))

	# Calcul des marges pour les cotations
	margin_bottom = 120
	margin_top = 80
	margin_sides = 100

	# Hauteur totale disponible pour le dessin (en pixels de figure)
	available_height = 0.7  # 70% de la figure pour les dessins

	# Pour que les deux vues aient la même hauteur visuelle du pont
	# on calcule le ratio pixels/mm pour que total_height occupe la même hauteur dans les deux vues
	target_height_in_figure = available_height  # hauteur cible pour le pont dans la figure

	# Vue de côté
	ax1_width = 0.35
	ax1_height = available_height
	ax1 = fig.add_axes([0.01, 0.15, ax1_width, ax1_height])

	# Calcul de l'échelle pour la vue de côté
	side_view_width_with_margins = DEPTH + 2 * margin_sides
	side_view_height_with_margins = total_height + margin_bottom + margin_top
	side_scale = min(ax1_width / (side_view_width_with_margins / 1000), 
					ax1_height / (side_view_height_with_margins / 1000))

	# Dessin des éléments
	support_section = patches.Rectangle((0, 0), DEPTH, deck_height, linewidth=2, edgecolor='black', facecolor='lightgray')
	ax1.add_patch(support_section)
	beam_section = patches.Rectangle((0, deck_height), DEPTH, DECK_THICKNESS, linewidth=2, edgecolor='black', facecolor='gray')
	ax1.add_patch(beam_section)

	# Cotations
	draw_dimension_line(ax1, 0, -40, DEPTH, -40, '500 mm', text_offset=15)
	draw_vertical_dimension(ax1, DEPTH + 20, deck_height, total_height, '64 mm')
	draw_vertical_dimension(ax1, DEPTH + 60, 0, total_height, '275 mm')

	# Labels
	ax1.set_xlim(-margin_sides, DEPTH + margin_sides)
	ax1.set_ylim(-margin_bottom, total_height + margin_top)
	ax1.set_aspect('equal')
	ax1.axis('off')

	# Vue de face - avec même hauteur visuelle
	# On calcule la largeur nécessaire pour maintenir l'aspect ratio correct
	front_view_width_with_margins = WIDTH_TOTAL + 2 * margin_sides
	front_view_height_with_margins = total_height + margin_bottom + margin_top

	# Pour avoir la même hauteur visuelle, on ajuste la largeur de ax2
	ax2_width = ax1_width * (front_view_width_with_margins / side_view_width_with_margins)
	ax2_height = ax1_height
	ax2 = fig.add_axes([0.35, 0.15, ax2_width, ax2_height])

	# Dessin des éléments
	beam_y = deck_height
	beam = patches.Rectangle((0, beam_y), WIDTH_TOTAL, DECK_THICKNESS, linewidth=2, edgecolor='black', facecolor='gray')
	ax2.add_patch(beam)

	# Supports
	support_left_x = (WIDTH_TOTAL - PIER_SPACING_2D) / 2 - PIER_WIDTH_2D / 2
	support_left = patches.Rectangle((support_left_x, 0), PIER_WIDTH_2D, deck_height, linewidth=2, edgecolor='black', facecolor='lightgray')
	ax2.add_patch(support_left)

	support_center_x = (WIDTH_TOTAL - PIER_WIDTH_2D) / 2
	support_center = patches.Rectangle((support_center_x, 0), PIER_WIDTH_2D, deck_height, linewidth=2, edgecolor='black', facecolor='lightgray')
	ax2.add_patch(support_center)

	support_right_x = (WIDTH_TOTAL + PIER_SPACING_2D) / 2 - PIER_WIDTH_2D / 2
	support_right = patches.Rectangle((support_right_x, 0), PIER_WIDTH_2D, deck_height, linewidth=2, edgecolor='black', facecolor='lightgray')
	ax2.add_patch(support_right)

	# Cotations
	draw_dimension_line(ax2, 0, total_height + 40, WIDTH_TOTAL, total_height + 40, '983 mm', text_offset=15)
	center_x = WIDTH_TOTAL / 2
	left_support_center = support_left_x + PIER_WIDTH_2D / 2
	right_support_center = support_right_x + PIER_WIDTH_2D / 2
	draw_dimension_line(ax2, left_support_center, -40, center_x, -40, '461 mm', text_offset=15)
	draw_dimension_line(ax2, center_x, -40, right_support_center, -40, '461 mm', text_offset=15)
	draw_vertical_dimension(ax2, WIDTH_TOTAL + 20, beam_y, total_height, '64 mm')
	draw_vertical_dimension(ax2, WIDTH_TOTAL + 60, 0, total_height, '275 mm')

	# Labels
	ax2.set_xlim(-margin_sides, WIDTH_TOTAL + margin_sides)
	ax2.set_ylim(-margin_bottom, total_height + margin_top)
	ax2.set_aspect('equal')
	ax2.axis('off')

	plt.savefig('Pictures/CH8_final_simulation/bridge_combined_views.pdf', dpi=300, bbox_inches='tight')
	plt.show()

	# ===================================
	# FIGURE 3: TOP VIEW (VUE AÉRIENNE) 
	# ===================================

	# Dimensions du canal (converties en mm)
	CANAL_LENGTH_BEFORE = 1500  # 1.5 m avant le pont
	CANAL_LENGTH_AFTER = 1000    # 0.5 m après le pont
	TOTAL_LENGTH = CANAL_LENGTH_BEFORE + DEPTH + CANAL_LENGTH_AFTER

	plt.figure(figsize=(12, 6))
	ax3 = plt.gca()

	# Dessin du canal
	canal = patches.Rectangle((0, 0), TOTAL_LENGTH, WIDTH_TOTAL,
							linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
	ax3.add_patch(canal)

	# Dessin du pont (zone dans le canal)
	bridge_start = CANAL_LENGTH_BEFORE
	bridge_rect = patches.Rectangle((bridge_start, 0), DEPTH, WIDTH_TOTAL,
								linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
	ax3.add_patch(bridge_rect)

	# Flèche d'écoulement
	ax3.arrow(100, WIDTH_TOTAL/2, TOTAL_LENGTH - 200, 0, 
			head_width=50, head_length=60, fc='red', ec='red', width=10)

	# Ajustement des limites avec marges proportionnelles
	x_margin = TOTAL_LENGTH * 0.1
	y_margin = WIDTH_TOTAL * 0.3
	ax3.set_xlim(0 - x_margin, TOTAL_LENGTH + x_margin)
	ax3.set_ylim(-y_margin, WIDTH_TOTAL + y_margin)

	# Forcer l'échelle égale
	ax3.set_aspect('equal', adjustable='datalim')

	# Cotations HORIZONTALES avec décalage vertical
	draw_dimension_line(ax3, 0, -40, CANAL_LENGTH_BEFORE, -40, '1500 mm', text_offset=-45)
	draw_dimension_line(ax3, CANAL_LENGTH_BEFORE, -80, bridge_start + DEPTH, -80, '500 mm', text_offset=-45)
	draw_dimension_line(ax3, bridge_start + DEPTH, -40, TOTAL_LENGTH, -40, '1000 mm', text_offset=-45)

	# Cotation VERTICALE (largeur du canal)
	draw_vertical_dimension(ax3, TOTAL_LENGTH + 40, 0, WIDTH_TOTAL, '983 mm', offset=20)

	# Texte et labels
	ax3.text(bridge_start + DEPTH/2, WIDTH_TOTAL/2, 'BRIDGE', 
			ha='center', va='center', fontsize=16, fontweight='bold')
	ax3.text(100, WIDTH_TOTAL/2 + 60, 'Inflow', 
			ha='left', va='center', fontsize=16, color='red')
	ax3.text(TOTAL_LENGTH - 100, WIDTH_TOTAL/2 + 60, 'Outflow', 
			ha='right', va='center', fontsize=16, color='red')

	# Position des piliers PERPENDICULAIRES à l'écoulement
	pier_length = 64  # Longueur dans le sens de l'écoulement
	pier_width = 64   # Largeur perpendiculaire

	# Positions en Y (largeur) - mêmes que dans la vue de face
	y_positions = [
		(WIDTH_TOTAL - PIER_SPACING_2D) / 2 - PIER_WIDTH_2D / 2 + PIER_WIDTH_2D/2,
		WIDTH_TOTAL / 2,
		(WIDTH_TOTAL + PIER_SPACING_2D) / 2 - PIER_WIDTH_2D / 2 + PIER_WIDTH_2D/2
	]

	# Ajouter les piliers CORRIGÉS (perpendiculaires à l'écoulement)
	for y in y_positions:
		# Position X centrée dans la section du pont
		x_center = bridge_start + DEPTH/2 - pier_length/2
		pier = patches.Rectangle((x_center, y - pier_width/2), 
								pier_length, pier_width,
							linewidth=1, edgecolor='black', facecolor='darkgray', alpha=0.9)
		ax3.add_patch(pier)
		
		# Ajouter indication de position
		ax3.text(x_center + pier_length/2, y + 60, f'Pier', 
				ha='center', va='center', fontsize=12)

	ax3.set_xlim(-100, TOTAL_LENGTH + 100)
	ax3.set_ylim(-150, WIDTH_TOTAL + 100)
	ax3.set_aspect('equal')
	ax3.axis('off')
	plt.tight_layout()
	plt.savefig('Pictures/CH8_final_simulation/bridge_top_view.pdf', dpi=300, bbox_inches='tight')
	plt.show()


def main():


	
	turbulent_config()
	#free_surface_config()
	#bridge_config()
	plt.show()

if __name__=="__main__":
	main()