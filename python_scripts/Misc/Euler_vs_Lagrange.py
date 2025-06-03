import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# Configuration
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Ajout des titres principaux pour les colonnes
fig.text(0.275, 0.96, 'Lagrangian', fontsize=18, fontweight='bold', ha='center')
fig.text(0.725, 0.96, ' Eulerian', fontsize=18, fontweight='bold', ha='center')

# Couleurs cohérentes
color_lagrangian = 'lightblue'
color_eulerian = 'lightcoral'
color_grid = 'black'
color_axis = 'blue'

# === Initial state (référence) ===
# Structure rectangulaire initiale
rect_corners = np.array([[1, 1], [3, 1], [3, 2.5], [1, 2.5], [1, 1]])

# Grille interne pour mieux visualiser la déformation
grid_x = np.linspace(1, 3, 5)
grid_y = np.linspace(1, 2.5, 4)

def plot_initial_state(ax, subtitle, color):
	ax.set_title(subtitle, fontsize=12, fontweight='normal')
	ax.set_xlim(0, 4)
	ax.set_ylim(0, 4)
	ax.set_aspect('equal')
	ax.grid(True, alpha=0.3)
	
	# Rectangle initial
	rect = Polygon(rect_corners[:-1], fill=True, facecolor=color, 
				edgecolor='black', linewidth=2, alpha=0.7)
	ax.add_patch(rect)
	
	# Grille interne
	for x in grid_x:
		ax.plot([x, x], [1, 2.5], 'k-', linewidth=1, alpha=0.5)
	for y in grid_y:
		ax.plot([1, 3], [y, y], 'k-', linewidth=1, alpha=0.5)
	
	# Repère de référence au nœud (2, 1.5)
	ax.arrow(2, 1.5, 0.4, 0, head_width=0.08, head_length=0.08, 
			fc=color_axis, ec=color_axis, linewidth=2)
	ax.arrow(2, 1.5, 0, 0.4, head_width=0.08, head_length=0.08, 
			fc=color_axis, ec=color_axis, linewidth=2)
	ax.text(2.32, 1.62, 'x', fontsize=12, color=color_axis, fontweight='bold')
	ax.text(1.7, 1.8, 'y', fontsize=12, color=color_axis, fontweight='bold')

# === DÉFORMATION APPLIQUÉE ===
def apply_deformation(x, y):
	"""Applique une déformation de cisaillement simple"""
	x_def = x + 0.3 * (y - 1)  # Cisaillement simple en x
	y_def = y  # Pas de déformation en y
	return x_def, y_def

# Calcul de la structure déformée
rect_deformed = np.array([apply_deformation(x, y) for x, y in rect_corners])

# Initial state - Lagrangien (bleu)
plot_initial_state(ax1, 'Initial State', color_lagrangian)

# Initial state - Eulérien (rouge)
plot_initial_state(ax2, 'Initial State', color_eulerian)

# === APPROCHE LAGRANGIENNE ===
ax3.set_title('Final State', fontsize=12, fontweight='normal')
ax3.set_xlim(0, 4)
ax3.set_ylim(0, 4)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Structure déformée (recalcul pour s'assurer de la déformation)
rect_corners_deformed = []
for x, y in rect_corners[:-1]:  # Exclut le dernier point qui est identique au premier
    x_def, y_def = apply_deformation(x, y)
    rect_corners_deformed.append([x_def, y_def])

rect_lag = Polygon(rect_corners_deformed, fill=True, facecolor=color_lagrangian, 
				edgecolor='black', linewidth=2, alpha=0.7)
ax3.add_patch(rect_lag)

# Grille déformée (suit la matière)
for x in grid_x:
	y_line = np.linspace(1, 2.5, 20)
	x_line = np.full_like(y_line, x)
	x_def, y_def = apply_deformation(x_line, y_line)
	ax3.plot(x_def, y_def, 'k-', linewidth=1, alpha=0.7)

for y in grid_y:
	x_line = np.linspace(1, 3, 20)
	y_line = np.full_like(x_line, y)
	x_def, y_def = apply_deformation(x_line, y_line)
	ax3.plot(x_def, y_def, 'k-', linewidth=1, alpha=0.7)

# Repère déformé (suit la matière) - même origine qu'initialement
ref_point = np.array([2.0, 1.5])  # Même position initiale
ref_def = apply_deformation(ref_point[0], ref_point[1])

# Vecteurs tangents déformés (même échelle qu'initialement)
eps = 0.1
dx_point = apply_deformation(ref_point[0] + eps, ref_point[1])
dy_point = apply_deformation(ref_point[0], ref_point[1] + eps)

vec_x = (np.array(dx_point) - np.array(ref_def)) / eps * 0.4
vec_y = (np.array(dy_point) - np.array(ref_def)) / eps * 0.4

# Repère déformé en bleu (même couleur qu'initialement)
ax3.arrow(ref_def[0], ref_def[1], vec_x[0], vec_x[1], 
		head_width=0.08, head_length=0.08, fc=color_axis, ec=color_axis, linewidth=2)
ax3.arrow(ref_def[0], ref_def[1], vec_y[0], vec_y[1], 
		head_width=0.08, head_length=0.08, fc=color_axis, ec=color_axis, linewidth=2)
ax3.text(ref_def[0] + vec_x[0] + 0.1, ref_def[1] + vec_x[1], 'x', 
		fontsize=12, color=color_axis, fontweight='bold')
ax3.text(ref_def[0] + vec_y[0]+0.1, ref_def[1] + vec_y[1] + 0.1, 'y', 
		fontsize=12, color=color_axis, fontweight='bold')

# === APPROCHE EULÉRIENNE ===
ax4.set_title('Final State', fontsize=12, fontweight='normal')
ax4.set_xlim(0, 4)
ax4.set_ylim(0, 4)
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)

# Structure déformée (couleur rouge pour Eulériens)
rect_eul = Polygon(rect_deformed[:-1], fill=True, facecolor=color_eulerian, 
				edgecolor='black', linewidth=2, alpha=0.7)
ax4.add_patch(rect_eul)

# Grille spatiale fixe (ne suit pas la déformation)
for x in grid_x:
	ax4.plot([x, x], [1, 2.5], 'k--', linewidth=1, alpha=0.7)
for y in grid_y:
	ax4.plot([1, 3], [y, y], 'k--', linewidth=1, alpha=0.7)

# Repère fixe (ne change pas) - même position qu'initialement
ax4.arrow(2, 1.5, 0.4, 0, head_width=0.08, head_length=0.08, 
		fc=color_axis, ec=color_axis, linewidth=2)
ax4.arrow(2, 1.5, 0, 0.4, head_width=0.08, head_length=0.08, 
		fc=color_axis, ec=color_axis, linewidth=2)
ax4.text(2.32, 1.62, 'x', fontsize=12, color=color_axis, fontweight='bold')
ax4.text(1.7, 1.8, 'y', fontsize=12, color=color_axis, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.25)  # Ajuste l'espace pour les titres principaux et sépare les rangées
plt.savefig('Pictures/CH2_sph_fund/euler_vs_lagrange.pdf', bbox_inches='tight', dpi=300)
plt.show()