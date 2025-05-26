import matplotlib.pyplot as plt
import numpy as np

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

# Configuration du graphique
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Définition du domaine
x = np.linspace(7, 15, 1000)
z_obstacle = parabole(x)

# Fond du canal (plus épais)
ax.plot([7, 15], [0, 0], 'k-', linewidth=4, label='Channel bed')

# Obstacle parabolique (ligne épaisse seulement)
x_obstacle = x[(x >= 8) & (x <= 12)]
z_obstacle_plot = parabole(x_obstacle)
ax.plot(x_obstacle, z_obstacle_plot, color='black', linewidth=5, label="Topography")
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
ax.text(x_outlet + 0.3, 0.15, '$U_{outlet}$', fontsize=12, color='red', fontweight='bold')
ax.text(7, 0.15, '$Q_{0}$', fontsize=12, color='green', fontweight='bold')


# Annotation de l'équation de l'obstacle
ax.text(10, 0.25, r'$Z(x) = 0.2 - 0.05(x-10)^2$', fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        ha='center')

# Annotation de la longueur
ax.annotate('', xy=(7, -0.05), xytext=(15, -0.05),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
ax.text(11, -0.08, 'L = 8 [m]', fontsize=12, ha='center', 
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))



# Configuration des axes et labels
ax.set_xlim(6.8, 15.5)
ax.set_ylim(-0.12, 0.4)
ax.set_xlabel('Distance x [m]', fontsize=12)
ax.set_ylabel('Height z [m]', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('auto')

plt.legend()

plt.tight_layout()
plt.show()