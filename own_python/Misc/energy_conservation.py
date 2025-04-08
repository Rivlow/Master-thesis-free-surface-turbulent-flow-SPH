import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres de simulation
k = 1.0       # Constante de ressort
m = 1.0       # Masse
T = 30.0      # Temps total de simulation
dt = 0.1      # Pas de temps
steps = int(T / dt)

# Conditions initiales (position et vitesse)
x0 = 1.0
v0 = 0.0

# Fonction pour calculer l'énergie totale (cinétique + potentielle)
def total_energy(x, v):
    kinetic = 0.5 * m * v**2        # Énergie cinétique
    potential = 0.5 * k * x**2       # Énergie potentielle
    return kinetic + potential

# Accélération pour un oscillateur harmonique: a = -k*x/m
def acceleration(x):
    return -k * x / m

# 1. Méthode d'Euler explicite
def euler_explicit():
    x = np.zeros(steps)
    v = np.zeros(steps)
    energy = np.zeros(steps)
    
    x[0] = x0
    v[0] = v0
    energy[0] = total_energy(x[0], v[0])
    
    for i in range(1, steps):
        # Mise à jour selon Euler explicite
        x[i] = x[i-1] + v[i-1] * dt
        v[i] = v[i-1] + acceleration(x[i-1]) * dt
        energy[i] = total_energy(x[i], v[i])
    
    return x, v, energy

# 2. Méthode d'Euler implicite
def euler_implicit():
    x = np.zeros(steps)
    v = np.zeros(steps)
    energy = np.zeros(steps)
    
    x[0] = x0
    v[0] = v0
    energy[0] = total_energy(x[0], v[0])
    
    for i in range(1, steps):
        # Pour un oscillateur harmonique, on peut résoudre analytiquement
        # l'équation implicite: v[i] = v[i-1] + a(x[i]) * dt
        # où x[i] = x[i-1] + v[i] * dt
        # Cela donne une formule pour v[i] puis x[i]
        
        # Résolution du système implicite pour l'oscillateur harmonique
        v[i] = (v[i-1] - k * x[i-1] * dt / m) / (1 + k * dt**2 / m)
        x[i] = x[i-1] + v[i] * dt
        energy[i] = total_energy(x[i], v[i])
    
    return x, v, energy

# 3. Méthode d'Euler symplectique (semi-implicite)
def euler_symplectic():
    x = np.zeros(steps)
    v = np.zeros(steps)
    energy = np.zeros(steps)
    
    x[0] = x0
    v[0] = v0
    energy[0] = total_energy(x[0], v[0])
    
    for i in range(1, steps):
        # Mise à jour selon Euler symplectique
        v[i] = v[i-1] + acceleration(x[i-1]) * dt
        x[i] = x[i-1] + v[i] * dt  # Utilise v[i] (mise à jour) au lieu de v[i-1]
        energy[i] = total_energy(x[i], v[i])
    
    return x, v, energy

# Solution analytique pour référence
def analytical_solution():
    t = np.linspace(0, T, steps)
    omega = np.sqrt(k / m)
    x = x0 * np.cos(omega * t)
    v = -x0 * omega * np.sin(omega * t)
    energy = np.array([total_energy(x[i], v[i]) for i in range(steps)])
    return x, v, energy, t

# Exécuter les simulations
x_explicit, v_explicit, energy_explicit = euler_explicit()
x_implicit, v_implicit, energy_implicit = euler_implicit()
x_symplectic, v_symplectic, energy_symplectic = euler_symplectic()
x_exact, v_exact, energy_exact, t = analytical_solution()

# Création des graphiques
plt.figure(figsize=(12, 10))

# Graphique des positions
plt.subplot(3, 1, 1)
plt.plot(t, x_exact, 'k-', label='Solution exacte')
plt.plot(t, x_explicit, 'r--', label='Euler explicite')
plt.plot(t, x_implicit, 'g--', label='Euler implicite')
plt.plot(t, x_symplectic, 'b--', label='Euler symplectique')
plt.xlabel('Temps')
plt.ylabel('Position')
plt.legend()
plt.title('Comparaison des trajectoires')
plt.grid(True)

# Graphique des vitesses
plt.subplot(3, 1, 2)
plt.plot(t, v_exact, 'k-', label='Solution exacte')
plt.plot(t, v_explicit, 'r--', label='Euler explicite')
plt.plot(t, v_implicit, 'g--', label='Euler implicite')
plt.plot(t, v_symplectic, 'b--', label='Euler symplectique')
plt.xlabel('Temps')
plt.ylabel('Vitesse')
plt.legend()
plt.title('Comparaison des vitesses')
plt.grid(True)

# Graphique des énergies
plt.subplot(3, 1, 3)
plt.plot(t, energy_exact, 'k-', label='Solution exacte')
plt.plot(t, energy_explicit, 'r--', label='Euler explicite')
plt.plot(t, energy_implicit, 'g--', label='Euler implicite')
plt.plot(t, energy_symplectic, 'b--', label='Euler symplectique')
plt.xlabel('Temps')
plt.ylabel('Énergie totale')
plt.legend()
plt.title('Comparaison de la conservation d\'énergie')
plt.grid(True)

plt.tight_layout()
plt.savefig('euler_comparison.png', dpi=300)
plt.show()

# Création d'un portrait de phase (vitesse vs position)
plt.figure(figsize=(10, 8))
plt.plot(x_exact, v_exact, 'k-', label='Solution exacte')
plt.plot(x_explicit, v_explicit, 'r--', label='Euler explicite')
plt.plot(x_implicit, v_implicit, 'g--', label='Euler implicite')
plt.plot(x_symplectic, v_symplectic, 'b--', label='Euler symplectique')
plt.xlabel('Position')
plt.ylabel('Vitesse')
plt.legend()
plt.title('Portrait de phase')
plt.grid(True)
plt.axis('equal')  # Même échelle pour les deux axes
plt.savefig('phase_portrait.png', dpi=300)
plt.show()

# Examinons l'évolution sur une plus longue période pour mieux voir les différences
# d'accumulation d'erreur dans l'énergie
T_long = 100.0
steps_long = int(T_long / dt)
t_long = np.linspace(0, T_long, steps_long)

def run_simulation_long(method):
    x = np.zeros(steps_long)
    v = np.zeros(steps_long)
    energy = np.zeros(steps_long)
    
    x[0] = x0
    v[0] = v0
    energy[0] = total_energy(x[0], v[0])
    
    for i in range(1, steps_long):
        if method == 'explicit':
            x[i] = x[i-1] + v[i-1] * dt
            v[i] = v[i-1] + acceleration(x[i-1]) * dt
        elif method == 'implicit':
            v[i] = (v[i-1] - k * x[i-1] * dt / m) / (1 + k * dt**2 / m)
            x[i] = x[i-1] + v[i] * dt
        elif method == 'symplectic':
            v[i] = v[i-1] + acceleration(x[i-1]) * dt
            x[i] = x[i-1] + v[i] * dt
        
        energy[i] = total_energy(x[i], v[i])
    
    return energy

energy_explicit_long = run_simulation_long('explicit')
energy_implicit_long = run_simulation_long('implicit')
energy_symplectic_long = run_simulation_long('symplectic')
energy_exact_long = np.ones(steps_long) * total_energy(x0, v0)  # Énergie constante

plt.figure(figsize=(12, 6))
plt.plot(t_long, energy_exact_long, 'k-', label='Solution exacte')
plt.plot(t_long, energy_explicit_long, 'r--', label='Euler explicite')
plt.plot(t_long, energy_implicit_long, 'g--', label='Euler implicite')
plt.plot(t_long, energy_symplectic_long, 'b--', label='Euler symplectique')
plt.xlabel('Temps')
plt.ylabel('Énergie totale')
plt.legend()
plt.title('Conservation d\'énergie sur longue durée')
plt.grid(True)
plt.savefig('Pictures/CH4_splishsplash/energy_long_term.png', dpi=300)
plt.show()

# Pour mieux visualiser l'erreur relative dans l'énergie
energy_error_explicit = np.abs((energy_explicit_long - energy_exact_long[0]) / energy_exact_long[0])
energy_error_implicit = np.abs((energy_implicit_long - energy_exact_long[0]) / energy_exact_long[0])
energy_error_symplectic = np.abs((energy_symplectic_long - energy_exact_long[0]) / energy_exact_long[0])

plt.figure(figsize=(12, 6))
plt.semilogy(t_long, energy_error_explicit, 'r-', label='Euler explicite')
plt.semilogy(t_long, energy_error_implicit, 'g-', label='Euler implicite')
plt.semilogy(t_long, energy_error_symplectic, 'b-', label='Euler symplectique')
plt.xlabel('Temps')
plt.ylabel('Erreur relative d\'énergie (échelle log)')
plt.legend()
plt.title('Erreur dans la conservation d\'énergie')
plt.grid(True)
plt.savefig('Pictures/CH4_splishsplash/energy_error.png', dpi=300)
plt.show()