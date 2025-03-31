import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import os
from tkinter import filedialog
import tkinter as tk

# Variable globale pour stocker la référence à la ligne CSV
csv_line = None

def compute_u_th(y_line, U_0, delta, gamma):
    y_half = np.linspace(0, 0.5, len(y_line)//2)  # semi line y
    
    U_L = U_0 * 4 * y_half * (1 - y_half)  # laminar
    U_T = U_0 * (1 - np.exp(1 - np.exp(y_half / delta)))  # turublent
    U_GHE_half = gamma * U_T + (1 - gamma) * U_L  # hybrid model
    
    y_full = np.concatenate((-y_half[::-1], y_half))
    U_full = np.concatenate((U_GHE_half, U_GHE_half[::-1])) 
    U_L_full = np.concatenate((U_L, U_L[::-1]))
    U_T_full = np.concatenate((U_T, U_T[::-1]))
        
    return U_full, U_L_full, U_T_full

# Définition des paramètres initiaux
U_0_init = 7.5
delta_init = 1.0
gamma_init = 0.5

# Création de la figure et des axes
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Création de la grille y
y_line = np.linspace(-0.5, 0.5, 200)

# Calcul des vitesses initiales
U_full, _, _ = compute_u_th(y_line, U_0_init, delta_init, gamma_init)

# Création des lignes (uniquement courbe hybride)
line_full, = ax.plot(np.linspace(-1.6155, 1.615, 200), U_full, 'b-', linewidth=3, label='Profil de vitesse (U)')

# Charger les données CSV automatiquement au démarrage
csv_line = None
try:
    # Vérifier si les deux fichiers nécessaires existent
    if os.path.exists('data.csv') and os.path.exists('y_line.csv'):
        print("Chargement des données depuis data.csv et y_line.csv")
        
        try:
            # Charger les vitesses
            velocities_df = pd.read_csv('data.csv')
            if len(velocities_df.columns) >= 2:
                velocity_values = pd.to_numeric(velocities_df.iloc[:, 1], errors='coerce')
                velocity_values = velocity_values[~np.isnan(velocity_values)].values
                
                # Charger les positions
                positions_df = pd.read_csv('y_line.csv')
                if len(positions_df.columns) >= 2:
                    position_values = pd.to_numeric(positions_df.iloc[:, 1], errors='coerce')
                    position_values = position_values[~np.isnan(position_values)].values
                    
                    # Ajuster les longueurs si nécessaire
                    min_length = min(len(velocity_values), len(position_values))
                    if min_length > 0:
                        velocity_values = velocity_values[:min_length]
                        position_values = position_values[:min_length]
                        
                        # Tracer les données
                        csv_line, = ax.plot(position_values, velocity_values, 'ko', markersize=4, alpha=0.7, label='Données expérimentales')
                        print(f"Données chargées: {min_length} points")
                    else:
                        print("Pas assez de données valides")
                else:
                    print("Format incorrect pour y_line.csv")
            else:
                print("Format incorrect pour data.csv")
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
    else:
        if not os.path.exists('data.csv'):
            print("Fichier data.csv non trouvé")
        if not os.path.exists('y_line.csv'):
            print("Fichier y_line.csv non trouvé")
except Exception as e:
    print(f"Erreur lors de la vérification des fichiers: {e}")

ax.set_ylabel('Vitesse', fontsize=14)
ax.set_xlabel('y', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True)
ax.set_title(f'Profil de vitesse (U_0={U_0_init}, delta={delta_init}, gamma={gamma_init})')

# Ajout des axes pour les curseurs
ax_U0 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_delta = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_gamma = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Création des curseurs
slider_U0 = Slider(ax_U0, 'U_0', 5.0, 10.0, valinit=U_0_init, valstep=0.1)
slider_delta = Slider(ax_delta, 'delta', 0.01, 10.0, valinit=delta_init, valstep=0.01)
slider_gamma = Slider(ax_gamma, 'gamma', 0.0, 1.0, valinit=gamma_init, valstep=0.01)

# Fonction pour sauvegarder les données en CSV
def save_data_to_csv():
    # Récupération des valeurs actuelles
    U_0 = slider_U0.val
    delta = slider_delta.val
    gamma = slider_gamma.val
    
    # Recalcul des profils de vitesse pour être sûr d'avoir les dernières valeurs
    U_full, _, _ = compute_u_th(y_line, U_0, delta, gamma)
    
    # Création d'un DataFrame avec toutes les données
    data = pd.DataFrame({
        'y': y_line,          # Position (abscisse)
        'U': U_full           # Vitesse hybrid (ordonnée)
    })
    
    # Ajout des paramètres utilisés comme métadonnées
    metadata = pd.DataFrame({
        'parameter': ['U_0', 'delta', 'gamma'],
        'value': [U_0, delta, gamma]
    })
    
    # Sauvegarde des données
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale
    
    file_path = filedialog.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[("CSV files", "*.csv")],
        title="Sauvegarder les données"
    )
    
    if file_path:
        # Sauvegarder les données principales
        data.to_csv(file_path, index=False)
        
        # Sauvegarder les métadonnées dans un fichier séparé
        metadata_path = file_path.replace('.csv', '_params.csv')
        metadata.to_csv(metadata_path, index=False)
        
        print(f"Données sauvegardées dans {file_path}")
        print(f"Paramètres sauvegardés dans {metadata_path}")
    
    root.destroy()

# Fonction pour charger des données expérimentales depuis deux fichiers CSV (vitesses et positions)
def load_experimental_data():
    global csv_line  # Déclaration de csv_line comme variable globale
    
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale
    
    # Demander à l'utilisateur de sélectionner le fichier des vitesses
    velocity_file = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Sélectionner le fichier des VITESSES"
    )
    
    if not velocity_file or not os.path.exists(velocity_file):
        print("Aucun fichier de vitesses sélectionné.")
        root.destroy()
        return
    
    # Demander à l'utilisateur de sélectionner le fichier des positions
    position_file = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Sélectionner le fichier des POSITIONS (y_line)"
    )
    
    if not position_file or not os.path.exists(position_file):
        print("Aucun fichier de positions sélectionné.")
        root.destroy()
        return
    
    try:
        # Charger les données de vitesse
        velocities_df = pd.read_csv(velocity_file)
        if len(velocities_df.columns) < 2:
            print("Format de fichier de vitesses incorrect: besoin d'au moins 2 colonnes")
            root.destroy()
            return
        
        # Utiliser la seconde colonne comme valeurs de vitesse
        velocity_values = pd.to_numeric(velocities_df.iloc[:, 1], errors='coerce')
        velocity_values = velocity_values[~np.isnan(velocity_values)].values
        
        # Charger les données de position
        positions_df = pd.read_csv(position_file)
        if len(positions_df.columns) < 2:
            print("Format de fichier de positions incorrect: besoin d'au moins 2 colonnes")
            root.destroy()
            return
        
        # Utiliser la seconde colonne comme valeurs de position
        position_values = pd.to_numeric(positions_df.iloc[:, 1], errors='coerce')
        position_values = position_values[~np.isnan(position_values)].values
        
        # Vérifier que nous avons assez de données
        if len(velocity_values) == 0 or len(position_values) == 0:
            print("Données invalides dans les fichiers CSV")
            root.destroy()
            return
        
        # Ajuster les longueurs des tableaux si nécessaire
        min_length = min(len(velocity_values), len(position_values))
        velocity_values = velocity_values[:min_length]
        position_values = position_values[:min_length]
        
        # Tracer les données combinées
        if csv_line is not None:
            csv_line.set_xdata(position_values)
            csv_line.set_ydata(velocity_values)
        else:
            csv_line, = ax.plot(position_values, velocity_values, 'ko', markersize=4, alpha=0.7, label='Données CSV')
            ax.legend()
        
        # Mise à jour du titre et des axes
        ax.set_title(f"Données expérimentales chargées ({min_length} points)")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        
        print(f"Données chargées: {min_length} points")
        
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers: {e}")
    
    root.destroy()

# Fonction de mise à jour
def update(val):
    # Récupération des valeurs des curseurs
    U_0 = slider_U0.val
    delta = slider_delta.val
    gamma = slider_gamma.val
    
    # Recalcul des profils de vitesse
    U_full, _, _ = compute_u_th(y_line, U_0, delta, gamma)
    
    # Mise à jour uniquement de la courbe hybride
    line_full.set_ydata(U_full)
    
    # Mise à jour du titre
    ax.set_title(f'Profil de vitesse (U_0={U_0:.2f}, delta={delta:.2f}, gamma={gamma:.2f})')
    
    # Ajustement automatique des axes si nécessaire
    ax.relim()
    ax.autoscale_view()
    
    # Rafraîchissement du graphique
    fig.canvas.draw_idle()

# Connexion des curseurs à la fonction de mise à jour
slider_U0.on_changed(update)
slider_delta.on_changed(update)
slider_gamma.on_changed(update)

# Ajout des boutons pour les différentes actions
button_ax_reset = plt.axes([0.05, 0.025, 0.1, 0.04])
button_ax_save = plt.axes([0.2, 0.025, 0.15, 0.04])
button_ax_load = plt.axes([0.4, 0.025, 0.2, 0.04])

reset_button = plt.Button(button_ax_reset, 'Reset')
save_button = plt.Button(button_ax_save, 'Save Data')
load_button = plt.Button(button_ax_load, 'Load Experimental Data')

def reset(event):
    slider_U0.reset()
    slider_delta.reset()
    slider_gamma.reset()

reset_button.on_clicked(reset)
save_button.on_clicked(lambda event: save_data_to_csv())
load_button.on_clicked(lambda event: load_experimental_data())

# Connexion des curseurs à la fonction de mise à jour
slider_U0.on_changed(update)
slider_delta.on_changed(update)
slider_gamma.on_changed(update)

plt.show()