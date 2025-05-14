import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from math import pi

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from kernels import *

# Style configuration
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def isLatex(latex):
	if latex:
		plt.rc('text', usetex=True)
		plt.rc('font', family='lmodern')


def main():
	# Example usage
	save = True
	latex = True
	isLatex(latex)  # Activez LaTeX si demandé

	kernel_name = "cubic"
	kernel_func = extractKernel(kernel_name)
	gradient_func = extractGradients(kernel_name)    
	
	# Utiliser r/h de 0 à 0.6 comme dans le graphique de référence
	h = 1.0  # Rayon de lissage
	r = np.linspace(0.0, 0.6*h, 1000)
	q = r/h
	
	# Calculer le coefficient alpha à partir de la fonction de noyau cubique
	h3 = h * h * h
	alpha = 8.0 / (pi * h3)  # Coefficient k dans vos fonctions
	
	# Calculer W
	W = kernel_func(r, h)
	W_norm = W / alpha
	
	# Pour éviter la division par zéro à r=0 pour le gradient
	r_grad = np.linspace(1.0e-9, 0.6*h, 1000)
	q_grad = r_grad/h
	dW = gradient_func(r_grad, h)
	dW_norm = dW / (alpha/h)  # dW/(α/h)
	
	# Créer une seule figure avec les deux courbes
	plt.figure(figsize=(10, 8))
	
	# Tracer W(r,h)/α
	plt.plot(q, W_norm, 'b-', linewidth=2, label='$W(r,h)/\\alpha$' if latex else 'W(r,h)/α')
	
	# Tracer dW(r,h)/dr/(α/h)
	plt.plot(q_grad, dW_norm, 'r--', linewidth=2, label='$dW(r,h)/dr/(\\alpha/h)$' if latex else 'dW(r,h)/dr/(α/h)')
	
	plt.xlabel('$r/h$' if latex else 'r/h')
	plt.ylabel('Value')
	plt.grid(True)
	plt.legend()
	
	if save:
		plt.savefig(f"Pictures/CH4_splishsplash/W_and_gradW_{kernel_name}.pdf", bbox_inches='tight')

	plt.show()


if __name__ == "__main__":
	main()