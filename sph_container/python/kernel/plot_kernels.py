import numpy as np
import matplotlib.pyplot as plt
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from kernels import *

# Style configuration
SMALL_SIZE = 10
MEDIUM_SIZE = 14
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
    save = False
    latex = False
    
    kernel_name = "cubic"
    kernel_func = extractKernel(kernel_name)
    gradient_func = extractGradients(kernel_name)    
    r = np.linspace(1.0e-9, 1, 1000) # Avoid r=0 for gradient calculation
    r_max = 1
    q = r/r_max
    
    W = kernel_func(r, r_max)
    dW = gradient_func(r, r_max)

    plt.figure(figsize=(10, 6))
    plt.plot(q, W, 'b-')
    plt.xlabel('$q = r/h$' if latex else 'q = r/h')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()