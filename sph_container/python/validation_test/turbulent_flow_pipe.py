import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def compute_u_th(y, U_0, delta, gamma):
    
    U_L = U_0 * 4*(y)*(1-y)
    U_T = U_0 * (1 - np.exp(1 - np.exp(y/delta)))
    U_GHE = gamma*U_T + (1-gamma)*U_L

    return U_GHE[::-1]
