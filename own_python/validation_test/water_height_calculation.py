import numpy as np
import matplotlib.pyplot as ptl

m = 1
s = 1
g = 9.81 *(m/np.power(s, 2))

# L = 

def compute_Fr(u, L):
    return u/np.sqrt(g*L)

def compute_dhdx(i, J, Fr):
    return (i - J)/(1 - Fr**2 )

def compute_J (u, h, K, R_h, S):
    Q = u*h
    return np.power(Q, 2)/(np.power(K, 2)*np.power(S, 2), np.power(R_h, 4/3))

def update_h(h_0, dhdx, dx):
    return h_0 + dhdx*dx

def update_x(x_0, dhdx, dh):
    return x_0 + dh/dhdx



# Assumption 1: circular section
# Assumption 2: wetted section = total cross section
R = 1
p = 2*np.pi*R
A = np.pi*R*R
R_h = A/p

K_s = 100 #  slip wall condition


# Assumption 3 : uniform outlet velocity = 5 m/s
for i in range(100):


    dhdx = compute_dhdx(i, J, Fr)