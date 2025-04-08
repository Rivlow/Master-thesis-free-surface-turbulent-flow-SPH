import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.optimize import fsolve, minimize_scalar
from scipy.interpolate import splev, splrep



# Hydraulic functions
def compute_Fr(q, g, h):
    return q / (np.sqrt(g) * h**(3/2))

def solve_height(q, g, x, z_b, H):
    def eq(h): 
        if h <= 0.001:  
            return float('inf')
        return z_b(x) + h + q**2 / (2 * g * h**2) - H
    
    h0 = 0.1  # Initial guess
    return fsolve(eq, h0, full_output=True)

def conjugate_height(q, g, h):

    Fr = compute_Fr(q, g, h)
    return h * 0.5 * (np.sqrt(1 + 8 * Fr**2) - 1)


def solve_height_amont(q, g, x, z_b, H):

    # h³ + (z_b(x) - H)·h² + 0·h + q²/(2*g)
    a = z_b(x) - H
    b = 0 
    c = q**2 / (2 * g)
    
    coeffs = [1, a, b, c]
    roots = np.roots(coeffs)
    
    real_positive_roots = np.array([root.real for root in roots if root.real > 0])

    if 10 < x <= 12 :
        return np.min(real_positive_roots)
    elif x == 10:
         return real_positive_roots[0]
    else:
        return np.max(real_positive_roots)
    
def solve_height_aval(q, g, x, z_b, H):

    # h³ + (z_b(x) - H)·h² + 0·h + q²/(2*g)
    a = z_b(x) - H
    b = 0 
    c = q**2 / (2 * g)
    
    coeffs = [1, a, b, c]
    roots = np.roots(coeffs)
    
    real_positive_roots = np.array([root.real for root in roots if root.real > 0])

    if 10 < x <= 11 :
        return np.min(real_positive_roots)
    elif x == 10:
         return real_positive_roots[0]
    else:
        return np.max(real_positive_roots)
    
def extract_water_height(vtk_file, save=False):

    points = np.array(vtk_file.points)

    #Sort particles along x axis
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Sample distance x in bins
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    num_bins = 500 
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)

    # Highest (y_max) point on each bin
    surface_points = []
    for i in range(num_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i+1]
        mask = (sorted_points[:, 0] >= bin_start) & (sorted_points[:, 0] < bin_end)
        bin_points = sorted_points[mask]
        
        if len(bin_points) > 0:
            max_y_index = np.argmax(bin_points[:, 1]) # y_max
            surface_points.append(bin_points[max_y_index])

    surface_points = np.array(surface_points)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.3, color='blue', label='Particles')
    ax.scatter(surface_points[:, 0], surface_points[:, 1], s=20, color='red', label='Water height')

    ax.set_xlabel('Position X')
    ax.set_ylabel('Height')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig('water_height_sph.png', dpi=300)
    plt.show()

def analytical_water_height():
    pass


def main():

    cm = 1e-2
    m = 1 
    s = 1

    g = 9.81*(m/(s**2))          

    # Parabola function
    z_b = lambda x: 0.2 - 0.05*((x-10)**2) if 8 <= x <= 12 else 0 

    U_0 = 0.36*(m/s)
    D = 0.5*(m)
    
    # Boundary conditions
    q = 0.18*(m**2/s)             # inlet massflow
    h_aval = 33*cm                # outlet water height
    Fr_aval = 0.3                 # outlet Fr

    # Compute critical height
    h_cr = (q**2/g)**(1/3)       
    x_cr = 10*m
    x_ressaut = 11.6657 # evaluated before
    
    # Using Bernoulli, up/downstream total head
    H_amont = z_b(10) + h_cr + q**2/(2*g*h_cr**2)
    H_aval = 0 + h_aval + q**2/(2*g*h_aval**2)
    
    nb_elem = 2000
    x_amont = np.linspace(0, x_ressaut, nb_elem)
    x_aval = np.linspace(13, x_ressaut, nb_elem)

    # Topography
    z_aval = np.array([z_b(x_aval[i]) for i in range(len(x_aval))])
    z_amont = np.array([z_b(x_amont[i]) for i in range(len(x_amont))])

    # Compute water height from "z_b(x) + h(x) + (q²/2g)(1/h²(x)) = H_aval (or H_amont)"
    h_amont = np.array([solve_height_amont(q, g, x, z_b, H_amont) for x in x_amont])
    h_aval = np.array([solve_height_aval(q, g, x, z_b, H_aval) for x in x_aval])

    h_inter = np.linspace(h_amont[-1], h_aval[-1], 100)
    x_inter = np.linspace(x_amont[-1], x_aval[-1], 100)
    z_inter = np.array([z_b(x_inter[i]) for i in range(len(x_inter))])
    
    # Froude numbers over whole distance
    Fr_amont = np.array([compute_Fr(q, g, h) for h in h_amont])
    Fr_aval = np.array([compute_Fr(q, g, h) for h in h_aval])
    Fr_inter = np.array([compute_Fr(q, g, h) for h in h_inter])

    # Conjugated heights h2 from h1(x_aval)
    h2_conj = []
    for i, h1 in enumerate(h_amont):
        h2_conj.append(conjugate_height(q, g, h1))
    h2_conj = np.array(h2_conj)
       
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x_all = np.concatenate((x_amont, x_inter, x_aval))
    z_all = np.concatenate((z_amont, z_inter, z_aval))
    h_all = np.concatenate((h_amont, h_inter, h_aval))
    Fr_all = np.concatenate((Fr_amont, Fr_inter, Fr_aval))

    ax1.scatter(x_inter, h_inter+z_inter, s=5, color='red')
    ax1.plot(x_all, h_all+z_all)

    '''
    # Plot water height
    ax1.plot(x_aval, h_aval + z_aval, 'b-')
    ax1.plot(x_amont, h_amont + z_amont, 'b-')
    ax1.plot(x_inter, h_inter + z_inter, 'b-', label='Free surface')

    #ax1.plot(x_amont, z_amont + h2_conj, linestyle='--', label=r'conjugated $h2_{\text{(x_amont)}}$')
    ax1.plot(x_amont, z_amont, 'k-')
    ax1.plot(x_aval, z_aval, 'k-', label='Topography')

    ax1.fill_between(x_aval, z_aval, z_aval + h_aval, color='lightblue', alpha=0.5)
    ax1.fill_between(x_amont, z_amont, z_amont + h_amont, color='lightblue', alpha=0.5)

    # Test
    z_aval_extned = np.array([z_b(x) for x in np.linspace(13, 9, nb_elem)])
    #ax1.plot(np.linspace(13, 9, nb_elem), h_aval_extend + z_aval_extned, 'g-',)

    # Plot Froude number 
    ax2 = ax1.twinx()
    ax2.plot(x_amont, Fr_amont, 'r',linestyle='--')
    ax2.plot(x_inter, Fr_inter, 'r',linestyle='--')
    ax2.plot(x_aval, Fr_aval, 'r', linestyle='--', label='Fr')
    ax2.set_ylabel('Froude number Fr [-]')
    ax2.tick_params(axis='y')
    
    # Plot params
    
    ax1.set_ylabel('Water height h[m]')
    ax1.set_xlabel('Distance x [m]')
    ax1.legend(loc = 'best')
    ax1.set_xlim(7, 13)
    ax1.grid()
    '''
    plt.show()


if __name__=="__main__":
    main()