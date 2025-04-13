import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from scipy.optimize import minimize
from shapely.geometry import Point, LineString, box
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from scipy.integrate import simpson
from scipy.optimize import curve_fit

sys.path.append((os.getcwd()))

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

def configure_latex():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')



def compute_u_ghe(y_line, U_0, delta, gamma):

    if len(y_line) % 2 == 0:  
        size_half = len(y_line) // 2
    else: 
        size_half = (len(y_line) + 1) // 2


    y_half = np.linspace(0, 0.5, size_half)
    
    U_L = U_0 * 4 * y_half * (1 - y_half)  # laminar
    U_T = U_0 * (1 - np.exp(1 - np.exp(y_half / delta)))  # turbulent
    U_GHE_half = gamma * U_T + (1 - gamma) * U_L  # hybrid model

    if len(y_line) % 2 == 0:
        y_full = np.concatenate((-y_half[::-1], y_half))
        U_full = np.concatenate((U_GHE_half, U_GHE_half[::-1]))
        U_L_full = np.concatenate((U_L, U_L[::-1]))
        U_T_full = np.concatenate((U_T, U_T[::-1]))
    else:
        y_full = np.concatenate((-y_half[::-1], y_half[1:]))
        U_full = np.concatenate((U_GHE_half, U_GHE_half[1:][::-1]))
        U_L_full = np.concatenate((U_L, U_L[1:][::-1]))
        U_T_full = np.concatenate((U_T, U_T[1:][::-1]))
    
    return U_full, U_L_full, U_T_full

def fit_ghe_model(u_all, y_all, y_min, plot=False, save=False):
    
    # Flatten all slices into single arrays
    u_flat = np.concatenate([u_slice for u_slice in u_all])
    y_flat = np.concatenate([y_slice for y_slice in y_all])
    
    # Sort by y position for cleaner plotting
    sort_idx = np.argsort(y_flat)
    u_flat_sorted = u_flat[sort_idx]
    y_flat_sorted = y_flat[sort_idx]
    
    # Alternatively, you can use mean profile as you were doing
    u_mean_exp = np.mean(u_all, axis=0)
    y_exp_single = y_all[0]
    
    # For fitting, you can either:
    # 1. Use the mean profile (as you were doing)
    # 2. Use all data points for a more robust fit
    
    U_0 = np.max(u_flat_sorted)  # or np.max(u_flat)
    
    # Rest of your fitting code...
    # To fit with all points:
    def fit_func(y, delta, gamma):
        # You'll need to modify this to handle unsorted y values
        # or pre-sort y_flat before fitting
        u_model = compute_u_ghe(y, U_0, delta, gamma)[0]
        return u_model
    
    initial_guess = [0.01, 0.5]
    
    # Choose which approach to use:
    # 1. Fit using mean profile (as you were doing):
    popt, pcov = curve_fit(fit_func, y_flat_sorted, u_flat_sorted, p0=initial_guess)
    delta_opt, gamma_opt = popt
    perr = np.sqrt(np.diag(pcov))
    
    u_fitted, u_laminar, u_turbulent = compute_u_ghe(u_flat_sorted, U_0, delta_opt, gamma_opt)
    y_plot = np.linspace(y_min, -y_min, len(u_fitted))
    
    print(f"Optimal param: delta = {delta_opt} +- {perr[0]}, gamma = {gamma_opt} +- {perr[1]}")

    if plot:
        configure_latex()
        plt.figure(figsize=(6.7, 5))
        plt.scatter(y_flat_sorted, u_flat_sorted, s=7, alpha = 0.7, color='blue', label= 'SPH data')
        plt.plot(y_plot, u_fitted, color = 'red', label='Fitted model')
        plt.xlabel('Diameter y [m]')
        plt.ylabel('Velocity u(y) [m/s]')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        
        plt.legend(loc='best')
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/u_fit_single.pdf')
    
    
    


def analyze_particle_distribution(vtk_data, x_slice, delta_x=0.1, n_bins=50, plot=True):
   
    
    points = vtk_data.points
    x_pos = points[:, 0]
    mask = (x_pos >= x_slice - delta_x/2) & (x_pos <= x_slice + delta_x/2) # [x - dx/2, x + dx/2]
    slice_points = points[mask]
    
    if len(slice_points) == 0:
        print(f"No particles found at  x = {x_slice} +- {delta_x/2}")
        return None, (None, None, None)
    
  
    # Y domain (for bins)
    y_pos = slice_points[:, 1]
    y_min, y_max = np.min(y_pos), np.max(y_pos)
    y_range = y_max - y_min
    bins = np.linspace(y_min - 0.05*y_range, y_max + 0.05*y_range, n_bins)
    
    # Histogram
    counts, bin_edges = np.histogram(y_pos, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Find potential low density regions
    bin_width = bin_edges[1] - bin_edges[0]
    density_per_bin = counts / bin_width
    mean_density = np.mean(density_per_bin)
    high_density_threshold = 1.5 * mean_density
    low_density_threshold = 0.5 * mean_density
    
    high_density_regions = bin_centers[density_per_bin > high_density_threshold]
    low_density_regions = bin_centers[density_per_bin < low_density_threshold]
    
    if plot:
        configure_latex()
        fig, ax = plt.subplots(figsize=(6.7, 5))
        ax.bar(bin_centers, counts, width=bin_width*0.9, alpha=0.7, color='steelblue')
        
        if len(high_density_regions) > 0:
            for y_pos in high_density_regions:
                ax.axvline(x=y_pos, color='red', linestyle='--', alpha=0.5)
        
        if len(low_density_regions) > 0:
            for y_pos in low_density_regions:
                ax.axvline(x=y_pos, color='orange', linestyle='--', alpha=0.5)
                
        ax.set_xlabel('Position y')
        ax.set_ylabel('Number of particules')
        ax.set_title(f'Particle distribution at x = {x_slice} +- {delta_x/2}')
        ax.grid(True, alpha=0.3)
        
        ax.legend()
        plt.tight_layout()
        

def computeRe(nu, D, U_0):
    Re = int((U_0*D)/nu)
    print(f"Maximal velocity = {U_0}")
    print(f'Reynolds number Re = {Re}')

    if Re < 2300:
        print('Laminar flow')
    else:
        print('Turbulent flow')




def find_particles_in_rectangle(points, x_min, y_min, x_max, y_max):

    rectangle = box(x_min, y_min, x_max, y_max)
    inside_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        shapely_point = Point(point[0], point[1])
        inside_mask[i] = rectangle.contains(shapely_point)
    
    return inside_mask, rectangle


def project_particles(vtk_data, mask, rectangle):

    # Particles inside the rectangle
    points = vtk_data.points
    inside_points = points[mask]
    
    min_x, min_y, max_x, maxy = rectangle.bounds
    middle_x = (min_x + max_x) / 2
    vertical_line = LineString([(middle_x, min_y), (middle_x, maxy)])
    
    # Project (orthogonaly) the particles on the vertical line
    projected_points = np.zeros_like(inside_points)
    for i, point_coords in enumerate(inside_points):

        shapely_point = Point(point_coords[0], point_coords[1])
        
        _, projected = nearest_points(shapely_point, vertical_line)
        
        projected_points[i, 0] = projected.x
        projected_points[i, 1] = projected.y

        # Keep the z coordinate if it exists
        if inside_points.shape[1] > 2:
            projected_points[i, 2] = point_coords[2] 
    
    projected_attributes = {}
    point_arrays = list(vtk_data.point_data.keys())
    
    for key in point_arrays:
        values = vtk_data[key]
        projected_attributes[key] = values[mask]
    
    # Sort the projected points by y coordinate
    if len(projected_points) > 0:
        sort_indices = np.argsort(projected_points[:, 1])
        projected_points = projected_points[sort_indices]
        
        for key in projected_attributes:
            projected_attributes[key] = projected_attributes[key][sort_indices]
    
    return projected_points, projected_attributes, vertical_line


def remove_part(u_data, rho_data, y_data, min_part):

    idx_sorted = np.argsort(rho_data)
    idx_kept = np.sort(idx_sorted[:min_part])
    return u_data[idx_kept], rho_data[idx_kept], y_data[idx_kept]
    

def single_slice(fully_dev_vtk, y_min, y_max, x_pos, slice_width, plot=False, save=False):
    
    u_all = []
    rho_all = []
    y_all = []

    x_min = x_pos - slice_width
    x_max = x_pos + slice_width

    current_min_particles = float('inf')

    for i, single_vtk in enumerate(fully_dev_vtk):
        if i == 0:
            continue

        inside_mask, rectangle = find_particles_in_rectangle(single_vtk.points, x_min, y_min, x_max, y_max)
        projected_points, projected_attributes, vertical_line = project_particles(
            single_vtk, inside_mask, rectangle)
        
        u_data = projected_attributes['velocity'][:, 0]
        rho_data = projected_attributes['density']
        y_data = projected_points[:, 1]

        nb_part = len(y_data)
        
        if nb_part < current_min_particles:
            current_min_particles = nb_part
            
            for j in range(len(u_all)):
                if len(u_all[j]) > nb_part:
                    u_all[j], rho_all[j], y_all[j] = remove_part(u_all[j], rho_all[j], y_all[j], nb_part)
                    
                    assert len(u_all[j]) == nb_part, f"Erreur: après remove_part, j={j}, len(u_all[j])={len(u_all[j])}, should be {nb_part}"
        
        rho_all.append(rho_data)
        u_all.append(u_data)
        y_all.append(y_data)
        
        if len(u_data) > current_min_particles:
            u_all[-1], rho_all[-1], y_all[-1] = remove_part(u_data, rho_data, y_data, current_min_particles)
            
            assert len(u_all[-1]) == current_min_particles

      
    u_min, u_max = [], []
    rho_min, rho_max = [], []
    if plot:
        configure_latex()
        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):

            plt.scatter(y_all[i], u_all[i], s=5)
            u_min.append(np.min(u_all[i]))
            u_max.append(np.max(u_all[i]))
            rho_min.append(np.min(rho_all[i]))
            rho_max.append(np.max(rho_all[i]))

        u_min_min = np.min(u_min)
        u_max_max = np.max(u_max)
        rho_min_min = np.min(rho_min)
        rho_max_max = np.max(rho_max)

        plt.ylabel('Velocity u(y) [m/s]')
        plt.xlabel('Distance y [m]')
        plt.ylim(u_min_min, u_max_max)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/u_single_x_{x_pos}.pdf')

        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], rho_all[i], s=5)
        plt.ylim(990, 1010)
        plt.ylabel(r'Density $\rho$ [kg/$m^3$]')
        plt.xlabel('Distance y [m]')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/rho_single_x_{x_pos}.pdf')

    return u_all, rho_all, y_all



def multiple_slices(vtk_data, 
                    x_start, x_end, 
                    num_slices, 
                    y_min, y_max, 
                    slice_width, 
                    plot=False,
                    save=False):
    
    x_positions = np.linspace(x_start, x_end, num_slices)
    u_all = []
    rho_all = []
    y_all = []
    current_min_particles = float('inf')
        
    for x_pos in x_positions:

        x_min = x_pos - slice_width
        x_max = x_pos + slice_width
        
        inside_mask, rectangle = find_particles_in_rectangle(vtk_data.points, x_min, y_min, x_max, y_max)   
          
        projected_points, projected_attributes, vertical_line = project_particles(
            vtk_data, inside_mask, rectangle)
        
        u_data = projected_attributes['velocity'][:, 0]
        rho_data = projected_attributes['density']
        y_data = projected_points[:, 1]
        
        nb_part = len(y_data)
        
        if nb_part < current_min_particles:
            current_min_particles = nb_part
            
            for j in range(len(u_all)):
                if len(u_all[j]) > nb_part:
                    u_all[j], rho_all[j], y_all[j] = remove_part(u_all[j], rho_all[j], y_all[j], nb_part)
                    
                    assert len(u_all[j]) == nb_part, f"Erreur: après remove_part, j={j}, len(u_all[j])={len(u_all[j])}, should be {nb_part}"
        
        
        
        
        u_all.append(u_data)
        rho_all.append(rho_data)
        y_all.append(y_data)

        if len(u_data) > current_min_particles:
            u_all[-1], rho_all[-1], y_all[-1] = remove_part(u_data, rho_data, y_data, current_min_particles)
            
            assert len(u_all[-1]) == current_min_particles, f"Erreur: après remove_part pour le dernier élément, len(u_all[-1])={len(u_all[-1])}, should be {current_min_particles}"
    
        
    if plot:
        configure_latex()
        u_min, u_max = [], []
        rho_min, rho_max = [], []

        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], u_all[i], s=5)
            u_min.append(np.min(u_all[i]))
            u_max.append(np.max(u_all[i]))
        
        u_min_min = np.min(u_min)
        u_max_max = np.max(u_max)
        
            
        #plt.legend()
        plt.ylabel('Velocity u(y) [m/s]')
        plt.xlabel('Distance y [m]')
        #plt.ylim(u_min_min, u_max_max)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/u_multiple.pdf')

        plt.figure(figsize=(6.7, 5))
        for i in range(len(u_all)):
            plt.scatter(y_all[i], rho_all[i], s=5)
            rho_min.append(np.min(rho_all[i]))
            rho_max.append(np.max(rho_all[i]))

        rho_min_min = np.min(rho_min)
        rho_max_max = np.max(rho_max)

        plt.ylabel(r'Density $\rho$ [kg/$m^3$]')
        plt.xlabel('Distance y [m]')
        #plt.ylim(rho_min_min, rho_max_max)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'Pictures/CH5_valid_test/turbulent/rho_multiple.pdf')

        plt.show()

    return u_all, y_all

def visualize_results(vtk_data, inside_mask, projected_points, rectangle, vertical_line):

    plt.figure(figsize=(6.7, 5))

    configure_latex()

    points = vtk_data.points
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5, label='All particles')
    inside_points = points[inside_mask]
    plt.scatter(inside_points[:, 0], inside_points[:, 1], s=3, c='red', label='Framed particles ')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=3, c='green')
    
    x, y = rectangle.exterior.xy
    plt.plot(x, y, 'b-')
    x, y = vertical_line.xy
    plt.plot(x, y, 'g-', linewidth=2, label='Projected particles')
    #plt.axis('equal')
    plt.xlim((34.5 , 36))
    plt.ylim(rectangle.bounds[1] - 0.3 , 1.2*rectangle.bounds[3])
    plt.ylabel('Distance y [m]')
    plt.xlabel('Position x [m]')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('Pictures/CH5_valid_test/turbulent/euler_area.pdf', )


def integrate_slice(Q_init, x_span, u_all, y_all):
    integrals = []
    valid_indices = []
    
    for idx, (u, y) in enumerate(zip(u_all, y_all)):
        u, y = np.array(u), np.array(y)
        
        if not np.all(np.diff(y) > 0):
            sort_idx = np.argsort(y)
            y, u = y[sort_idx], u[sort_idx]
        
        int_value = simpson(u, x=y)
        if int_value < 0:
            print(f"Negative integral at x = {x_span[idx]}")
            continue
            
        integrals.append(int_value)
        valid_indices.append(idx)
    
    filtered_x_span = x_span[valid_indices]
    print(f"Valid number of integrals: {len(integrals)} over {len(u_all)}")
    
    # Filtrer les outliers
    p95 = np.percentile(integrals, 90)
    outlier_filter = [i for i, val in enumerate(integrals) if val <= p95]
    filtered_integrals = [integrals[i] for i in outlier_filter]
    filtered_mean = np.mean(filtered_integrals)
    filtered_x = filtered_x_span[outlier_filter]
    
    # Graphiques
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(filtered_x_span, integrals, width=0.6, alpha=0.7, color='royalblue')
    plt.hlines(Q_init, 0, np.max(x_span), color='red', linestyle='--', linewidth=2, label=r'$Q_{\text{init}}$')
    plt.hlines(filtered_x_span, 0, np.mean(integrals), color='darkgreen', linestyle='-', linewidth=2, label='Mean')
    plt.xlabel('Position x [m]')
    plt.ylabel('Integral int u(y)dy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(filtered_x, filtered_integrals, alpha=0.7, color='seagreen')
    plt.hlines(Q_init, 0, np.max(x_span), color='red', linestyle='--', linewidth=2, label='Q_init')
    plt.hlines(filtered_mean, 0, np.max(x_span), color='darkgreen', linestyle='-', linewidth=2, label='Mean')
    plt.xlabel('Position x')
    plt.ylabel('Integral intu(y)dy')
    plt.title('95e percentile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    #plt.tight_layout()
    #plt.savefig('Pictures/CH5_valid_test/slice_integrals_comparison.pdf', dpi=300)
    plt.show()
    
    return integrals, filtered_integrals


def main():

    U_0 = 5
    delta = 0.1
    gamma = 0.7

    u_ghe = compute_u_ghe(np.linspace(-0.5, 0.5, 100), U_0, delta, gamma)[0]
    plt.figure()
    plt.plot(np.linspace(-0.5, 0.5, 100), u_ghe)
    plt.show()

if __name__=="__main__":
    main()

