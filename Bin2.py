# -*- coding: utf-8 -*-

from mpi4py import MPI
from interpolation_rbf import *  # Ensure the module name is correct
from PostParticle import *  # Ensure this module contains the necessary functions
import numpy as np
import matplotlib.pyplot as plt
import pickle

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # MPI process rank
size = comm.Get_size()  # Number of MPI processes

# Case configuration
case = '/DISK2/Anis/GWU/5 Hz - H1 - G12.dat'
SAVE_DIRECTORY = '/DISK2/Anis/GWU/plot'

# Experimental configuration
R = 14.25 / 2  # Cylinder radius
p = 19  # Distance between cylinder centers
gap = 4.75  # Bypass direction Z
y_offset = p + R + p / 2
z_offset = 66.81960000000001 + (0.81 + 0.5) * 0

# Numerical configuration
dx, dy, dz = 0.05, 0.04, 0.04  # Evaluation grid parameters

# RBF regression parameters
dxp, dyp, dzp = 1, 0.75, 0.5  # Spacing between patch centers
overlap = 0.45  # Overlap factor between patches
n_halton_points = 100000  # Total number of Halton collocation points
kappa_l = 1e12  # Constant for calculating alpha in RBF regression
L = 10  # Characteristic length of the measurement volume

if rank == 0:
    """
        Read the case and build the grid
    """
    # Read case data
    data, cylinder_centers = read_case(case, y_offset, z_offset, p, R, dimension="3D")
    data = data.dropna(subset=['x', 'y', 'z'])

    # Generate cylinder centers
    cylinder_centers = []
    for i in range(6):
        for j in range(6):
            xc = 26.125 + i * p
            yc = 61.875 + j * p + 0.6 * 0 - 0.5
            if j == 0:
                yc = 61.875 - 1.5
            cylinder_centers.append((xc, yc))

    # Prepare training data
    X_train = data[['x', 'y', 'z']].to_numpy()  # Positions (x, y, z)
    U_train = data[['Vx', 'Vy', 'Vz']].to_numpy()  # Velocities (Vx, Vy, Vz)

    # Define domain bounds
    bounds = [
        (X_train[:, 0].min(), X_train[:, 0].max()),
        (X_train[:, 1].min(), X_train[:, 1].max()),
        (X_train[:, 2].min(), X_train[:, 2].max())
    ]

    # Patch and RBF parameters
    patch_radius = np.sqrt(dxp**2 + dyp**2 + dzp**2) / np.sqrt(3) * (1 + overlap)
    epsilon = 1 / L

    # Generate patch centers and Halton collocation points
    patch_centers = generate_patch_centers(bounds, dxp, dyp, dzp)
    halton_points = generate_halton_points(n_halton_points, bounds)

    # Define evaluation grid
    a = (36, 77.5)
    b = (37.5, 66)
    X_eval = prepare_eval_domain(X_train, cylinder_centers, a, b, R, dx, dy, dz)

    print(f'Patch radius = {patch_radius}')
    print(f'Epsilon = {epsilon}')
    print(f'Total number of patches = {len(patch_centers)}')
    print(f'Total Halton points = {len(halton_points)}')
else:
    X_train = None
    U_train = None
    patch_centers = None
    halton_points = None
    X_eval = None
    epsilon = None
    patch_radius = None
    bounds = None
    cylinder_centers = None

# Synchronize data across all processes
X_train = comm.bcast(X_train, root=0)
U_train = comm.bcast(U_train, root=0)
patch_centers = comm.bcast(patch_centers, root=0)
halton_points = comm.bcast(halton_points, root=0)
X_eval = comm.bcast(X_eval, root=0)
epsilon = comm.bcast(epsilon, root=0)
patch_radius = comm.bcast(patch_radius, root=0)
bounds = comm.bcast(bounds, root=0)
cylinder_centers = comm.bcast(cylinder_centers, root=0)
kappa_l = comm.bcast(kappa_l, root=0)

if rank == 0:
    print(f"Length of X_eval = {len(X_eval)}")
    print(f"Length of U_train = {len(U_train)}")
    print(f"Number of patch centers = {len(patch_centers)}")

"""
    Build the RBF-PUM model with MPI
"""
if rank == 0:
    print('Building mean field model...')
model_data = build_pum_rbf_model_mpi(X_train, U_train, epsilon, patch_centers,
                                     patch_radius, kappa_l, halton_points)
model_data = comm.bcast(model_data, root=0)  # Broadcast the mean model


# Evaluate on the evaluation grid
if rank == 0:
    print('Evaluating mean velocity field on the evaluation grid...')
U_eval_local, dUdx_local, dUdy_local, dUdz_local = evaluate_pum_rbf_model_mpi(model_data, X_eval)

# Gather results on the master process
U_eval_list = comm.gather(U_eval_local, root=0)
dUdx_list = comm.gather(dUdx_local, root=0)
dUdy_list = comm.gather(dUdy_local, root=0)
dUdz_list = comm.gather(dUdz_local, root=0)

if rank == 0:
    U_eval = np.concatenate(U_eval_list, axis=0)
    dUdx = np.concatenate(dUdx_list, axis=0)
    dUdy = np.concatenate(dUdy_list, axis=0)
    dUdz = np.concatenate(dUdz_list, axis=0)

    # Save combined data
    with open('evaluation_data.pkl', 'wb') as f:
        pickle.dump((X_eval, U_eval, dUdx, dUdy, dUdz), f)
    print("Evaluation variables have been saved in 'evaluation_data.pkl'.")

# Only the master process performs visualization
if rank == 0 and U_eval.size > 0:
    """
        Visualize the reconstructed field
    """
    print("Evaluation complete. Plot 1...")

    # Plot Vx - YZ - x = 5
    normal = 'x'
    point = 5
    tolerance = None

    # Make sure the extract_slice function is correctly defined and imported
    slice_2d_coords, slice_U = extract_slice(X_eval, U_eval, normal, point, tolerance)
    U = slice_U[:, 0]

    y_vals, z_vals = slice_2d_coords[:, 1], slice_2d_coords[:, 2]  # Adjust indices
    y_unique = np.unique(y_vals)
    z_unique = np.unique(z_vals)
    Y, Z = np.meshgrid(y_unique, z_unique)
    U_grid = np.zeros_like(Y)
    for i in range(len(y_unique)):
        for j in range(len(z_unique)):
            idx = np.where((y_vals == y_unique[i]) & (z_vals == z_unique[j]))
            if len(idx[0]) > 0:
                U_grid[j, i] = U[idx[0][0]]

    plt.figure(figsize=(8, 6))
    plt.contourf(Y, Z, U_grid * 1000, cmap='turbo', levels=50)
    plt.title(f'Mean axial velocity at {normal} = {point} mm')
    plt.xlabel('Y' if normal == 'x' else 'X')
    plt.ylabel('Z' if normal in ['x', 'y'] else 'Y')
    plt.colorbar(label='Axial velocity [mm/s]')
    plt.savefig(f'{SAVE_DIRECTORY}/Vx_2D_{normal}.png')
    print(f'Plot saved in directory {SAVE_DIRECTORY}/')
    plt.show()

    # Add more visualizations if necessary...
