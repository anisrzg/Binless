# interpolate_rbf.py

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance_matrix
from scipy.stats import qmc
from mpi4py import MPI
from tqdm import tqdm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
    Fonction RBF Gaussienne
"""
def gaussian_rbf(X, centers, epsilon):
    # X: (n_points, n_dims)
    # centers: (n_centers, n_dims)
    # epsilon: float
    dists = distance_matrix(X, centers)  # (n_points, n_centers)
    Φ = np.exp(- (epsilon * dists) ** 2)
    return Φ  # (n_points, n_centers)

def gaussian_rbf_derivative_first(X, centers, epsilon, axis):
    # X: (n_points, n_dims)
    # centers: (n_centers, n_dims)
    # axis: int, 0 for x, 1 for y, 2 for z
    dists = distance_matrix(X, centers)  # (n_points, n_centers)
    Φ = np.exp(- (epsilon * dists) ** 2)
    diff = X[:, axis].reshape(-1, 1) - centers[:, axis].reshape(1, -1)  # (n_points, n_centers)
    dΦdX = -2 * epsilon ** 2 * diff * Φ
    return dΦdX  # (n_points, n_centers)

def calculate_alpha(H, kappa_l):
    # H : matrice carrée symétrique (nb x nb)
    eigenvalues = np.linalg.eigvalsh(H)
    lambda_max = np.max(eigenvalues)
    alpha = lambda_max / kappa_l
    return alpha

def wendland_c2(r):
    # Fonction de Wendland C2 compacte et à support compact
    # r : distances normalisées (r / rayon)
    result = np.where(r < 1, (1 - r) ** 4 * (4 * r + 1), 0)
    return result

"""
    Fonctions de régression
"""
def rbf_regression_divergence_free(X_collocation, X_train_patch, U_train_patch, epsilon, kappa_l, n_constraints):
    # Calcul de la matrice des RBF
    Φ_train = gaussian_rbf(X_train_patch, X_collocation, epsilon)
    
    # Construction de la matrice bloc diagonale Φ_train_block
    Φ_train_block = np.block([
        [Φ_train, np.zeros_like(Φ_train), np.zeros_like(Φ_train)],
        [np.zeros_like(Φ_train), Φ_train, np.zeros_like(Φ_train)],
        [np.zeros_like(Φ_train), np.zeros_like(Φ_train), Φ_train]
    ])
    
    # Vecteur des vitesses d'entraînement empilé
    U_train_vector = U_train_patch.flatten(order='F')
    
    # Sélection des points de contrainte
    if n_constraints >= X_collocation.shape[0]:
        n_constraints = X_collocation.shape[0]
        X_constraints = X_collocation
    else:
        # Par exemple, sélectionner des points régulièrement espacés
        idx_constraints = np.linspace(0, X_collocation.shape[0]-1, n_constraints, dtype=int)
        X_constraints = X_collocation[idx_constraints]
    
    # Calcul des dérivées des RBF aux points de contrainte
    Φ_dx = gaussian_rbf_derivative_first(X_constraints, X_collocation, epsilon, axis=0)
    Φ_dy = gaussian_rbf_derivative_first(X_constraints, X_collocation, epsilon, axis=1)
    Φ_dz = gaussian_rbf_derivative_first(X_constraints, X_collocation, epsilon, axis=2)
    
    # Construction de la matrice de divergence D∇
    D_divergence = np.hstack([Φ_dx, Φ_dy, Φ_dz])
    
    # Calcul de la matrice A et du vecteur b1
    H = Φ_train_block.T @ Φ_train_block
    alpha = calculate_alpha(H, kappa_l)
    A = H + alpha * np.eye(H.shape[0])
    b1 = Φ_train_block.T @ U_train_vector
    
    # Assemblage du système linéaire augmenté
    zero_block = np.zeros((D_divergence.shape[0], D_divergence.shape[0]))
    KKT_matrix = np.block([
        [A, D_divergence.T],
        [D_divergence, zero_block]
    ])
    
    rhs = np.concatenate([b1, np.zeros(D_divergence.shape[0])])
    
    # Résolution du système linéaire
    solution = np.linalg.solve(KKT_matrix, rhs)
    W_full = solution[:A.shape[0]]
    λ = solution[A.shape[0]:]
    
    # Reshape W into (n_collocation, 3)
    W = W_full.reshape((3, -1)).T
    
    return W

def build_pum_rbf_model_mpi(X_train, U_train, epsilon, patch_centers, patch_radius, kappa_l, halton_points):
    # Diviser les patches entre les processus MPI
    num_patches = len(patch_centers)
    patches_per_process = num_patches // size
    remainder = num_patches % size

    if rank < remainder:
        start_idx = rank * (patches_per_process + 1)
        end_idx = start_idx + patches_per_process + 1
    else:
        start_idx = rank * patches_per_process + remainder
        end_idx = start_idx + patches_per_process

    local_patch_indices = range(start_idx, end_idx)

    # Préparer les structures locales
    W_local = []
    centers_local = []
    X_collocation_local = []

    # Utiliser tqdm pour le suivi de l'avancement
    for idx in tqdm(local_patch_indices, desc=f"Process {rank+1}/{size} - Building model", position=rank):
        center = patch_centers[idx]
        # Récupérer les points d'entraînement dans le patch
        X_train_patch, U_train_patch = get_patch_data(X_train, U_train, center, patch_radius)
        # Récupérer les points de collocation dans le patch
        X_collocation_patch = get_collocation_points_in_patch(halton_points, center, patch_radius)
        if X_train_patch.shape[0] == 0 or X_collocation_patch.shape[0] == 0:
            continue  # Passer si pas de données dans le patch
        # Calculer les poids locaux W
        #W_patch = rbf_regression_divergence_free(X_collocation_patch, X_train_patch, U_train_patch, epsilon, kappa_l)
        n_constraints = int(0.8 * X_collocation_patch.shape[0])  # Utiliser 20% des points de collocation
        W_patch = rbf_regression_divergence_free(X_collocation_patch, X_train_patch, U_train_patch, epsilon, kappa_l, n_constraints)
        W_local.append(W_patch)
        centers_local.append(center)
        X_collocation_local.append(X_collocation_patch)

    # Rassembler les résultats de tous les processus
    W_total = comm.gather(W_local, root=0)
    centers_total = comm.gather(centers_local, root=0)
    X_collocation_total = comm.gather(X_collocation_local, root=0)

    if rank == 0:
        # Fusionner les résultats
        W_total = [w for sublist in W_total for w in sublist]
        centers_total = [c for sublist in centers_total for c in sublist]
        X_collocation_total = [x for sublist in X_collocation_total for x in sublist]

        model = {
            'W': W_total,
            'centers': centers_total,
            'X_collocation': X_collocation_total,
            'patch_radius': patch_radius,
            'epsilon': epsilon
        }
        return model
    else:
        return None

def evaluate_pum_rbf_model_mpi(model, X_eval):
    if model is None:
        return None, None, None, None

    W_total = model['W']
    centers_total = model['centers']
    X_collocation_total = model['X_collocation']
    patch_radius = model['patch_radius']
    epsilon = model['epsilon']

    # Distribute evaluation points among MPI processes
    num_points = X_eval.shape[0]
    points_per_process = num_points // size
    remainder = num_points % size

    if rank < remainder:
        start_idx = rank * (points_per_process + 1)
        end_idx = start_idx + points_per_process + 1
    else:
        start_idx = rank * points_per_process + remainder
        end_idx = start_idx + points_per_process

    X_eval_local = X_eval[start_idx:end_idx]

    # Initialize local results
    U_eval_local = np.zeros((X_eval_local.shape[0], 3))
    dUdx_local = np.zeros((X_eval_local.shape[0], 3))
    dUdy_local = np.zeros((X_eval_local.shape[0], 3))
    dUdz_local = np.zeros((X_eval_local.shape[0], 3))

    # Initialize weight sum for normalization
    weight_sum_local = np.zeros((X_eval_local.shape[0], 1))

    # Loop over all patches
    for idx_patch, center in enumerate(tqdm(centers_total, desc=f"Process {rank+1}/{size} - Evaluating model", position=rank)):
        # Find evaluation points within the patch
        distances = np.linalg.norm(X_eval_local - center, axis=1)
        idx_eval = np.where(distances <= patch_radius)[0]
        if idx_eval.size == 0:
            continue
        X_eval_patch = X_eval_local[idx_eval]

        # Compute partition of unity weights (Wendland C2 function)
        r = distances[idx_eval] / patch_radius
        psi_m_eval = wendland_c2(r).reshape(-1, 1)  # (n_eval_patch, 1)

        # Accumulate weight sum for normalization
        weight_sum_local[idx_eval] += psi_m_eval

        # Retrieve weights and collocation points for the patch
        W = W_total[idx_patch]
        X_collocation = X_collocation_total[idx_patch]

        # Compute RBF values and derivatives at evaluation points
        Φ_eval = gaussian_rbf(X_eval_patch, X_collocation, epsilon)
        dΦdx_eval = gaussian_rbf_derivative_first(X_eval_patch, X_collocation, epsilon, axis=0)
        dΦdy_eval = gaussian_rbf_derivative_first(X_eval_patch, X_collocation, epsilon, axis=1)
        dΦdz_eval = gaussian_rbf_derivative_first(X_eval_patch, X_collocation, epsilon, axis=2)

        # Apply partition of unity weights (note that Ω_m(x) = ψ_m(x) / sum_q ψ_q(x))
        Φ_eval_weighted = Φ_eval * psi_m_eval
        dΦdx_eval_weighted = dΦdx_eval * psi_m_eval
        dΦdy_eval_weighted = dΦdy_eval * psi_m_eval
        dΦdz_eval_weighted = dΦdz_eval * psi_m_eval

        # Compute evaluations
        U_eval_patch = Φ_eval_weighted @ W
        dUdx_patch = dΦdx_eval_weighted @ W
        dUdy_patch = dΦdy_eval_weighted @ W
        dUdz_patch = dΦdz_eval_weighted @ W

        # Accumulate results
        U_eval_local[idx_eval] += U_eval_patch
        dUdx_local[idx_eval] += dUdx_patch
        dUdy_local[idx_eval] += dUdy_patch
        dUdz_local[idx_eval] += dUdz_patch

    # Normalize accumulated results by the sum of weights
    # Avoid division by zero
    weight_sum_local[weight_sum_local == 0] = 1.0

    U_eval_local /= weight_sum_local
    dUdx_local /= weight_sum_local
    dUdy_local /= weight_sum_local
    dUdz_local /= weight_sum_local

    return U_eval_local, dUdx_local, dUdy_local, dUdz_local

def get_patch_data(X_train, U_train, center, patch_radius):
    idx_in_patch = np.where(np.linalg.norm(X_train - center, axis=1) <= patch_radius)[0]
    X_train_patch = X_train[idx_in_patch]
    U_train_patch = U_train[idx_in_patch]
    return X_train_patch, U_train_patch

def get_collocation_points_in_patch(X_collocation, center, patch_radius):
    idx_in_patch = np.where(np.linalg.norm(X_collocation - center, axis=1) <= patch_radius)[0]
    X_collocation_patch = X_collocation[idx_in_patch]
    return X_collocation_patch

def generate_patch_centers(bounds, dxp, dyp, dzp):
    x_centers = np.arange(bounds[0][0], bounds[0][1], dxp)
    y_centers = np.arange(bounds[1][0], bounds[1][1], dyp)
    z_centers = np.arange(bounds[2][0], bounds[2][1], dzp)
    centers = np.array(np.meshgrid(x_centers, y_centers, z_centers)).T.reshape(-1, 3)
    return centers

def generate_halton_points(n_points, bounds):
    sampler = qmc.Halton(d=3, scramble=False)
    halton_points = sampler.random(n=n_points)
    # Mise à l'échelle des points Halton pour correspondre aux limites
    for i in range(3):
        halton_points[:, i] = halton_points[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    return halton_points


def prepare_eval_domain(X_train, cylinder_centers, a, b, R, dx, dy, dz):
    """
    Prépare la grille d'évaluation en supprimant les points à l'extérieur des cylindres.
    
    Paramètres :
    - X_train : numpy array des points d'entraînement
    - cylinder_centers : liste de tuples des centres des cylindres
    - a, b : non utilisés ici, mais conservés pour le futur
    - R : rayon des cylindres
    - dx, dy, dz : pas de la grille en x, y, z

    Retourne :
    - X_eval_filtered : numpy array de la grille filtrée sans les points à l'extérieur des cylindres
    """
    # Définir la taille du domaine
    Lx = X_train[:, 0].max() - X_train[:, 0].min()
    Ly = X_train[:, 1].max() - X_train[:, 1].min()
    Lz = X_train[:, 2].max() - X_train[:, 2].min()

    Nx, Ny, Nz = int(Lx/dx), int(Ly/dy), int(Lz/dz)
    print(f'Nx = {Nx}, Ny = {Ny}, Nz = {Nz}')

    # Créer la grille en x, y, z
    x = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), Nx)
    y = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), Ny)
    z = np.linspace(X_train[:, 2].min(), X_train[:, 2].max(), Nz)
    X_eval = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    # Appliquer le masque pour garder uniquement les points à l'intérieur du domaine
    mask = mask_outside(X_eval, cylinder_centers, R, a, b)
    
    # Utiliser le masque pour filtrer les points à l'extérieur
    X_eval_filtered = X_eval[~mask]  # Filtrer les points où mask est False (donc à l'intérieur)

    return X_eval_filtered

def prepare_eval_domain_simple(X_train, a, b, c, dx, dy, dz):
    # Préparer le domaine d'évaluation
    x_min, x_max = a
    y_min, y_max = b
    z_min, z_max = c
    x_vals = np.arange(x_min, x_max, dx)
    y_vals = np.arange(y_min, y_max, dy)
    z_vals = np.arange(z_min, z_max, dz)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    X_eval = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return X_eval

"""
    Autre fonctions
"""

def mask_outside(X_eval, cylinder_centers, R, a, b):
    Y = X_eval[:, 1]
    Z = X_eval[:, 2]
    
    cylinders_mask = is_inside_cylinders(Y, Z, cylinder_centers, R) # Masquue des cylindres
    
    # Masque gauche
    y1, z1 = a[0], a[1]  # Premier point de la droite
    y2, z2 = b[0], b[1]  # Deuxième point de la droite
    m = (z2 - z1) / (y2 - y1)
    b = z1 - m * y1
    mask_left_of_line = Z < m * Y + b  # Masque pour les points sous la droite
    
    mask = mask_left_of_line | cylinders_mask  # Masque total
    
    return(mask)


def is_inside_cylinders(Y, Z, cylinder_centers, R):
    """
    Vérifie si les points (Y, Z) sont à l'intérieur de l'un des cylindres.
    
    Paramètres :
    - Y : numpy array des coordonnées y des points
    - Z : numpy array des coordonnées z des points
    - cylinder_centers : liste de tuples (xc, yc) des centres des cylindres
    - R : rayon des cylindres
    
    Retourne :
    - mask : numpy array de booléens, True si le point est à l'intérieur d'un cylindre
    """
    mask = np.zeros(len(Y), dtype=bool)
    for xc, yc in cylinder_centers:
        dist = np.sqrt((Y - xc)**2 + (Z - yc)**2)
        mask |= dist <= R
    return mask




def plot_reconstructed_field(U_eval, X_eval, x_target=None):
    if x_target is not None:
        # Visualiser le champ reconstruit à un plan spécifique
        tolerance = (X_eval[:, 0].max() - X_eval[:, 0].min()) / 50
        indices_plane = np.abs(X_eval[:, 0] - x_target) < tolerance
        X_plane = X_eval[indices_plane]
        U_plane = U_eval[indices_plane]
        
        # Création d'une grille pour la visualisation
        y_vals = X_plane[:, 1]
        z_vals = X_plane[:, 2]
        U_magnitude = np.linalg.norm(U_plane, axis=1)
        
        y_unique = np.unique(y_vals)
        z_unique = np.unique(z_vals)
        Y, Z = np.meshgrid(y_unique, z_unique)
        
        U_grid = np.zeros_like(Y)
        
        for i in range(len(y_unique)):
            for j in range(len(z_unique)):
                idx = np.where((y_vals == y_unique[i]) & (z_vals == z_unique[j]))
                if len(idx[0]) > 0:
                    U_grid[j, i] = U_magnitude[idx[0][0]]
        
        # Affichage du champ reconstruit
        plt.figure(figsize=(8, 6))
        plt.contourf(Y, Z, U_grid, cmap='turbo', levels=100, vmax = 0.15)
        plt.title(f'Reconstructed velocity field at x = {x_target}')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.colorbar()
        plt.show()
    else:
        # Visualisation 3D du champ reconstruit
        from mpl_toolkits.mplot3d import Axes3D
        U_magnitude = np.linalg.norm(U_eval, axis=1)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(X_eval[:, 0], X_eval[:, 1], X_eval[:, 2], c=U_magnitude, cmap='turbo', s=1, vmax = 0.15)
        fig.colorbar(p)
        ax.set_title('Reconstructed velocity field')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()



"""
    Fonctions d'extraction, sampling
"""
def extract_slice(U_eval, X_eval, normal, point, tolerance=None):
    """
    Extrait une slice du champ de vitesse U_eval et des coordonnées X_eval suivant une normale.
    
    U_eval : Champ de vitesse (N, 3)
    X_eval : Coordonnées spatiales (N, 3)
    normal : Normale du plan (string: 'x', 'y', 'z')
    point : Coordonnée sur la normale où prendre la coupe
    tolerance : Intervalle autour du point pour la sélection (facultatif). Par défaut, un intervalle basé sur les dimensions du domaine est utilisé.
    
    Retourne les coordonnées 2D (y, z) et les valeurs du champ de vitesse correspondantes dans la slice.
    """
    # Déterminer l'indice de la normale
    normal_index = {'x': 0, 'y': 1, 'z': 2}[normal]
    
    # Si aucun intervalle de tolérance n'est donné, on le définit automatiquement
    if tolerance is None:
        tolerance = (X_eval[:, normal_index].max() - X_eval[:, normal_index].min()) / 50
    
    # Sélectionner les points proches de la coordonnée spécifiée
    indices_slice = np.abs(X_eval[:, normal_index] - point) < tolerance
    
    # Extraire les points et le champ de vitesse dans cette slice
    slice_coords = X_eval[indices_slice]
    slice_U = U_eval[indices_slice]
    
    # Retourner les coordonnées restantes (par exemple, si normale est x, retourner y et z)
    if normal == 'x':
        slice_2d_coords = slice_coords[:, [1, 2]]  # Retourne y et z
    elif normal == 'y':
        slice_2d_coords = slice_coords[:, [0, 2]]  # Retourne x et z
    else:
        slice_2d_coords = slice_coords[:, [0, 1]]  # Retourne x et y
    
    return slice_2d_coords, slice_U



def interpolate_along_line(X_eval, U_eval, point1, point2, n_points=100):
    """
    Interpoler le champ U_eval le long d'une ligne entre point1 et point2.
    
    Args:
    - X_eval : Points d'évaluation (N, 3)
    - U_eval : Champ de vitesse (N, 3)
    - point1 : Premier point (x, y, z)
    - point2 : Second point (x, y, z)
    - n_points : Nombre de points sur la ligne
    
    Returns:
    - distances : Distances interpolées le long de la ligne
    - U_profile : Champ de vitesse interpolé le long de la ligne
    """
    # Générer des points le long de la ligne
    line_points = np.linspace(point1, point2, n_points)
    
    # Calculer la distance entre chaque point le long de la ligne
    distances = np.linalg.norm(line_points - point1, axis=1)
    
    # Trouver les indices des points d'évaluation les plus proches
    indices_nearest = np.argmin(np.linalg.norm(X_eval[:, None] - line_points[None, :], axis=2), axis=0)
    
    # Extraire le champ de vitesse aux points les plus proches
    U_profile = U_eval[indices_nearest]
    
    return distances, U_profile

