import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import *
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
from scipy.optimize import least_squares
from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from fractions import Fraction
from scipy.signal import welch
from tqdm import tqdm


def read_data(fichier):
    with open(fichier, 'r') as file:
        lines = file.readlines()

    data_list = []
    solution_time = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("ZONE"):
            # Extraire SOLUTIONTIME
            solution_time = None
            for j in range(i, min(i + 10, len(lines))):
                match = re.search(r'SOLUTIONTIME\s*=\s*(\d*\.?\d+)', lines[j])
                if match:
                    solution_time = float(match.group(1))
                    break

            # Vérification si le début du timestep a été trouvé
            if solution_time is None:
                print(f"SOLUTIONTIME not found after ZONE at line {i}")
                i += 1
                continue  # Passer à la prochaine ZONE si SOLUTIONTIME n'est pas trouvé

            # Recherche du début d'un timestep
            data_start_index = i + 6  # Les données commencent 6 lignes après la ligne "ZONE"
            print(f"Reading data starting at line {data_start_index} for solution_time {solution_time}")

            # Lire les données jusqu'à la prochaine entête ou la fin du fichier
            j = data_start_index
            while j < len(lines) and not lines[j].startswith("ZONE"):
                if lines[j].strip():  # Ignore les lignes vides
                    data_list.append([solution_time] + lines[j].split())
                j += 1
            i = j  # Passer à la prochaine ZONE
        else:
            i += 1

    if not data_list:
        print("No data found.")

    # Créer un DataFrame à partir de la liste de données
    columns = ['instant', 'x', 'y', 'z', 'I', 'Vx', 'Vy', 'Vz', '|V|', 'trackID', 
               'Ax', 'Ay', 'Az', '|A|', 'UncertaintyX', 'UncertaintyY', 'UncertaintyZ', 
               'UncertaintyVx', 'UncertaintyVy', 'UncertaintyVz', 'UncertaintyV', 
               'UncertaintyAx', 'UncertaintyAy', 'UncertaintyAz', 'UncertaintyA']
    data = pd.DataFrame(data_list, columns=columns)

    if data.empty:
        print("DataFrame is empty after reading data.")
        return data  # Retourner le DataFrame vide

    # Convertir les colonnes numériques
    for col in columns[1:]:  # Sauf 'instant' qui est déjà un float
        data[col] = pd.to_numeric(data[col])

    # Ajouter les vecteurs
    data['x'] = data['x'] + abs(np.min(data['x']))
    data['y'] = data['y'] + abs(np.min(data['y']))
    
    # Changement de variables
    x, y = data['x'], data['y']
    Vx, Vy = data['Vx'], data['Vy']
    data['x'], data['y'] = y, x
    data['Vx'], data['Vy'] = Vy, Vx

    # Vérification des dimensions
    unique_x = data['x'].unique()
    unique_y = data['y'].unique()
    unique_z = data['z'].unique()
    print(f"Unique x values: {len(unique_x)}, Unique y values: {len(unique_y)}, Unique z values: {len(unique_z)}")

    # Sélectionner les colonnes pertinentes pour le résultat final
    #final_columns = ['trackID', 'instant', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'UncertaintyX', 'UncertaintyY', 'UncertaintyZ', 'UncertaintyVx', 'UncertaintyVy', 'UncertaintyVz', 'position']
    final_columns = columns
    return data[final_columns]



def read_data_optimized(fichier):
    data_list = []
    solution_time = None
    
    # Ouvrir le fichier pour calculer le nombre total de lignes
    with open(fichier, 'r') as file:
        total_lines = sum(1 for _ in file)  # Compter les lignes pour tqdm
    
    # Réouvrir le fichier pour la lecture avec une barre de progression
    with open(fichier, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Lecture du fichier"):
            line = line.strip()

            # Chercher les zones dans le fichier
            if line.startswith("ZONE"):
                solution_time = None

                # Lire les 10 lignes suivantes pour trouver SOLUTIONTIME
                for _ in range(10):
                    line = next(file).strip()  # Lire la ligne suivante
                    match = re.search(r'SOLUTIONTIME\s*=\s*(\d*\.?\d+)', line)
                    if match:
                        solution_time = float(match.group(1))
                        break

                if solution_time is None:
                    print(f"SOLUTIONTIME not found.")
                    continue

                # Lire les données 6 lignes après "ZONE"
                for _ in range(6):
                    next(file)  # Sauter les 6 lignes suivantes

                # Lire jusqu'à la prochaine entête "ZONE"
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("ZONE"):
                        break  # Fin des données ou début de la prochaine ZONE
                    
                    # Ajouter les données à la liste (solution_time + valeurs numériques de la ligne)
                    data_list.append([solution_time] + line.split())

    # Vérification si des données ont été trouvées
    if not data_list:
        print("No data found.")
        return pd.DataFrame()  # Retourner un DataFrame vide

    # Créer le DataFrame avec les colonnes appropriées
    columns = ['instant', 'x', 'y', 'z', 'I', 'Vx', 'Vy', 'Vz', '|V|', 'trackID', 
               'Ax', 'Ay', 'Az', '|A|', 'UncertaintyX', 'UncertaintyY', 'UncertaintyZ', 
               'UncertaintyVx', 'UncertaintyVy', 'UncertaintyVz', 'UncertaintyV', 
               'UncertaintyAx', 'UncertaintyAy', 'UncertaintyAz', 'UncertaintyA']
    
    data = pd.DataFrame(data_list, columns=columns)

    # Convertir les colonnes numériques
    for col in columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convertir avec gestion des erreurs

    # Retraitement des données : Ajout des vecteurs et transformation des coordonnées
    data['x'] = data['x'] + abs(np.min(data['x']))
    data['y'] = data['y'] + abs(np.min(data['y']))

    # Inversion des coordonnées x et y, Vx et Vy
    data[['x', 'y']] = data[['y', 'x']]
    data[['Vx', 'Vy']] = data[['Vy', 'Vx']]



    # Vérification des dimensions
    unique_x = data['x'].unique()
    unique_y = data['y'].unique()
    unique_z = data['z'].unique()
    print(f"Unique x values: {len(unique_x)}, Unique y values: {len(unique_y)}, Unique z values: {len(unique_z)}")

    return data[columns]

def read_case(case, y_offset, z_offset, p, R, dimension='2D'):
    # Lecture et traitement des données
    data = read_data_optimized(case)

    # Si les données sont 2D, projection dans le plan YZ
    if dimension == '2D':
        data = project_to_yz_with_fields(data)

    # Application des offsets
    data['x'] += data['x'].min()  # Min une seule fois
    data['z'] += z_offset
    data['y'] += y_offset

    # Calcul des centres des cylindres avec une approche vectorisée
    ix, iy = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    xc = 26.125 + ix * p
    yc = 61.875 + iy * p
    yc[0, :] = 61.875# Ajustement de yc pour la première rangée
    cylinder_centers = np.vstack([xc.ravel(), yc.ravel()]).T


    # Suppression des colonnes inutiles et nettoyage des données
    data.drop(columns=['instant', 'Track ID'], errors='ignore', inplace=True)
    data.dropna(subset=['x', 'y', 'z'], inplace=True)
    data = data[np.isfinite(data['x']) & np.isfinite(data['y']) & np.isfinite(data['z'])]

    return data, cylinder_centers

"""
    Filter, sampling and track functions
"""

def project_to_yz_with_fields(initial_data):
    data = initial_data.copy()
    data['x'] = 0
    return data

def project_to_xz_with_fields(initial_data):
    data = initial_data.copy()
    data['y'] = 0
    return data


def is_inside_cylinder(x, z, centers, R):
    for (xc, yc) in centers:
        if (x - xc)**2 + (z - yc)**2 < R**2 :
            return True
    return False

