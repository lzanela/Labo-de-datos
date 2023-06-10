"""

Trabajo Práctico Nº2

Materia Laboratorio de Datos, FCEyN UBA
1er Cuatrimestre 2023

Grupo: Pythonisos

Integrantes: Nicolás Rozenberg, Joaquín Viera, Luca Zanela. 

Nombre del archivo: utils.py
Descripción: Funciones auxiliares utilizadas para el análisis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED=42

import numpy as np
from itertools import product

def kfold_cross_validation(X, y, model, param_grid, k=5):
    rng = np.random.default_rng(RANDOM_SEED)
    n_samples = len(X)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_size = n_samples // k
    scores = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size

        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        best_score = 0
        best_params = {}

        for params in product(*param_grid.values()):
            param_combination = dict(zip(param_grid.keys(), params))
            model.set_params(**param_combination)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)

            if score > best_score:
                best_score = score
                best_params = param_combination

        scores.append(best_score)

    return scores, best_params

def plot_digit(df, index):
    digit = df.drop("class", axis=1).iloc[index, :].to_numpy()
    digit_image = digit.reshape(28, 28)

    plt.imshow(digit_image)
    plt.axis('off')
    plt.show()
    plt.close()

def random_subset(df, column, k):
    subset = pd.DataFrame(columns=df.columns)
    classes = df[column].unique()
    
    for c in classes:
        class_rows = df[df[column] == c]
        if len(class_rows) > k:
            class_subset = class_rows.sample(
                k, 
                random_state=RANDOM_SEED
            )
        else:
            class_subset = class_rows
        subset = pd.concat([subset, class_subset])
    
    return subset
