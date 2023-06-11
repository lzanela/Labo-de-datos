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
import numpy as np
from itertools import product

RANDOM_SEED=42



def kfold_cross_validation(X, y, model, score_metric, k=5):
    rng = np.random.default_rng(RANDOM_SEED)
    n_samples = len(X)
    indices = np.arange(n_samples)

    # Reordenamos los índices aleatoriamente.
    rng.shuffle(indices)

    fold_size = n_samples // k
    scores = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size

        # Datos de testing en el fold
        val_indices = indices[start:end]

        #Datos de training
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train, y_train = X.iloc[train_indices, :], y[train_indices]
        X_val, y_val = X.iloc[val_indices, :], y[val_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = score_metric(y_pred, y_val)

        scores.append(score)

    return np.mean(scores)


def rescale_features(X: pd.DataFrame):
    for feature in X.columns:
        min_value = np.min(X[feature])
        max_value = np.max(X[feature])
        X[feature] = X[feature].apply(
            lambda x: (x - min_value) / (max_value - min_value)
        )
    


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
