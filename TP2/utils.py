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
