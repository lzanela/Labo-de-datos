"""

Trabajo Práctico Nº2

Materia Laboratorio de Datos, FCEyN UBA
1er Cuatrimestre 2023

Grupo: Pythonisos

Integrantes: Nicolás Rozenberg, Joaquín Viera, Luca Zanela. 

Nombre del archivo: digitos_pythonisos.py

"""

#%%----------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import utils

# Dejamos un seed fijo
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)

#%%----------------------------------------------------------------

df = pd.read_csv('./data/mnist_desarrollo.csv', header=None)
df.head()
#%%----------------------------------------------------------------
# Podemos deducir que la primera columna corresponde a las clases
# (dígitos del 1 al 9). Renombramos las columnas 
# para que la primera se llame 'classes', y todas las demás 
# les asignamos un número según el orden.

COLUMNS = [
    "class",
    *[str(i-1) for i in range(1, len(df.columns))]
]
df.columns = COLUMNS

#%%----------------------------------------------------------------
# Evaluemos si el dataset provisto
# posee balanceadas las clases.

classes = df["class"]
counts = classes.value_counts()

counts.plot(kind="bar", xlabel="Clase", ylabel="Ocurrencias")
plt.show()
plt.close()
#%%----------------------------------------------------------------
# Observamos que hay ciertas clases que están desbalanceadas
# con el resto (principalmente el 1 y el 7, con el 5). Calculamos
# la mediana de la cantidad de ocurrencias, y para todas las clases que
# posean una cantidad de registros mayor, nos quedamos con un subconjunto
# aleatorio.

classes_median = counts.median()

df = utils.random_subset(df, 'class', int(classes_median))

classes = df["class"]
classes.value_counts()*100/len(df)
#%%----------------------------------------------------------------
# Observamos que la clase '5' posee un 0,75% menos de registros que 
# el que esperaríamos tener. 
# Eso lo podríamos tener en cuenta luego a la hora de analizar 
# los resultados. Sin embargo, esperamos que se corregirá realizando
# ensamblaje de modelos.

utils.plot_digit(df, 15)
# %%
