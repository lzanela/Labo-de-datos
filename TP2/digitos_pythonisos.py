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
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import utils

# Dejamos un seed fijo
RANDOM_SEED=42

#%%----------------------------------------------------------------

df = pd.read_csv('./data/mnist_desarrollo.csv', header=None)
df.head()
#%%----------------------------------------------------------------
# Podemos deducir que la primera columna corresponde a las clases
# (dígitos del 0 al 9). Renombramos las columnas 
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
# con el resto (principalmente el 1 y el 7, con el 5). 
# Las demás 784 columnas representan la matriz estirada de la imagen
# Cada imagen es, por tanto, de 28x28 píxeles 

np.random.seed(RANDOM_SEED)
random_index = np.random.randint(0, 1000)
utils.plot_digit(df, random_index)
print(f"Clase a representar: {df.iloc[random_index]['class']}")

#%%----------------------------------------------------------------
# Verificamos si los digitos son distinguibles usando reducción
# de dimensionalidad, para determinar si los datos nos permitirán
# entrenar los modelos de forma performante

pca = PCA()
pca.n_components = 2
pca_data = pca.fit_transform(df.drop("class", axis=1))
pca_df = pd.DataFrame(data = pca_data, columns = ["PC1", "PC2"])
pca_df["class"] = df["class"]

#%%----------------------------------------------------------------
# Graficamos

sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class")
plt.show()
plt.close()

print(pca.explained_variance_ratio_)
print(f"Varianza explicada: {round(sum(pca.explained_variance_ratio_), 2)*100}%")
#%%----------------------------------------------------------------
# Se ven algo solapados, sin embargo las componentes principales
# solo explican el 17% de la varianza. E igualmente algunos dígitos
# se distinguen de los demás, como el 0.
#%%----------------------------------------------------------------
# Obtenemos el dataset binario


df_binary  = df[df["class"].isin([0, 1])]
#%%----------------------------------------------------------------


# ACLARACION: ESTO VA PARA EL FINAL.
# Calculamos la mediana de la cantidad de ocurrencias, y para todas las clases que
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
#%%----------------------------------------------------------------

# %%
