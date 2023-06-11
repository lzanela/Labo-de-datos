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
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random

import utils

# Dejamos un seed fijo
RANDOM_SEED=42
rng = np.random.default_rng(RANDOM_SEED)

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

print("Media: ", counts.mean())
print("Desvío estándar: ", counts.std())
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

pca = PCA(n_components=4)
pca_data = pca.fit_transform(df.drop("class", axis=1))
pca_df = pd.DataFrame(data = pca_data, columns = ["PC1", "PC2", "PC3","PC4"])
pca_df["class"] = df["class"]

#%%----------------------------------------------------------------
# Graficamos un scatterplot comparando el PCA1 con el PCA2

sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class", palette="bright")
plt.show()
plt.close()

print(pca.explained_variance_ratio_[0:2])
print(f"Varianza explicada: {round(sum(pca.explained_variance_ratio_[0:2]), 2)*100}%")
#%%----------------------------------------------------------------
# Se ven en cierto punto solapados, sin embargo las componentes principales graficados
# solo explican el 17% de la varianza. A pesar de esto, algunos dígitos
# se distinguen de los demás, como el 0. Observemos qué áreas 
# son explican la mayor varianza.

# Obtenemos los componentes principales. Podemos pensarlas como
# direcciones en el espacio de atributos, que transforman la información
# de forma que expliquen la mayor varianza en la dimensión indicada
components = pca.components_


# Tomando valor absoluto en cada posicion de estos vectores las podemos
# pensar como pesos de cada feature en la componente principal respectiva.
# Eso se debe a que determina cuánto "empuja" cada feature a la dirección
# de una componente principal. 
pixel_weights = np.abs(components)
pixel_weights_image = pixel_weights.reshape(-1, 28, 28)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(pixel_weights_image[i], cmap='hot', vmin=0, vmax=np.max(pixel_weights_image[i]))
    ax.set_axis_off()
    ax.set_title(f"PCA {i+1}")

plt.tight_layout()
plt.show()

# Observamos que los píxeles que están en los bordes 
# tienen una influencia prácticamente nula con las 
# componentes principales, por lo que nos son de poco interés.
# También, notar que existe menor influencia arriba que abajo.
#%%----------------------------------------------------------------
# Obtenemos el dataset binario, y vemos si 
# las clases están balanceadas (como observamos arriba
# no deberían estarlo)

df_binary  = df[df["class"].isin([0, 1])]
df_binary.reset_index(drop=True, inplace=True)

classes = df_binary["class"]
counts = classes.value_counts()

counts.plot(kind="bar", xlabel="Clase", ylabel="Ocurrencias")
plt.show()
plt.close()

print(f"Diferencia de porcentaje entre ocurrencias de 0 y 1: {np.abs((counts[1]-counts[0])*100/len(df_binary))} %")
# Observamos un pequeño desbalanceo, donde la clase 1 posee un 6% más muestras que la clase 0
#%%----------------------------------------------------------------
# Comencemos eligiendo como primeros atributos para la
# la clasificación por kNN, las 3 componentes principales.

pca_binary = PCA(n_components=3)
pca_data_binary = pca_binary.fit_transform(df_binary.drop("class", axis=1))
pca_df_binary = pd.DataFrame(data = pca_data_binary, columns = ["PC1", "PC2", "PC3"])
pca_df_binary["class"] = df_binary["class"]
pca_df_binary.head()

#%%----------------------------------------------------------------
X=pca_df_binary.drop("class", axis=1)
Y=pca_df_binary["class"]

# Reescalamos features entre 0 y 1
utils.rescale_features(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify= Y) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
cm = metrics.confusion_matrix(Y_test, Y_pred)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
disp.plot()
print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))

# Tomamos a 0 como la clase Verdadera, ya que posee
# menor muestra.
print("Precisión del modelo: ", metrics.precision_score(Y_test, Y_pred, pos_label=0))
print("Sensitividad del modelo: ", metrics.recall_score(Y_test, Y_pred, pos_label=0))
print("F1 Score del modelo: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))

#%%----------------------------------------------------------------
# También podríamos elegir como features a los píxeles que más
# empujan en la dirección de cada componente principal.

pixels = []

for component in pca_binary.components_:
    pixel = np.argmax(np.abs(component))
    pixels.append(pixel)

print("Pixels", pixels)

positions = np.zeros(784)
for pixel in pixels:
    positions[pixel] = 1

# Graficamos las posiciones de los píxeles a analizar
plt.imshow(positions.reshape(28, 28), cmap="hot")
plt.show()
plt.close()

#%%----------------------------------------------------------------
# Los mismos son buenos candidatos, y se encuentran suficientemente separados.
# Entrenamos el modelo.

df_pixels = df_binary.loc[:, (str(pixel) for pixel in pixels)]
X=df_pixels
Y=df_binary["class"]

# Reescalamos features entre 0 y 1
utils.rescale_features(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify= Y) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
cm = metrics.confusion_matrix(Y_test, Y_pred)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
disp.plot()
print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
print("Precisión del modelo: ", metrics.precision_score(Y_test, Y_pred, pos_label=0))
print("Sensitividad del modelo: ", metrics.recall_score(Y_test, Y_pred, pos_label=0))
print("F1 Score del modelo: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))
#%%----------------------------------------------------------------
# Ahora bien, construiremos una grilla de 5 x 5, y seleccionaremos atributos
# aleatoriamente. Los tamaños de los subconjuntos irán desde 4 hasta 7

grid = []
for i in range(6, 23, 4):
    grid += list(range(6+28*i, 22+28*i, 4))


positions = np.zeros(784)
for pixel in grid:
    positions[pixel] = 1

# Graficamos las posiciones de los píxeles a analizar
plt.imshow(positions.reshape(28, 28), cmap="hot")
plt.show()
plt.close()

subsets = [list(rng.choice(grid, 4, replace=False))]

for i in range(3):
    last_subset = subsets[i]
    new_subset = [*last_subset, *rng.choice(list(set(grid)-set(last_subset)), 1, replace=False)]
    subsets.append(new_subset)

#%%----------------------------------------------------------------

for subset in subsets:
    print("Píxeles", subset)

    df_pixels_grid = df_binary.loc[:, (str(pixel) for pixel in subset)]
    X=df_pixels
    Y=df_binary["class"]

    # Reescalamos features entre 0 y 1
    utils.rescale_features(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify= Y) # 70% para train y 30% para test

    model = KNeighborsClassifier(n_neighbors = 5)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    print(f"Subconjunto de pixels: {subset}")

    cm = metrics.confusion_matrix(Y_test, Y_pred)

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=model.classes_)
    disp.plot()
    plt.show()
    plt.close()
    print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precisión del modelo: ", metrics.precision_score(Y_test, Y_pred, pos_label=0))
    print("Sensitividad del modelo: ", metrics.recall_score(Y_test, Y_pred, pos_label=0))
    print("F1 Score del modelo: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))
    print("---------------------------------")
#%%----------------------------------------------------------------
# k-Fold Cross Validation
# Probamos realizar Cross Validation para los Principal Components

k_hyperparam = [1, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for k in k_hyperparam:
    model = KNeighborsClassifier(n_neighbors = k)
    X=pca_df_binary.drop("class", axis=1)
    Y=pca_df_binary["class"]

    # Reescalamos features entre 0 y 1
    utils.rescale_features(X)
    accuracy_scores.append(
        utils.kfold_cross_validation(X, Y, model, score_metric=metrics.accuracy_score)
    )
    precision_scores.append(
        utils.kfold_cross_validation(X, Y, model, score_metric=metrics.precision_score, pos_label=0)
    )
    recall_scores.append(
        utils.kfold_cross_validation(X, Y, model, score_metric=metrics.recall_score, pos_label=0)
    )
    f1_scores.append(
        utils.kfold_cross_validation(X, Y, model, score_metric=metrics.f1_score, pos_label=0)
    )


#%%----------------------------------------------------------------
# Graficamos las métricas en función al hiperparámetro k.

plt.plot(k_hyperparam, accuracy_scores)
plt.xlabel("K")
plt.ylabel("Exactitud promedio")
plt.show()
plt.close()

plt.plot(k_hyperparam, precision_scores)
plt.xlabel("K")
plt.ylabel("Precisión promedio")
plt.show()
plt.close()

plt.plot(k_hyperparam, recall_scores)
plt.xlabel("K")
plt.ylabel("Recall promedio")
plt.show()
plt.close()

plt.plot(k_hyperparam, f1_scores)
plt.xlabel("K")
plt.ylabel("F1 score promedio")
plt.show()
plt.close()


# Observamos que el valor k=10 es el que mejor
# performa en todas las métricas

#%%----------------------------------------------------------------
# Modelo de Árbol de Decisión
#%%----------------------------------------------------------------
# Como observamos al principio, las clases no se encuentran del todo balanceadas.
# Considerando que la muestra es suficiente para la clasificación, realizaremos undersampling
# Calculamos la mediana de la cantidad de ocurrencias, y para todas las clases que
# posean una cantidad de registros mayor, nos quedamos con un subconjunto
# aleatorio.

classes = df["class"]
counts = classes.value_counts()
classes_median = counts.median()

df_undersampled = utils.random_subset(df, 'class', int(classes_median))
classes = df_undersampled["class"]
counts = classes.value_counts()*100/len(df_undersampled)
counts.plot(kind="bar", xlabel="Clase", ylabel="Ocurrencias")
plt.show()
plt.close()
df_undersampled.reset_index(inplace=True)
df_undersampled.drop("index", axis=1, inplace=True)
#%%----------------------------------------------------------------
# Observamos que la clase '5' posee un 0,75% menos de registros que 
# el que esperaríamos tener. 
# Eso lo podríamos tener en cuenta luego a la hora de analizar 
# los resultados.
# Ajustamos un clasificador por árbol de decisión
# para distintas profundidades, ya usando Cross Validation


# Comenzamos probando con los PCAs
max_depths = [3, 5, 7, 10, 15, 20, 30]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for max_depth in max_depths:
    dt_model = DecisionTreeClassifier(
        criterion = "entropy",
        max_depth=max_depth
    )
    X=pca_df.drop("class", axis=1)
    Y=pca_df["class"]

    accuracy_scores.append(
        utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.accuracy_score)
    )
    precision_scores.append(
        utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.precision_score, labels=Y.unique(), average="macro")
    )

#%%----------------------------------------------------------------
plt.plot(max_depths, accuracy_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Exactitud")
plt.show()
plt.close()

plt.plot(max_depths, precision_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Precisión promedio")
plt.show()
plt.close()

#%%----------------------------------------------------------------
# Realizamos lo mismo, pero con la grilla presentada
# previamente

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
df_pixels_grid = df_undersampled.loc[:, (str(pixel) for pixel in grid)]
X=df_pixels_grid
Y=df_undersampled["class"].astype('int')

for max_depth in max_depths:
    dt_model = DecisionTreeClassifier(
        criterion = "entropy",
        max_depth=max_depth
    )

    accuracy_scores.append(
        utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.accuracy_score)
    )
    precision_scores.append(
        utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.precision_score, labels=Y.unique(), average="macro")
    )

#%%----------------------------------------------------------------
plt.plot(max_depths, accuracy_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Exactitud")
plt.show()
plt.close()

plt.plot(max_depths, precision_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Precisión promedio")
plt.show()
plt.close()

#%%----------------------------------------------------------------

best_max_depth = max_depths[np.argmax(accuracy_scores)]
print(f"Profundidad óptima: {best_max_depth}")
#%%----------------------------------------------------------------
# %%
