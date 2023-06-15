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
import time

import utils

# Dejamos un seed fijo
RANDOM_SEED=42
rng = np.random.default_rng(RANDOM_SEED)

#%%----------------------------------------------------------------

# Para correr ésta y las siguientes celdas, crear una nueva
# carpeta llamada 'data', y ubicar los archivos de datos.

df = pd.read_csv('./data/mnist_desarrollo.csv', header=None)
df.head()

df_test_binario = pd.read_csv('./data/mnist_test_binario.csv', header=None)
df_test = pd.read_csv('./data/mnist_test.csv', header=None)

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
df_test.columns = COLUMNS
df_test_binario.columns = COLUMNS

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
print("Cantidad de datos: ", len(df))
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
pixel_weights = np.abs(components) * np.vstack(pca.explained_variance_ratio_)
pixel_weights_image = pixel_weights.reshape(-1, 28, 28)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(pixel_weights_image[i], cmap='hot', vmin=0, vmax=np.max(pixel_weights_image))
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
len(df_binary)
len(df_binary[df_binary["class"] == 1])

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
print("Exactitud del modelo en test:", metrics.accuracy_score(Y_test, Y_pred))

# Tomamos a 0 como la clase Verdadera, ya que posee
# menor muestra.
print("F1 Score del modelo en test: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))

Y_pred = model.predict(X_train)

print("Exactitud del modelo en training:", metrics.accuracy_score(Y_train, Y_pred))
print("F1 Score del modelo en training: ", metrics.f1_score(Y_train, Y_pred, pos_label=0))
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
print("Exactitud del modelo en test:", metrics.accuracy_score(Y_test, Y_pred))
print("F1 Score del modelo en test: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))

Y_pred = model.predict(X_train)

print("Exactitud del modelo en training:", metrics.accuracy_score(Y_train, Y_pred))
print("F1 Score del modelo en training: ", metrics.f1_score(Y_train, Y_pred, pos_label=0))
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
# Guardo los conjuntos de datos pertenecientes
# a cada subconjunto de atributos en possible data,
# para luego compararlos
possible_data=[]
for subset in subsets:
    df_pixels_grid = df_binary.loc[:, (str(pixel) for pixel in subset)]
    X=df_pixels_grid
    possible_data.append(X)
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
    print("Exactitud del modelo en test:", metrics.accuracy_score(Y_test, Y_pred))
    print("F1 Score del modelo en test: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))
    
    Y_pred = model.predict(X_train)
    
    print("Exactitud del modelo en training:", metrics.accuracy_score(Y_train, Y_pred))
    print("F1 Score del modelo en training: ", metrics.f1_score(Y_train, Y_pred, pos_label=0))
    print("---------------------------------")
#%%----------------------------------------------------------------
# Probamos usar toda la grilla

df_pixels_total_grid = df_binary.loc[:, (str(pixel) for pixel in grid)]
X=df_pixels_total_grid
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
print("Exactitud del modelo en test:", metrics.accuracy_score(Y_test, Y_pred))
print("F1 Score del modelo en test: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))

Y_pred = model.predict(X_train)

print("Exactitud del modelo en training:", metrics.accuracy_score(Y_train, Y_pred))
print("F1 Score del modelo en training: ", metrics.f1_score(Y_train, Y_pred, pos_label=0))
print("---------------------------------")
#%%----------------------------------------------------------------
# k-Fold Cross Validation
# Probamos realizar Cross Validation para los Principal Components

k_hyperparam = [1, 3, 5, 10, 15, 20, 30, 50]
possible_data_dict = {f'{len(x.columns)} píxeles de la grilla': x for x in possible_data}
possible_data_dict = {"PCAs": pca_df_binary.drop("class", axis=1), "Píxeles influyentes": df_pixels, "Grilla completa": df_pixels_total_grid, **possible_data_dict}
best_k_data=[]


f1_scores_dict = {}
df_f1_scores = pd.DataFrame({"Conjunto": [], "value": [], "k": []})

for x in possible_data_dict.keys():
    f1_scores = []
    for k in k_hyperparam:
        model = KNeighborsClassifier(n_neighbors = k)
        X=possible_data_dict[x]
        Y=df_binary["class"]

        # Reescalamos features entre 0 y 1
        utils.rescale_features(X)
        score = utils.kfold_cross_validation(X, Y, model, score_metric=metrics.f1_score, pos_label=0)
        f1_scores.append(score)
        df_f1_scores = df_f1_scores.append({'Conjunto': x, 'value': score, 'k': k}, ignore_index=True)
    # Graficamos las métricas en función al hiperparámetro k
    # y su data
    f1_scores_dict[x] = f1_scores 
    
    max_score = max(f1_scores)
    best_k = k_hyperparam[np.argmax(f1_scores)]
    best_k_data.append((best_k, max_score, x))
    print(f"F1 score máximo: {max_score}")
    print(f"Profundidad óptima: {best_k}")
    print(x)


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x="k", y='value', hue='Conjunto', data=df_f1_scores, marker="o")
plt.xlabel("K")
plt.ylabel("F1 score")
plt.show()
plt.close()

print(max(best_k_data, key=lambda x: x[1]))

#%%----------------------------------------------------------------+

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
print("Registros perdidos", len(df)-len(df_undersampled))
#%%----------------------------------------------------------------
# Observamos que la clase '5' posee un 0,75% menos de registros que 
# el que esperaríamos tener. Sin embargo, consideramos que es despreciable
# debido a la cantidad de datos.
# Ajustamos un clasificador por árbol de decisión
# para distintas profundidades, ya usando Cross Validation

# Vamos guardando los scores en un DataFrame
df_scores = pd.DataFrame({"Conjunto": [], "value": [], "Maxima profundidad": []})

# Comenzamos probando con los PCAs
pca = PCA(n_components=4)
pca_data = pca.fit_transform(df.drop("class", axis=1))
pca_df = pd.DataFrame(data = pca_data, columns = ["PC1", "PC2", "PC3","PC4"])
pca_df["class"] = df["class"]

max_depths = [3, 5, 7, 10, 15, 20, 30]
accuracy_scores = []

for max_depth in max_depths:
    dt_model = DecisionTreeClassifier(
        criterion = "entropy",
        max_depth=max_depth
    )
    X=pca_df.drop("class", axis=1)
    Y=pca_df["class"]

    accuracy_score = utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.accuracy_score, k=10)
    accuracy_scores.append(accuracy_score)
    df_scores = df_scores.append({"Conjunto": "PCAs", "value": accuracy_score, "Maxima profundidad": max_depth}, ignore_index=True)

#%%----------------------------------------------------------------
plt.plot(max_depths, accuracy_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Exactitud")
plt.show()
plt.close()

max_accuracy = max(accuracy_scores)
best_max_depth = max_depths[np.argmax(accuracy_scores)]
print(f"Exactitud máxima: {max_accuracy}")
print(f"Profundidad óptima: {best_max_depth}")


#%%----------------------------------------------------------------
# Realizamos lo mismo, pero con la grilla presentada
# previamente

accuracy_scores = []
df_pixels_grid = df_undersampled.loc[:, (str(pixel) for pixel in grid)]
X=df_pixels_grid
Y=df_undersampled["class"].astype('int')

for max_depth in max_depths:
    dt_model = DecisionTreeClassifier(
        criterion = "entropy",
        max_depth=max_depth
    )

    accuracy_score = utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.accuracy_score, k=10)
    accuracy_scores.append(accuracy_score)
    df_scores = df_scores.append({"Conjunto": "Grilla completa", "value": accuracy_score, "Maxima profundidad": max_depth}, ignore_index=True)

#%%----------------------------------------------------------------
plt.plot(max_depths, accuracy_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Exactitud")
plt.show()
plt.close()

max_accuracy = max(accuracy_scores)
best_max_depth = max_depths[np.argmax(accuracy_scores)]
print(f"Exactitud máxima: {max_accuracy}")
print(f"Profundidad óptima: {best_max_depth}")

#%%----------------------------------------------------------------
# Nos fijamos si los pixeles que más se usan tienen mejor rendimiento a la hora
# de entrenar el arbol
pixeles=list(range(0,784))
strings=[str(x) for x in pixeles]

promedios=list(df_undersampled[strings].mean())

reshape_promedios = []
for i in range(0, len(promedios), 28):
    reshape_promedios.append(promedios[i:i+28])
plt.imshow(reshape_promedios, cmap="hot")
plt.show()
plt.close()


high_percentile = np.percentile(promedios, 80)
promedios_altos=[x if x>high_percentile else 0 for x in promedios]

reshape_promedios = []
for i in range(0, len(promedios_altos), 28):
    reshape_promedios.append(promedios_altos[i:i+28])
plt.imshow(reshape_promedios, cmap="hot")
plt.show()
plt.close()

#Me fijo cuales pixles tienen promedio alto
pixeles_altos=[]
for i in range(0,784):
    if promedios_altos[i] != 0:
      pixeles_altos.append(i)

#%%----------------------------------------------------------------
# Pruebo el modelo
accuracy_scores = []
df_pixels_high = df_undersampled.loc[:, (str(pixel) for pixel in pixeles_altos)]
X=df_pixels_high
Y=df_undersampled["class"].astype('int')

for max_depth in max_depths:
    dt_model = DecisionTreeClassifier(
        criterion = "entropy",
        max_depth=max_depth
    )

    accuracy_score = utils.kfold_cross_validation(X, Y, dt_model, score_metric=metrics.accuracy_score, k=10)
    accuracy_scores.append(accuracy_score)
    df_scores = df_scores.append({"Conjunto": "Píxeles con promedio más alto", "value": accuracy_score, "Maxima profundidad": max_depth}, ignore_index=True)

#%%----------------------------------------------------------------
plt.plot(max_depths, accuracy_scores)
plt.xlabel("Profundidad máxima")
plt.ylabel("Exactitud")
plt.show()
plt.close()

max_accuracy = max(accuracy_scores)
best_max_depth = max_depths[np.argmax(accuracy_scores)]
print(f"Exactitud máxima: {max_accuracy}")
print(f"Profundidad óptima: {best_max_depth}")
#%%----------------------------------------------------------------
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x="Maxima profundidad", y='value', hue='Conjunto', data=df_scores, marker="o")
plt.xlabel("Maxima profundidad")
plt.ylabel("Exactitud")
plt.show()
plt.close()

#%%----------------------------------------------------------------

# Comparación de tiempo de ejecución, utilizando la grilla

X=df_pixels_total_grid
Y=df_binary["class"]

# Reescalamos features entre 0 y 1
utils.rescale_features(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify= Y) # 70% para train y 30% para test

start = time.time()

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

end = time.time()

print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
print("Tiempo de entrenamiento y predicción para el kNN: ", end - start)

start = time.time()

model = DecisionTreeClassifier(max_depth = 5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

end = time.time()

print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
print("Tiempo de entrenamiento y predicción para el Árbol de decisión: ", end - start)

#%%----------------------------------------------------------------
# Modelos definitivos

# Seleccionamos los modelos que mejor rendimiento obtuvieron,
# tanto para el caso binario como para el caso completo

#%%
# Para dataset binario, elegimos kNN con k=5 vecinos entrenado a partir 
# de la grilla de la figura 7

df_pixels_total_grid = df_binary.loc[:, (str(pixel) for pixel in grid)]
X=df_pixels_total_grid
Y=df_binary["class"]

# Reescalamos features entre 0 y 1 y evaluamos rendimiento sobre el conjunto de test provisto

utils.rescale_features(X)
X_train, X_test, Y_train, Y_test = X, df_test_binario.drop("class", axis=1).loc[:, (str(pixel) for pixel in grid)], Y, df_test_binario["class"]

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Exactitud del modelo en test:", metrics.accuracy_score(Y_test, Y_pred))
print("F1 Score del modelo en test: ", metrics.f1_score(Y_test, Y_pred, pos_label=0))

Y_pred = model.predict(X_train)

print("Exactitud del modelo en training:", metrics.accuracy_score(Y_train, Y_pred))
print("F1 Score del modelo en training: ", metrics.f1_score(Y_train, Y_pred, pos_label=0))
print("---------------------------------")

#%%
# Para dataset completo, elegimos el árbol de decisión con profundidad máxima 
# k=15 entrenado a partir del conjunto de píxeles con promedio de intensidad por 
# encima del percentil 80, mostrado en la figura 9.

df_pixels_high = df.loc[:, (str(pixel) for pixel in pixeles_altos)]
X=df_pixels_high
Y=df["class"].astype('int')


dt_model = DecisionTreeClassifier(criterion = "entropy", max_depth=15)
X_train, X_test, Y_train, Y_test = X, df_test.drop("class", axis=1).loc[:, (str(pixel) for pixel in pixeles_altos)], Y, df_test["class"]

dt_model.fit(X_train, Y_train)
Y_pred = dt_model.predict(X_test)

print("Exactitud del modelo en test:", metrics.accuracy_score(Y_test, Y_pred))

Y_pred = dt_model.predict(X_train)

print("Exactitud del modelo en training:", metrics.accuracy_score(Y_train, Y_pred))

# %%
