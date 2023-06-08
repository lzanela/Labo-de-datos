import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

huevos = pd.read_csv("Muestra aleatoria huevos.csv")

huevos.head()

# Generamos los dataframes para las variables

x = pd.DataFrame(huevos["RU"])
y = pd.DataFrame(huevos["ID"])

# Utilizamos el modelo de regresión lineal

model = linear_model.LinearRegression()

model.fit(x,y)

# Visualizamos los coeficientes determinodos por el modelo

b0 = model.coef_
b1 = model.intercept_

print(f"La pendiente estimada es {b0}")
print(f"La ordenada al origen estimada es {b1}")

#Graficamos los datos y la recta generada por el modelo

plt.scatter(x,y)

x1 = range(-100,1850,100)
y1 = b0[0][0]*x1+b1[0]

plt.plot(x1, y1, color = "red")
plt.title("Regresión lineal - Muestra aleatoria")
plt.xlabel("Dosis de RU (ug/huevo)")
plt.ylabel("Índice de daño")

# El r_cuadrado es un coeficiente de 

r_cuadrado = model.score(x,y)
print(f"El coeficiente R² es {r_cuadrado}")

# Un coeficiente R² elevado indica que el modelo aproxima bien los datos observados.