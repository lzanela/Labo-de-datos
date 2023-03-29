import numpy as np
import pandas as pd

archivo = 'Actividad-01-datos.csv'
df = pd.read_csv(archivo)

columnas = df.columns
cantidad_de_datos = df.index
provincias = df['provincia'].unique()

#print(df.describe())
#print('Las provincias que aparecen son:',provincias)
#print('Las columnas que aparecen son:',set(columnas))
#print(cantidad_de_datos)

# Creamos un diccionario con las provincias como claves, y la cantidad de ejemplares en cada provincia como significados
ejemplares_por_provincia = {}
for provincia in provincias:
    cant = (df['provincia'] == provincia).sum()
    ejemplares_por_provincia[provincia] = cant

# Diccionario con municipio, municipio_id

def ids_por_municipio(data):
    ids_por_muni = {}
    i=0
    data = data.drop_duplicates(subset = 'municipio')
    
    while (i < len(data)):
        muni = data.iloc[i]['municipio']
        id_muni = data.iloc[i]['municipio_id']
        ids_por_muni[muni] = id_muni
        i+=1
    return ids_por_muni

