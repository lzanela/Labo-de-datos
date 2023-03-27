#%% Imports
import random
import csv

#%% 
def tirar_dados():
    resultado = []
    for i in range(6):
        resultado.append(random.randrange(1,7))
    return(resultado)

#%% Lectura de archivos txt

nombre_archivo = 'datame.txt'

"""
f = open(nombre_archivo, 'rt', encoding="utf8" ) # abrir para lectura ('r' de read, 't' de text)
data = f.read()
f.close()
data

with open(nombre_archivo, 'rt', encoding="utf8") as file: # otra forma de abrir archivos
 data = file.read()
 # 'data' es una cadena con todo el texto en el archivo
data
print(data)

with open(nombre_archivo, 'rt', encoding="utf8") as file:
    for line in file:
        if 'estudiante' in line:
            print(line)
"""

#%% Lectura de archivos csv

nombre_archivo2 = 'cronograma_sugerido.csv'
lista_materias = []

with open(nombre_archivo2, 'rt', encoding="utf8") as file:
    for line in file:
        datos_linea = line.split(',')
        lista_materias.append(datos_linea[1])

def cuantas_materias(n):
    with open(nombre_archivo2, 'rt', encoding="utf8") as file:
        cant = 0
        for line in file:
            datos_linea = line.split(',')
            if datos_linea[0] == str(n):
                cant = cant + 1
    return cant

def registros(nombre_archivo):
    lista = []
    with open(nombre_archivo, 'rt', encoding="utf8") as f:
        filas = csv.reader(f)
        encabezado = next(filas)
        for fila in filas:
            registro = dict(zip(encabezado,fila)) # armo el diccionario de cada fila
            lista.append(registro) # lo agrego a la lista
    return lista

def materias_cuatrimestre(nombre_archivo, n):
    lista_materias = registros(nombre_archivo)
    lista = []
    for d in lista_materias:
        if d["Cuatrimestre"] == str(n):
            lista.append(d)
    return lista

print(materias_cuatrimestre('cronograma_sugerido.csv', 3))
