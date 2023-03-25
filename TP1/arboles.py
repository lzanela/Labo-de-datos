"""

TP Arbolado Porteño

Laboratorio de Datos

Grupo: Pythonisos
Alumnos: Joaquín Viera, Nicolás Rosenberg, Luca Zanela

"""


#%%
""" Imports """

import csv

#%%
""" Ejercicio 1

Definir una función leer_parque(nombre_archivo, parque) que abra el
archivo indicado y devuelva una lista de diccionarios con la información
del parque especificado. La lista debe tener un diccionario por cada árbol
del parque elegido. Dicho diccionario debe tener los datos
correspondientes a un árbol (recordá que cada fila del csv corresponde a
un árbol).

"""

def leer_parque(nombre_archivo, parque):
    lista_arboles=[]

    with open(nombre_archivo, "rt", encoding="utf8") as f:
        filas = csv.reader(f)
        encabezado = next(filas)

        for fila in filas:
            arbol = dict(zip(encabezado, fila))
            _limpiar_arbol(arbol)

            if arbol["espacio_ve"] == parque:
                lista_arboles.append(arbol)

    return lista_arboles

#%%
""" Ejercicio 2

Escribir una función especies(lista_arboles) que tome una lista de árboles
como la generada en el ejercicio anterior y devuelva el conjunto de
especies (la columna 'nombre_com' del archivo) que figuran en la lista.
"""

def especies(lista_arboles):
    res = set()

    for arbol in lista_arboles:
        res.add(arbol["nombre_com"])

    return res

#%%
""" Ejercicio 3 

Escribir una función contar_ejemplares(lista_arboles) que, dada una lista
como la que generada con leer_parque(), devuelva un diccionario en el
que las especies sean las claves y tengan como valores asociados la
cantidad de ejemplares en esa especie en la lista dada
"""

def contar_ejemplares(lista_arboles):
    res={}

    for especie in especies(lista_arboles):
        count = 0

        for arbol in lista_arboles:
            if arbol["nombre_com"] == especie:
                count += 1

        res[especie] = count

    return res

#%%
"""  Ejercicio 4

Escribir una función obtener_alturas(lista_arboles, especie) que, dada una
lista de árboles como la anterior y una especie de árbol (un valor de la
columna 'nombre_com' del archivo), devuelva una lista con las alturas
(columna 'altura_tot') de los ejemplares de esa especie en la lista.

"""

def obtener_alturas(lista_arboles, especie):
    alturas=[]

    for arbol in lista_arboles:
        if arbol["nombre_com"] == especie:
            altura = arbol["altura_tot"]
            alturas.append(altura)

    return alturas

#%%
"""  Ejercicio 5

Escribir una función obtener_inclinaciones(lista_arboles, especie) que,
dada una especie de árbol y una lista de árboles como la anterior,
devuelva una lista con las inclinaciones (columna 'inclinacio') de los
ejemplares de esa especie.

"""

def obtener_inclinaciones(lista_arboles, especie):
    inclinaciones = []

    for arbol in lista_arboles:
        if arbol["nombre_com"] == especie:
            inclinacion = arbol["inclinacio"]
            inclinaciones.append(inclinacion)

    return inclinaciones

#%%
"""  Ejercicio 6

Combinando la función especies() con obtener_inclinaciones() escribir una
función especimen_mas_inclinado(lista_arboles) que, dada una lista de
árboles devuelva la especie que tiene el ejemplar más inclinado y su
inclinación.

"""

def especimen_mas_inclinado(lista_arboles):
    conj_especies = especies(lista_arboles)
    especie_mas_inclinada = ""
    inclinacion_maxima = -1

    for especie in conj_especies:
        inclinaciones = obtener_inclinaciones(lista_arboles, especie)
        max_inclinacion = max(inclinaciones)
        
        if max_inclinacion > inclinacion_maxima:
            inclinacion_maxima = max_inclinacion
            especie_mas_inclinada = especie
    
    return {
        "especie": especie_mas_inclinada,
        "inclinacion": inclinacion_maxima
    }

# %%
""" Ejercicio 7 

Volver a combinar las funciones anteriores para escribir la función
especie_promedio_mas_inclinada(lista_arboles) que, dada una lista de
árboles devuelva la especie que en promedio tiene la mayor inclinación y
el promedio calculado.

"""
def especie_promedio_mas_inclinada(lista_arboles):
    conj_especies = especies(lista_arboles)
    especie_promedio_mas_inclinada = ""
    prom_inclinacion_maximo = -1

    for especie in conj_especies:
        inclinaciones = obtener_inclinaciones(lista_arboles, especie)
        prom_inclinacion = sum(inclinaciones)/len(inclinaciones)
        
        if prom_inclinacion > prom_inclinacion_maximo:
            prom_inclinacion_maximo = prom_inclinacion
            especie_promedio_mas_inclinada = especie
    
    return {
        "especie": especie_promedio_mas_inclinada,
        "promedio de inclinacion": prom_inclinacion_maximo
    }

#%%
"""
Funciones auxiliares
"""

def _limpiar_arbol(arbol: dict):
    arbol["altura_tot"] = float(arbol["altura_tot"])
    arbol["inclinacio"] = float(arbol["inclinacio"])

#%% 
""" Código extra """

# Definimos lista_espacios como un conjunto que tiene los nombres
# de todos los parques que figuran en la base de datos, para poder
# identificar bien el nombre de cada espacio (ya que, por ejemplo,
# el Parque los Andes se llama 'ANDES, LOS', y Parque Centenario 
# se llama 'CENTENARIO').

lista_espacios= set()
nombre_archivo = 'arbolado-en-espacios-verdes.csv'

with open(nombre_archivo, "rt", encoding="utf8") as f:
    filas = csv.reader(f)
    encabezado = next(filas)

    for fila in filas:
        arbol = dict(zip(encabezado, fila))
        lista_espacios.add(arbol["espacio_ve"])

# Código para testear las funciones definidas
 
arboles_por_parque = {
    "GENERAL PAZ": leer_parque("./arbolado-en-espacios-verdes.csv", "GENERAL PAZ"),
    "PARQUE DE LOS ANDES": leer_parque("./arbolado-en-espacios-verdes.csv", "ANDES, LOS"),
    "CENTENARIO": leer_parque('./arbolado-en-espacios-verdes.csv', "CENTENARIO")
}

a = especies(arboles_por_parque["GENERAL PAZ"])
b = contar_ejemplares(arboles_por_parque["GENERAL PAZ"])
c = contar_ejemplares(arboles_por_parque["PARQUE DE LOS ANDES"])
d = contar_ejemplares(arboles_por_parque["CENTENARIO"])


alturas_por_parque = {
    "GENERAL PAZ": obtener_alturas(arboles_por_parque["GENERAL PAZ"], "Jacarandá"),
    "PARQUE DE LOS ANDES": obtener_alturas(arboles_por_parque["PARQUE DE LOS ANDES"], "Jacarandá"),
    "CENTENARIO": obtener_alturas(arboles_por_parque["CENTENARIO"], "Jacarandá"),
}

maximos_por_parque = {
    "GENERAL PAZ": max(alturas_por_parque["GENERAL PAZ"]),
    "PARQUE DE LOS ANDES": max(alturas_por_parque["PARQUE DE LOS ANDES"]),
    "CENTENARIO": max(alturas_por_parque["CENTENARIO"]),
}

prom_por_parque = {
    "GENERAL PAZ": sum(alturas_por_parque["GENERAL PAZ"])/len(alturas_por_parque["GENERAL PAZ"]),
    "PARQUE DE LOS ANDES": sum(alturas_por_parque["PARQUE DE LOS ANDES"])/len(alturas_por_parque["PARQUE DE LOS ANDES"]),
    "CENTENARIO": sum(alturas_por_parque["CENTENARIO"])/len(alturas_por_parque["CENTENARIO"]),
}


especimen_mas_inclinado_por_parque = {
    "GENERAL PAZ": especimen_mas_inclinado(arboles_por_parque["GENERAL PAZ"]),
    "PARQUE DE LOS ANDES": especimen_mas_inclinado(arboles_por_parque["PARQUE DE LOS ANDES"]),
    "CENTENARIO": especimen_mas_inclinado(arboles_por_parque["CENTENARIO"]),
}


especie_promedio_mas_inclinada_por_parque = {
    "GENERAL PAZ": especie_promedio_mas_inclinada(arboles_por_parque["GENERAL PAZ"]),
    "PARQUE DE LOS ANDES": especie_promedio_mas_inclinada(arboles_por_parque["PARQUE DE LOS ANDES"]),
    "CENTENARIO": especie_promedio_mas_inclinada(arboles_por_parque["CENTENARIO"]),
}

#print(especimen_mas_inclinado_por_parque["CENTENARIO"])
#print(especie_promedio_mas_inclinada_por_parque["PARQUE DE LOS ANDES"])

#%% 
""" Observaciones sobre la estructura de la base de datos """

"""visualizar lista de datos
0=long  1=lat  2=id_arbol  3=altura_tot  4=diametro  5=inclinacio
6=id_especie  7=nombre_com  8=nombre_cie  9=tipo_folla  10=espacio_ve
11=ubicacion  12=nombre_fam  13=nombre_gen  14=origen  15=coord_x
16=coord_y
lista_datos=[long,lat,id_arbol,altura_tot,diametro,inclinacio,id_especie,nombre_com,nombre_cie,tipo_folla,espacio_ve,ubicacion,nombre_fam,nombre_gen,origen,coord_x,coord_y]
"""
"""nombre_archivo = 'arbolado-en-espacios-verdes.csv'
with open(nombre_archivo, 'rt') as file:
    for line in file:
        arbol = line.split(',')
        print(arbol[dato])"""
