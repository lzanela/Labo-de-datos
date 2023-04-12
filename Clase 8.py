import pandas as pd
from inline_sql import sql, sql_val
#%%
print()
print("# =============================================================================")
print("# Creamos los datasets que vamos a utilizar en este programa")
print("# =============================================================================")

# Ejercicio 1,2
vuelo      = pd.read_csv("vuelo.csv")    
aeropuerto = pd.read_csv("aeropuerto.csv")    
pasajero   = pd.read_csv("pasajero.csv")    
reserva    = pd.read_csv("reserva.csv")

# Ejercicio JOIN tuplas espúreas
empleadoRol= pd.read_csv("empleadoRol.csv")    
rolProyecto= pd.read_csv("rolProyecto.csv") 
# Funciones de Agregacion
examen     = pd.read_csv("examen.csv")
# OPERACIONES ENTRE DATABASES
def get_alumnosBD():
    # Genera el dataframe "alumnosBD" que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. ID
        # 2. Nombre
        
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    alumnosBD = pd.DataFrame(columns = ['ID', 'Nombre'])
    # ... Agregamos cada una de las filas al dataFrame
    alumnosBD = pd.concat([alumnosBD,pd.DataFrame([
        {'ID' : 1, 'Nombre' : 'Diego' },
        {'ID' : 2, 'Nombre' : 'Laura' },
        {'ID' : 3, 'Nombre' : 'Marina'}
                                                    ])
                        ])
    return alumnosBD


def get_alumnosTLeng():
    # Genera el dataframe alumnosTLeng que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. ID
        # 2. Nombre
        
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    alumnosTLeng = pd.DataFrame(columns = ['ID', 'Nombre'])
    # ... Agregamos cada una de las filas al dataFrame
    alumnosTLeng = pd.concat([alumnosTLeng,pd.DataFrame([
        {'ID' : 2, 'Nombre' : 'Laura'    },
        {'ID' : 4, 'Nombre' : 'Alejandro'}
                                                        ])
                        ])
    return alumnosTLeng
alumnosBD      = get_alumnosBD()
alumnosTLeng   = get_alumnosTLeng()

def get_persona_ejemploCrossJoin():
    # Genera el dataframe "persona" que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. Nombre
        # 2. Nacionalidad
        
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    persona = pd.DataFrame(columns = ['Nombre', 'Nacionalidad'])
    # ... Agregamos cada una de las filas al dataFrame
    persona = pd.concat([persona,pd.DataFrame([
        {'Nombre' : 'Diego'   , 'Nacionalidad' : 'AR'    },
        {'Nombre' : 'Laura'   , 'Nacionalidad' : 'BR'    },
        {'Nombre' : 'Marina'  , 'Nacionalidad' : 'AR'    }
                                              ])
                        ])
    return persona

def get_nacionalidades():
    # Genera el dataframe "nacionalidades" que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. IDN (Id Nacionalidad)
        # 2. Detalle
    
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    nacionalidades = pd.DataFrame(columns = ['IDN', 'Detalle'])
    # ... Agregamos cada una de las filas al dataFrame
    nacionalidades = pd.concat([nacionalidades,pd.DataFrame([
        {'IDN' : 'AR', 'Detalle' : 'Agentina'},
        {'IDN' : 'BR', 'Detalle' : 'Brasilera'},
        {'IDN' : 'CH', 'Detalle' : 'Chilena'}
                                                          ])
                        ])
    return nacionalidades
persona        = get_persona_ejemploCrossJoin()
nacionalidades = get_nacionalidades()

def get_persona_ejemplosJoin():
    # Genera el dataframe "persona" que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. Nombre
        # 2. Nacionalidad
        
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    persona = pd.DataFrame(columns = ['Nombre', 'Nacionalidad'])
    # ... Agregamos cada una de las filas al dataFrame
    persona = pd.concat([persona,pd.DataFrame([
        {'Nombre' : 'Diego'   , 'Nacionalidad' : 'BR'    },
        {'Nombre' : 'Laura'   , 'Nacionalidad' : None    },
        {'Nombre' : 'Marina'  , 'Nacionalidad' : 'AR'    },
        {'Nombre' : 'Santiago', 'Nacionalidad' : 'UY'    }
                                              ])
                        ])
    return persona
persona1 = get_persona_ejemplosJoin()

def get_se_inscribe_en_ejemploMismosNombres():
    # Genera el dataframe "se_inscribe_en" que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. LU
        # 2. Codigo_materia
        
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    se_inscribe_en = pd.DataFrame(columns = ['LU','Codigo_materia'])
    # ... Agregamos cada una de las filas al dataFrame
    se_inscribe_en = pd.concat([se_inscribe_en,pd.DataFrame([
        {'LU':'123/09','Codigo_materia': 1},
        {'LU':' 22/10','Codigo_materia': 1},
        {'LU':' 22/10','Codigo_materia': 2},
        {'LU':'344/09','Codigo_materia': 1}
                                              ])
                        ])
    return se_inscribe_en

def get_materia_ejemploMismosNombres():
    # Genera el dataframe "materia" que contiene las siguientes columnas 
    # (en el orden mencionado):
        # 1. Codigo_materia
        # 2. Nombre
        
    # ... Creamos el dataframe vacío (sólo con los nombres de sus columnas)
    materia = pd.DataFrame(columns = ['Codigo_materia','Nombre'])
    # ... Agregamos cada una de las filas al dataFrame
    materia = pd.concat([materia,pd.DataFrame([
        {'Codigo_materia': 1, 'Nombre':'Laboratorio de Datos'   },
        {'Codigo_materia': 2, 'Nombre':'Análisis II'   },
        {'Codigo_materia': 3, 'Nombre':'Probabilidad'   }
                                              ])
                        ])
    return materia
se_inscribe_en= get_se_inscribe_en_ejemploMismosNombres()
materia       =get_materia_ejemploMismosNombres()
#%%
# =============================================================================
# DEFINICION DE FUNCIÓN DE IMPRESIÓN EN PANTALLA
# =============================================================================
# Imprime en pantalla en un formato ordenado:
    # 1. Consigna
    # 2. Cada dataframe de la lista de dataframes de entrada
    # 3. Query
    # 4. Dataframe de salida
def imprimirEjercicio(consigna, listaDeDataframesDeEntrada, consultaSQL):
    
    print("# -----------------------------------------------------------------------------")
    print("# Consigna: ", consigna)
    print("# -----------------------------------------------------------------------------")
    print()
    for i in range(len(listaDeDataframesDeEntrada)):
        print("# Entrada 0",i,sep='')
        print("# -----------")
        print(listaDeDataframesDeEntrada[i])
        print()
    print("# SQL:")
    print("# ----")
    print(consultaSQL)
    print()
    print("# Salida:")
    print("# -------")
    print(sql^ consultaSQL)
    print()
    print("# -----------------------------------------------------------------------------")
    print("# -----------------------------------------------------------------------------")
    print()
    print()

# =============================================================================
# EJERCICIOS
# =============================================================================
#%% EJERCICIO 1
    
# Ejericicio 1.1

consigna = "Ejercicio 1.1"    

consultaSQL = """
                SELECT DISTINCT Codigo, Nombre
                FROM aeropuerto
                WHERE Ciudad='Londres'
              """

imprimirEjercicio(consigna, [aeropuerto], consultaSQL)

# Ejericicio 1.2

consigna = "Ejercicio 1.2"

consultaSQL = """
                SELECT DISTINCT Ciudad AS City
                FROM aeropuerto
                WHERE Codigo='ORY' OR Codigo='CDG'
              """

imprimirEjercicio(consigna, [aeropuerto], consultaSQL)

# Ejericicio 1.3

consigna = "Ejercicio 1.3"

consultaSQL = """
                SELECT DISTINCT Numero
                FROM vuelo
                WHERE Origen='CDG' AND Destino='LHR'
              """

imprimirEjercicio(consigna, [vuelo], consultaSQL)

# Ejericicio 1.4

consigna = "Ejercicio 1.4"

consultaSQL = """
                SELECT DISTINCT Numero
                FROM vuelo
                WHERE (Origen='CDG' AND Destino='LHR') OR (Destino='CDG' AND Origen='LHR')
              """

imprimirEjercicio(consigna, [vuelo], consultaSQL)

# Ejericicio 1.5

consigna = "Ejercicio 1.5"

consultaSQL = """
                SELECT DISTINCT Fecha
                FROM reserva
                WHERE Precio>200
              """

imprimirEjercicio(consigna, [reserva], consultaSQL)

#%% OPERACIONES ENTRE DATABASES
consigna = "Union"

consultaSQL = """
                SELECT DISTINCT *
                FROM alumnosBD
                UNION
                SELECT DISTINCT *
                FROM alumnosTLeng
              """

imprimirEjercicio(consigna, [alumnosBD,alumnosTLeng], consultaSQL)

consigna = "Union All"

consultaSQL = """
                SELECT DISTINCT *
                FROM alumnosBD
                UNION ALL
                SELECT DISTINCT *
                FROM alumnosTLeng
              """

imprimirEjercicio(consigna, [alumnosBD,alumnosTLeng], consultaSQL)

consigna = "Intersect"

consultaSQL = """
                SELECT DISTINCT *
                FROM alumnosBD
                INTERSECT
                SELECT DISTINCT *
                FROM alumnosTLeng
              """

imprimirEjercicio(consigna, [alumnosBD,alumnosTLeng], consultaSQL)

consigna = "Except"

consultaSQL = """
                SELECT DISTINCT *
                FROM alumnosBD
                EXCEPT
                SELECT DISTINCT *
                FROM alumnosTLeng
              """

imprimirEjercicio(consigna, [alumnosBD,alumnosTLeng], consultaSQL)

consigna = "Producto cartesiano"

consultaSQL = """
                SELECT DISTINCT *
                FROM persona
                CROSS JOIN
                nacionalidades
              """

imprimirEjercicio(consigna, [persona,nacionalidades], consultaSQL)

consultaSQL = """
                SELECT DISTINCT *
                FROM persona, nacionalidades
              """

imprimirEjercicio(consigna, [persona,nacionalidades], consultaSQL)

consigna = "Inner Join"

#opcion mas eficiente
consultaSQL = """
                SELECT DISTINCT *
                FROM persona
                INNER JOIN nacionalidades
                ON nacionalidad=IDN
              """

imprimirEjercicio(consigna, [persona,nacionalidades], consultaSQL)

#opcion menos eficiente
consultaSQL = """
                SELECT DISTINCT *
                FROM persona, nacionalidades
                WHERE nacionalidad=IDN
              """

imprimirEjercicio(consigna, [persona,nacionalidades], consultaSQL)

consigna = "LEFT OUTER JOIN"

consultaSQL = """
                SELECT DISTINCT *
                FROM persona1
                LEFT OUTER JOIN nacionalidades
                ON nacionalidad = IDN
              """

imprimirEjercicio(consigna, [persona1,nacionalidades], consultaSQL)

consigna = "INNER JOIN"

consultaSQL = """
                SELECT DISTINCT LU,Nombre
                FROM se_inscribe_en
                INNER JOIN materia
                ON se_inscribe_en.Codigo_Materia=materia.Codigo_Materia
              """

imprimirEjercicio(consigna, [se_inscribe_en,materia], consultaSQL)
#%% EJERCICIO 2

# Ejercicio 2.1
consigna = "Ejercicio 2.1"

consultaSQL = """
                SELECT DISTINCT Numero
                FROM vuelo
                INTERSECT
                SELECT DISTINCT NroVuelo
                FROM reserva
              """

imprimirEjercicio(consigna, [vuelo,reserva], consultaSQL)

# Ejercicio 2.2
consigna = "Ejercicio 2.2"

consultaSQL = """
                SELECT DISTINCT Numero
                FROM vuelo
                EXCEPT
                SELECT DISTINCT NroVuelo
                FROM reserva
              """

imprimirEjercicio(consigna, [vuelo,reserva], consultaSQL)

# Ejercicio 2.3
consigna = "Ejercicio 2.3"

consultaSQL = """
                SELECT DISTINCT Origen
                FROM vuelo
                UNION
                SELECT DISTINCT Destino
                FROM vuelo
              """

imprimirEjercicio(consigna, [vuelo], consultaSQL)

#%% EJERCICIO 3

# Ejercicio 3.1
consigna = "Ejercicio 3.1"

consultaSQL = """
                SELECT DISTINCT Ciudad
                FROM vuelo
                INNER JOIN aeropuerto
                ON Origen=Codigo
                WHERE Numero=165
              """

imprimirEjercicio(consigna, [vuelo,aeropuerto], consultaSQL)

# Ejercicio 3.2
consigna = "Ejercicio 3.2"

consultaSQL = """
                SELECT DISTINCT Nombre
                FROM pasajero
                INNER JOIN reserva
                ON pasajero.DNI=reserva.DNI
                WHERE Precio<200
              """

imprimirEjercicio(consigna, [pasajero,reserva], consultaSQL)

# Ejercicio 3.3
consigna = "Ejercicio 3.3"

consultaSQL = """
                SELECT DISTINCT pasajero.Nombre,Fecha,Destino
                FROM pasajero
                INNER JOIN reserva
                ON pasajero.DNI=reserva.DNI
                INNER JOIN vuelo
                ON Numero=NroVuelo
                INNER JOIN aeropuerto
                ON Origen=Codigo
                WHERE aeropuerto.Ciudad='Madrid'
              """

imprimirEjercicio(consigna, [pasajero,reserva,vuelo, aeropuerto], consultaSQL)

# Ejercicio 3.4
#forma poco optimizable
consigna = "Ejercicio 3.4"

consultaSQL = """
                SELECT DISTINCT r.fecha,v.Salida,p.Nombre
                FROM reserva AS r,pasajero AS p, vuelo AS v
                WHERE r.DNI=p.DNI AND r.NroVuelo=v.Numero
              """

imprimirEjercicio(consigna, [pasajero,reserva,vuelo, aeropuerto], consultaSQL)

#%% TUPLAS ESPUREAS, errores muy comunes y dificiles de detectar
# Joinear en atributos claves

consigna = ""

consultaSQL = """
                SELECT DISTINCT empleado, empleadoRol.rol, proyecto
                FROM empleadoRol
                INNER JOIN rolProyecto
                ON empleadoRol.rol=rolProyecto.rol
              """

imprimirEjercicio(consigna, [empleadoRol,rolProyecto], consultaSQL)

#%% Funciones de agregacion

consigna = "contar los elemnetos de una columna"

consultaSQL = """
                SELECT  count(*) AS cantidadExamenes
                FROM examen
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = "contar cuantos elementos hay para cada elto distinto de instancia"

consultaSQL = """
                SELECT  Instancia, COUNT(*) AS Asistieron
                FROM examen
                GROUP BY Instancia
                HAVING Asistieron<4
                ORDER BY Instancia ASC;
              """

imprimirEjercicio(consigna, [examen], consultaSQL)
#order de los ordena, having te permite poner condicinoes

consigna = "promedio de edad en cada instancia de examen"

consultaSQL = """
                SELECT  Instancia, AVG(Edad) AS PromedioEdad
                FROM examen
                GROUP BY Instancia
                ORDER BY Instancia
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = "promedio de nota en cada parcial"

consultaSQL = """
                SELECT  Instancia, AVG(Nota) AS PromedioNota
                FROM examen
                GROUP BY Instancia
                HAVING Instancia='Parcial-01' OR Instancia='Parcial-02'
                ORDER BY Instancia
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consultaSQL = """
                SELECT  Instancia, AVG(Nota) AS PromedioNota
                FROM examen
                GROUP BY Instancia
                HAVING Instancia LIKE 'Parcial%'
                ORDER BY Instancia
              """
#el LIKE sirve para que solo tenga en cuenta un patron dado, hay distintas formas de pasarle el patron
imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = "agregar columna dependiendo el valor de otra columna"

consultaSQL = """
                SELECT  Nombre, Nota,
                    CASE WHEN Nota>=4 THEN 'APROBO' ELSE 'NO APROBO' END AS Estado
                FROM examen
                WHERE Instancia='Parcial-01'
                ORDER BY Nombre;
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = ""

consultaSQL = """
                SELECT  Instancia,
                    CASE WHEN Nota>=4 THEN 'APROBO' ELSE 'NO APROBO' END AS Estado,
                    COUNT(*) AS Cantidad
                FROM examen
                GROUP BY Instancia,Estado
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = ""

consultaSQL = """
                SELECT e1.Nombre, e1.Instancia, e1.Nota
                FROM examen AS e1
                WHERE e1.Nota > (
                SELECT AVG(e2.Nota)
                FROM examen AS e2
                WHERE e2.Instancia = e1.Instancia
                )
                ORDER BY Instancia ASC, Nota DESC;
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = ""

consultaSQL = """
                SELECT e1.Nombre, e1.Instancia, e1.Nota
                FROM Examen AS e1
                WHERE e1.Nota >= ALL (
                SELECT e2.Nota
                FROM Examen AS e2
                WHERE e2.Instancia = e1.Instancia
                )
                ORDER BY e1.Instancia ASC, e.Nombre ASC;
              """

imprimirEjercicio(consigna, [examen], consultaSQL)

consigna = ""

consultaSQL = """
                SELECT e1.Nombre, e1.Instancia, e1.Nota
                FROM Examen AS e1
                WHERE NOT EXISTS (
                SELECT *
                FROM Examen AS e2
                WHERE e2.Nombre = e1.Nombre AND
                e2.Instancia LIKE ‘Recuperatorio%’
                )
                ORDER BY e1.Nombre ASC, e1.Instancia ASC;
              """

imprimirEjercicio(consigna, [examen], consultaSQL)