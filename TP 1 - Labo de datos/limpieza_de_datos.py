"""

Trabajo Práctico Nº1

Materia Laboratorio de Datos, FCEyN UBA
1er Cuatrimestre 2023

Grupo: Pythonisos

Integrantes: Nicolás Rozenberg, Joaquín Viera, Luca Zanela. 

Sección: Limpieza de datos.

Prerequisito para correr el presente script: Tener instalado la librería `unidecode`, 
además de las utilizadas en la materia.
"""
# %%----------------------------------------------------------------

import pandas as pd
from inline_sql import sql, sql_val
import numpy as np


# %%----------------------------------------------------------------

SOURCES_ROOT_PATH = "./TablasOriginales/"

padron_csv = (
    f"{SOURCES_ROOT_PATH}padron-de-operadores-organicos-certificados-MODIFICADO1.csv"
)
salarios_csv = f"{SOURCES_ROOT_PATH}w_median_depto_priv_clae2.csv"
localidades_csv = f"{SOURCES_ROOT_PATH}localidades-censales.csv"
dict_cod_depto_csv = f"{SOURCES_ROOT_PATH}diccionario_cod_depto.csv"
dict_clae2_csv = f"{SOURCES_ROOT_PATH}diccionario_clae2.csv"

# La fuente padrón ya ha sido modificada manualmente de la fuente original,
# según como se muestra en la sección de decisiones tomadas,
# por algunos errores de correctitud que se han observado, para simplificar
# un poco la sección de limpieza de datos.
df_padron = pd.read_csv(padron_csv, encoding="windows-1252")
df_deptos = pd.read_csv(dict_cod_depto_csv, encoding="utf-8")
df_salarios = pd.read_csv(salarios_csv, encoding="utf-8")
df_localidades = pd.read_csv(localidades_csv, encoding="utf-8")
df_clae2 = pd.read_csv(dict_clae2_csv, encoding="utf-8")

ORIGINAL_PADRON_COUNT = df_padron.shape[0]
ORIGINAL_DEPTOS_COUNT = df_deptos.shape[0]
ORIGINAL_SALARIOS_COUNT = df_salarios.shape[0]
ORIGINAL_LOCALIDADES_COUNT = df_localidades.shape[0]
ORIGINAL_CLAE2_COUNT = df_clae2.shape[0]
# %%----------------------------------------------------------------


"""

Primero, observamos que no hay un estándar en todas las tablas para llamar a los 
indefinidos. Los normalizaremos.
"""

df_padron.replace(
    ["INDEFINIDO", "INDEFINIDA", "SIN DEFINIR", "NC"], np.nan, inplace=True
)
df_localidades.replace([""], np.nan, inplace=True)
df_salarios.replace([-99], np.nan, inplace=True)

# %%----------------------------------------------------------------

"""Comenzaremos intentando pasar modelos extraídos de las fuentes a 1FN. Primero, exploramos la fuente Padrón. 

Problema:  

Analizar qué atributos formarían una clave primaria. Nuestro primer intento es analizar si hay repeticiones de `establecimento`. 
"""

consulta_registros = """
                SELECT count(*) as num, establecimiento
                FROM df_padron
                GROUP BY establecimiento
                HAVING num >= 2
                ORDER BY num DESC
            """

registros = sql ^ consulta_registros
registros_nulos = registros.iloc[0]["num"]

print("Registros nulos: ", registros_nulos / ORIGINAL_PADRON_COUNT)
print(
    "Registros repetidos no nulos (con respecto a los no nulos)",
    sum(registros.iloc[1:]["num"]) / (ORIGINAL_PADRON_COUNT - registros_nulos),
)
print("Registros resultantes: ", registros.shape[0])
registros.head()

# %%----------------------------------------------------------------

"""Vemos que hay una gran cantidad de establecimientos con valor NaN. 
Hemos notado que son aquellos que poseen como valor `Comercializadores` o 
`Elaboradores` en la columna `categoria`. Podríamos diferenciarlos gracias 
al atributo `razon_social`.
"""
# %%----------------------------------------------------------------

consulta_registros = """
                SELECT count(*) as num, establecimiento, razon_social
                FROM df_padron
                GROUP BY establecimiento, razon_social
                ORDER BY num DESC
            """

registros = sql ^ consulta_registros
registros_nulos = registros.loc[
    (registros["establecimiento"].isna()) | (registros["razon_social"].isna()), "num"
]
registros_no_nulos = registros.loc[
    (registros["num"] >= 2)
    & (~registros["establecimiento"].isna())
    & (~registros["razon_social"].isna()),
    "num",
]
print("Registros nulos: ", sum(registros_nulos) / ORIGINAL_PADRON_COUNT)
print(
    "Registros repetidos no nulos (con respecto a los no nulos)",
    sum(registros_no_nulos) / ORIGINAL_PADRON_COUNT,
)
print("Registros resultantes: ", registros.shape[0])
print(
    "Registro no nulo repetido",
    registros.loc[
        (registros["num"] >= 2)
        & (~registros["establecimiento"].isna())
        & (~registros["razon_social"].isna())
    ],
)
registros.head()

# %%----------------------------------------------------------------
"""
Como podemos observar, hay un sólo registro con valores no nulos de {establecimiento, razon_social}
Por lo que ésta sería una buena Clave Candidata
Primero, limpiaremos la tabla y la normalizaremos, al mismo tiempo. Comenzaremos 
seleccionando todos registros que difieran en todo el conjunto de atributos salvo
"productos". Puesto que consideramos que es un error de carga de datos, en estos caso.
"""
# %%----------------------------------------------------------------
consulta_registros = """
                SELECT count(*) as num
                FROM df_padron
                GROUP BY provincia_id, 
                provincia, 
                departamento, 
                localidad, 
                rubro,
                categoria_id, 
                categoria_desc, 
                Certificadora_id, 
                certificadora_deno, 
                razon_social, 
                establecimiento
                HAVING num >= 2
                ORDER BY num DESC
            """

registros = sql ^ consulta_registros
print(
    "Porcentaje de registros que comparten todos los valores excepto productos",
    sum(registros["num"]) * 100 / ORIGINAL_PADRON_COUNT,
)
# %%----------------------------------------------------------------
# Aproximadamente un 2% de los registros comparten todos los valores excepto,
# el valor de "productos". Para esos casos, nos quedamos solo con la primera aparición.

df_padron = df_padron.drop_duplicates(
    subset=set(df_padron.columns).difference(["productos"]), keep="first"
)


# %%----------------------------------------------------------------
# Le asignamos un valor a "establecimiento" para los registros que no poseen valor.
# El valor será "{Nombre de la razón social} {indice}" donde el
# índice es la n-ésima aparición de la razón social.

counts = {}

for i, row in df_padron.iterrows():
    razon_social_value = row["razon_social"]
    establecimiento_value = row["establecimiento"]

    if pd.isnull(establecimiento_value):
        if razon_social_value in counts:
            counts[razon_social_value] += 1
        else:
            counts[razon_social_value] = 1

        # Set the value of B
        df_padron.at[
            i, "establecimiento"
        ] = f"{razon_social_value} {counts[razon_social_value]}"


# %%----------------------------------------------------------------

"""
Analizamos otros problemas de calidad puede llegar a haber
"""

# Corroboramos que cada provincia se corresponda con un único id_provincia
consulta2 = """
              SELECT count(*) as cantidad, provincia_id
              FROM (SELECT DISTINCT provincia_id, provincia
                    FROM df_padron)
              GROUP BY provincia_id
              HAVING cantidad > 1
           """
provincias_repetidas = sql ^ consulta2

print(
    f"Cantidad de tuplas con distintas provincias y mismo id_provincia: {len(sql ^ consulta2)}"
)

# Corroboramos que cada certificadora_deno se corresponda con una única Certificadora_id.
consulta3 = """
              SELECT count(*) as cantidad, Certificadora_id
              FROM (SELECT DISTINCT Certificadora_id, certificadora_deno
                    FROM df_padron)
              GROUP BY Certificadora_id
              HAVING cantidad > 1
           """
certificadoras_repetidas = sql ^ consulta3

print(
    f"Cantidad de tuplas con distintas certificadora_deno y mismo Certificadora_id: {len(sql ^ consulta3)}"
)

# Mismo chequeo que arriba, pero con categoria_id y categoria_desc.

consulta4 = """
              SELECT count(*) as cantidad, categoria_id
              FROM (SELECT DISTINCT categoria_id, categoria_desc
                    FROM df_padron)
              GROUP BY categoria_id
              HAVING cantidad > 1
           """
categorias_repetidas = sql ^ consulta4

print(
    f"Cantidad de tuplas con distintas certificadora_deno y mismo Certificadora_id: {len(sql ^ consulta3)}"
)
# %%----------------------------------------------------------------

"""
Problema: Localizar a los operadores

Existe una gran cantidad de valores en `departamento` que no tienen una correspondencia 
en las que están registradas en el INDEC, tanto en la fuente de Localidades Censales 
como en Diccionario Departamentos, por lo que es un error de consistencia. Normalizaremos 
lo más posible esta correspondencia.

"""

# Utilizamos el módulo unidecode para normalizar lo más posible
# los strings de todas las divisiones políticas. Ésto para disminuir
# los errores de consistencia que poseen las denominaciones de las mismas.
# Optamos por hacerlo todo en pandas, ya que nos dará
# más flexibilidad para llevar a cabo la limpieza de los datos.

from unidecode import unidecode


# Por cuestiones de encoding, el caracter "Ñ" aparece como "ï¿½". Esta función
# lo reemplaza por "Ñ", además de reemplazar los caracteres en minúscula por
# mayúsculas y quitar las tildes.
def clean(x):
    x = unidecode(str(x).replace("ï¿½", "Ñ").upper())
    return x


def clean_capitales_localidades(x):
    if x["departamento_nombre"] == "CAPITAL":
        x["departamento_nombre"] = (
            x["provincia_nombre"] + " " + x["departamento_nombre"]
        )

    return x


def clean_capitales_padron(x):
    if x["departamento"] == "CAPITAL":
        x["departamento"] = x["provincia"] + " " + x["departamento"]

    return x


# Hemos observado un error de inconsistencia:
# En el padrón, CABA aparece como "CIUDAD AUTONOMA BUENOS AIRES", mientras que
# en la tabla de localidades aparece como "Ciudad Autónoma de Buenos Aires"
def clean_buenos_aires(x):
    if x["nombre"] == "CIUDAD AUTONOMA DE BUENOS AIRES":
        x["departamento_nombre"] = "CIUDAD AUTONOMA DE BUENOS AIRES"

        # Observando manualmente cual es el ID del Departamento CABA en la fuente
        # de Diccionario Departamentos
        x["departamento_id"] = 2000

    return x


df_padron["departamento"] = df_padron["departamento"].apply(clean)
df_localidades["provincia_nombre"] = df_localidades["provincia_nombre"].apply(clean)
df_localidades["departamento_nombre"] = df_localidades["departamento_nombre"].apply(
    clean
)
df_localidades["nombre"] = df_localidades["nombre"].apply(clean)
df_localidades["municipio_nombre"] = df_localidades["municipio_nombre"].apply(clean)
df_clae2["clae2_desc"] = df_clae2["clae2_desc"].apply(clean)

# Casos especiales

# Notamos que hay varios casos en los que las capitales provinciales aparecen como
# "Capital". Para mitigar los problemas, y no depender también de la provincia,
# se decidió agregar el nombre de su provincia correspondiente tanto en el padrón como
# en las localidades.
df_padron = df_padron.apply(clean_capitales_padron, axis=1)
df_localidades = df_localidades.apply(clean_capitales_localidades, axis=1)

df_padron["departamento"] = df_padron["departamento"].replace(
    "CIUDAD AUTONOMA BUENOS AIRES", "CIUDAD AUTONOMA DE BUENOS AIRES"
)
df_localidades = df_localidades.apply(clean_buenos_aires, axis=1)
# %%----------------------------------------------------------------
"""
Hay algunos casos de departamentos en Localidades Censales que no están 
en el Diccionario de Departamentos.
"""

deptos = df_deptos["codigo_departamento_indec"].unique()
deptos_erroneos = df_localidades.loc[
    ~df_localidades["departamento_id"].isin(deptos),
    ("departamento_nombre", "departamento_id"),
]
cant = len(deptos_erroneos["departamento_id"].unique())
total = len(df_localidades["departamento_id"].unique())
print(
    f"Porcentaje de departamentos en Localidades Censales que no están incluidos: {cant*100/total} %"
)
deptos_erroneos
# %%----------------------------------------------------------------
"""
Se observa que es una pequeña proporción. Los corregimos a mano, 
ya que asumimos más confiabilidad para la fuente de
Diccionario de Departamentos, para determinar los IDs. Para LA CALDERA
no se ha encontrado ID en el Diccionario de Departamentos
"""

df_localidades.loc[
    df_localidades["departamento_nombre"] == "USHUAIA", "departamento_id"
] = 94014

df_localidades.loc[
    df_localidades["departamento_nombre"] == "RIO GRANDE", "departamento_id"
] = 94007

# %%----------------------------------------------------------------

"""Luego de normalizar, evaluaremos el porcentaje de correspondencias"""

consulta_correspondencias = """
              SELECT
                SUM(
                  CASE 
                    WHEN departamento IN (
                      SELECT DISTINCT departamento_nombre FROM df_localidades
                    ) THEN 1
                    ELSE 0
                  END
                ) AS cantidad_correspondencias,
                COUNT (DISTINCT departamento) as cantidad_valores,
                cantidad_correspondencias*100/cantidad_valores as porcentaje
              FROM df_padron
           """
correspondencias = sql ^ consulta_correspondencias
correspondencias
# %%----------------------------------------------------------------

"""
Específicamente, sólo el 55% posee correspondencia. Por lo tanto, intentaremos 
encontrar un departamento en la fuente de Localidades Censales que corresponda 
con el departamento de cada Padrón.
"""

# Chequeamos que los valores que aparecen en la columna departamento
# se correspondan con un verdadero departamento. De paso, vamos guardando, en un diccionario,
# el ID correspondiente a cada departamento.

deptos_padron = df_padron["departamento"].unique()
deptos = df_localidades["departamento_nombre"].unique()
deptos_erroneos = []
dict_deptos = {}

for depto in deptos_padron:
    if depto not in deptos:
        deptos_erroneos.append(depto)
    else:
        depto_id = df_localidades[df_localidades["departamento_nombre"] == depto][
            "departamento_id"
        ].iloc[0]
        dict_deptos[depto] = depto_id

# %%----------------------------------------------------------------

"""Hemos notado que algunos departamentos corresponden a localidades.
Por eso, armamos lista de localidades y chequeamos si los departamentos del Padrón 
están cargados erróneamente con nombres de localidades. Al mismo tiempo, para los que 
le ocurre eso, moveremos el valor que poseían en `departamento` a `localidad`.
"""

localidades = df_localidades["nombre"].unique()
deptos_con_nombre_de_loc = []
deptos_erroneos2 = []
for depto in deptos_erroneos:
    if depto in localidades:
        # Buscamos el ID del departamento correspondiente a la localidad conseguida
        dict_deptos[depto] = df_localidades[df_localidades["nombre"] == depto][
            "departamento_id"
        ].iloc[0]
        deptos_con_nombre_de_loc.append(depto)

        # Al mismo tiempo, le asignamos un valor a la columna localidad
        df_padron.loc[df_padron["departamento"] == depto, "localidad"] = depto

    else:
        deptos_erroneos2.append(depto)

print(f"# Departamentos erróneos fijándonos en localidades: {len(deptos_erroneos2)}")

# Realizamos lo mismo con municipio, puesto que también se han notado
# algunos casos.
municipios = df_localidades["municipio_nombre"].unique()
deptos_con_nombre_de_muni = []
deptos_erroneos3 = []
for depto in deptos_erroneos2:
    if depto in municipios:
        dict_deptos[depto] = df_localidades[
            df_localidades["municipio_nombre"] == depto
        ]["departamento_id"].iloc[0]
        deptos_con_nombre_de_muni.append(depto)
    else:
        deptos_erroneos3.append(depto)

print(f"# Departamentos erróneos restantes: {len(deptos_erroneos3)}")
print("Listado de departamentos erroneos restantes: ", deptos_erroneos3)
# %%----------------------------------------------------------------

"""Vemos que quedan 72 departamentos erróneos únicos.
"""

print("Cantidad de operadores en padrón: ", ORIGINAL_PADRON_COUNT)
print(
    "Cantidad de operadores para los que no se encontró departamento: ",
    (df_padron[df_padron["departamento"].isin(deptos_erroneos3)].shape[0]),
)

# %%----------------------------------------------------------------

"""
Luego observamos que 103 de los 1395 operadores en el padrón poseen un departamento 
al que no pudimos encontrarle un matcheo. Ésto representa aproximadamente el 8% 
de los operadores en el padrón, por lo que decidimos descartarlos para nuestro análisis. 
Necesitaríamos más información que la provista para asignarles un departamento.
"""

# Eliminamos registros que decidimos que no íbamos a usar
df_padron.drop(
    df_padron[df_padron["departamento"].isin(deptos_erroneos3)].index, inplace=True
)

# Asignamos el ID correspondiente a cada departamento, en una nueva columna
df_padron["departamento_id"] = df_padron["departamento"].apply(
    lambda depto: dict_deptos.get(depto)
)
df_padron = df_padron.astype({"departamento_id": int})
# %%----------------------------------------------------------------

"""
Problema: 

Observamos una gran cantidad de registros en Padron que no poseen definida una localidad, 
por lo que es un error de completitud de información.

"""

cantidad_de_localidades_nan = df_padron["localidad"].isna().sum()
cantidad_de_localidades_padron = df_padron["localidad"].shape[0]

print(
    "Porcentaje de operadores sin localidad",
    str(round(cantidad_de_localidades_nan * 100 / cantidad_de_localidades_padron, 2))
    + "%",
)
# %%----------------------------------------------------------------

"""
Esto representa una gran parte de los operadores registrados. Además, existen localidades 
registradas en el padrón que no tienen correspondencia en localidades censales. Para éstas, 
les indefiniremos el valor, ya que no poseemos la suficiente información para asignarles una 
localidad registrada en Localidades Censales

"""

localidades_censales = df_localidades["nombre"].dropna().unique()

# Conseguimos todos los operadores cuya localidad no esté en localidades censales
df_padron.loc[
    (~df_padron["localidad"].isin(localidades_censales))
    & (df_padron["localidad"].notna()),
    "localidad",
] = np.nan

# %%----------------------------------------------------------------

"""
Y decidimos, para cada operador que no posea localidad, asignarle como localidad a 
localidad cabecera de departamento. De todas formas, es algo para lo que también 
se debería consultar su correctitud.
"""

departamento_ids = df_padron["departamento_id"].unique()

# Creamos un diccionario que nos servirá para asignarle el ID de la
# localidad perteneciente a cada operador registrado
dict_localidad_id = {}

# Primero completamos el diccionario de localidades con las que ya están
localidades = df_padron["localidad"].dropna().unique()

for localidad in localidades:
    localidad_id = df_localidades[df_localidades["nombre"] == localidad].iloc[0]["id"]
    dict_localidad_id[localidad] = localidad_id

# Aquí estamos consiguiendo, para cada ID de departamento,
# la localidad cabecera. La misma inferimos que se puede obtener,
# como la única en su departamento que posee un valor no vacío
# en función.
for id in departamento_ids:
    localidad = df_localidades[
        (df_localidades["departamento_id"] == id) & (df_localidades["funcion"].notna())
    ].iloc[0]

    dict_localidad_id[localidad["nombre"]] = localidad["id"]
    df_padron.loc[
        (df_padron["departamento_id"] == id) & (df_padron["localidad"].isna()),
        "localidad",
    ] = localidad["nombre"]

cantidad_de_localidades_nan = df_padron["localidad"].isna().sum()
cantidad_de_localidades_padron = df_padron["localidad"].shape[0]

print(
    "Porcentaje de operadores sin localidad",
    str(round(cantidad_de_localidades_nan * 100 / cantidad_de_localidades_padron, 2))
    + "%",
)
# %%----------------------------------------------------------------

"""
Ya les asignamos una localidad a cada uno de los operadores. Ahora creamos 
una nueva columna `localidad_id` en Padron, para terminar de resolver el problema, 
con la ayuda del diccionario de localidad con su ID que fuimos creando previamente.

"""

df_padron["localidad_id"] = df_padron["localidad"].apply(
    lambda localidad: dict_localidad_id.get(localidad)
)
df_padron = df_padron.astype({"localidad_id": int})

# %%----------------------------------------------------------------

"""
Problema: 

Observamos que la información para determinar el sector de actividad en el padrón 
(en la columna `rubro`) difiere enormemente de las categorías descriptas en `clae2_desc`, 
por lo que hay un error grave de consistencia y correctitud. Vamos a asignar a cada rubro 
una `clae2` para poder matchearlos al analizar los datos.

"""

# Primero veamos cuáles y cuántos son los rubros que aparecen, y lo mismo con clae2_desc

rubros = df_padron["rubro"].unique()
print(rubros)

# %%----------------------------------------------------------------
# Calculamos porcentaje de rubros que no poseen correspondencia con actividades
# registradas por el INDEC.
actividades = df_clae2["clae2_desc"].unique()

rubros_sin_correspondencia = df_padron.loc[
    df_padron["rubro"].isin(actividades), "rubro"
].unique()

print(
    f"Porcentaje de rubros que no poseen correspondencia con el INDEC: {len(rubros_sin_correspondencia)*100/len(rubros)}"
)
# %%----------------------------------------------------------------

# Hemos notado que algunas tuplas poseen más de un rubro.
mas_de_un_rubro = df_padron["rubro"].str.contains(",").value_counts()
print("Columnas que poseen más de un rubro")
print(mas_de_un_rubro)
print("Porcentaje: ", mas_de_un_rubro)

# %%----------------------------------------------------------------

"""
Creamos un diccionario a mano. Los rubros que poseen mas de un sector, se les 
asignará un sólo clae2, puesto que como se ha visto, han sido pocos los rubros compuestos, 
y nos parece sensato representarlos con uno sólo de los sectores. Es importante señalar 
que sería necesario consultar por la correctitud de esta asignación que realizaremos.
"""

dicc_rubros = {
    "AGRICULTURA": 1,
    "FRUTICULTURA/HORTICULTURA": 1,
    "GANADERIA": 1,
    "HORTICULTURA": 1,
    "FRUTICULTURA": 1,
    "AGRICULTURA/GANADERIA": 1,
    "APICULTURA": 1,
    "ACUICULTURA": 3,
    "AGICULTURA/HORTICULTURA": 1,
    "RECOLECCION SILVESTRE": 2,
    "AGICULTURA/FRUTICULTURA": 1,
    "FRUTICULTURA/AGRICULTURA": 1,
    "GANADERIA/FRUTICULTURA": 1,
    "PRODUCTOS PARA EL CUIDADO PERSONAL": 999,
    "PROCESAMIENTO APICOLA": 10,
    "PROCESAMIENTO FRUTALES Y HORTALIZAS": 10,
    "PROCESAMIENTO CEREALES Y OLEAGINOSAS": 10,
    "ACOPIO Y ACONDICIONAMIENTO DE GRANOS": 52,
    "ELABORACION Y EXTRACCION DE ACEITE": 10,
    "PROCESAMIENTO CULTIVOS INDUSTRIALES": 10,
    "INDUSTRIALIZACION DE YERBA MATE": 10,
    "FRACCIONAMIENTO DE MIEL": 10,
    "FRACCIONAMIENTO DE TE, YERBA MATE": 10,
    "OTROS": 999,
    "EMPAQUE DE FRUTAS NO CITRICAS": 10,
    "INDUSTRIA LACTEA": 10,
    "ALIMENTACION ANIMAL": 10,
    "ACONDICIONAMIENTO Y EMPAQUE": 10,
    "FRIGORIFICOS Y EMPAQUE PARA  FRUTAS": 10,
    "ELABORACION GRANOS": 10,
    "LIMPIEZA DE GRANOS": 10,
    "ELABORACION": 10,
    "EMPAQUE DE HORTALIZAS": 10,
    "ELABORACION ": 10,
    "BODEGA VITIVINICOLA": 11,
    "EXTRACCION DE ACEITE": 10,
    "PROCESAMIENTO DE CEREALES Y OLEAGINOSAS.": 10,
    "PROCESAMIENTO DE MANI": 10,
    "PROCESAMIENTO DE MANI Y SOJA": 10,
    "ELABORACION DE DULCES": 10,
    "PROCESAMIENTO TEXTIL": 13,
    "SECADO, ACONDICIONAMIENTO, ELABORACION Y ACOPIO": 10,
    "PROCESAMIENTO PRODUCTOS ORGANICOS": 10,
    "EXTRACCION DE MIEL": 10,
    "FRACCIONAMIENTO Y EMPAQUE DE ARROZ": 10,
    "ELABORACION DE  JUGOS CONCENTRADOS Y FABRICA DE ACEITES ESENCIALES": 11,
    "ELABORACION DE JUGOS CONCENTRADOS, ACEITE ESCENCIAL Y PULPA DE CITRICOS": 11,
    "ELABORACION DE JUGOS Y BODEGA VITIVINICOLA": 11,
    "SECADERO DE FRUTAS": 10,
    "ELABORACION DE PASAS DE UVA": 10,
    "SECADO DE FRUTAS": 10,
    "INDUSTRIA VITIVINICOLA": 11,
    "INDUSTRIALIZACION DE FRUTAS DESECADAS": 10,
    "ALMACENAMIENTO": 52,
    "FRIGORIFICO PARA  MOSTO": 11,
    "SECADO": 10,
    "FRACCIONAMIENTO DE VINO": 11,
    "PROCESADO Y ENVASADO DE HORTALIZAS": 10,
    "ELABORACION DE JUGO CONCENTRADO": 11,
    "ELABORACION, FRACCIONAMIENTO, EMPAQUE Y ACOPIO": 10,
    "SECADERO , MOLINO Y FRACCIONAMIENTO DE TE, YERBA MATE": 10,
    "DEPOSITO DE YERBA": 52,
    "MOLINO DE YERBA MATE": 10,
    "PROCESAMIENTO DE TE": 10,
    "EMPAQUE DE HORTALIZAS Y FRUTAS NO CITRICAS": 10,
    "FRIGORIFICOS PARA  FRUTAS": 10,
    "ELABORACION DE JUGO CONCENTRADO DE MANZANA Y PERA ": 11,
    "EMPAQUE Y FRIGORIFICO DE FRUTAS NO CITRICAS": 10,
    "ELABORACION DE ACEITE DE ROSA MOSQUETA": 10,
    "EMPAQUE Y FRIGORIFICO FRUTAS NO CITRICAS": 10,
    "ELABORACION DE MANZANA Y PERA DEHIDRATADA": 10,
    "ALMACENAMIENTO Y FRIO PARA FRUTAS NO CITRICAS": 10,
    "EMPAQUE FRUTAS NO CITRICAS": 10,
    "EMPAQUE PARA FRUTA NO CITRICA": 10,
    "EMPAQUE Y FRIO": 10,
    "SECADO, PELADO, ENVASADO Y ALMACENAMIENTO": 10,
    "BODEGA VITIVINICOLA. ELABORACION DE MOSTO CONCENTRADO DE UVAS ": 11,
    "ELABORACION DE MOSTO CONCENTRADO DE UVA": 11,
    "BODEGA VITIVINICOLA Y ELABORACION DE  VINAGRE, MERMELADAS, HUMUS DE LOMBRIZ": 11,
    "FRACCIONAMIENTO": 10,
    "ELABORACION DE MOSTO CONCENTRADO": 11,
    "SECADO - DESPALILLADO - EMBOLSADO": 10,
    "ELABORACION DE DULCES Y FRUTAS EN ALMIBAR": 10,
    "EXTRACCION Y FRACCIONAMIENTO DE MIEL": 10,
    "INDUSTRIA CARNICA": 10,
    "ELABORACION, FRACCIONAMIENTO, ALMACENAMIENTO, CONGELADO": 10,
    "INDUSTRIALIZACION DE LIMON": 10,
    "EMPAQUE DE PRODUCTOS DE LIMON": 10,
    np.nan: 45,
}

# Los que están sin definir son todos comercializadores. Decidimos asignarle 45
# (Comercio al por mayor y al por menor y reparación de vehículos automotores y motos)
# Aplicamos los valores del diccionario a una nueva columna (clae2)

df_padron["clae2"] = df_padron["rubro"].apply(lambda rubro: dicc_rubros.get(rubro))
df_padron = df_padron.astype({"clae2": int})
# %%----------------------------------------------------------------

"""
Problema: 

Observamos que la columna de `productos` en Padron posee valores no atómicos. 
Definiremos la entidad `Producto` y crearemos una relación muchos a muchos con el `padrón`, 
llamada `producto_por_operador`.

"""

# Obtenemos todos los productos que se encuentran en dicha columna
# A cada valor de tipo string de la columna productos, la convertimos a una lista
# cortando el string por ", ". Luego, los valores que no hayan quedado como lista
# (principalmente NaN) se convierten a lista vacía.

# productos_raw es de la forma [["producto1", "producto2"], ["producto1"], ...]

productos_raw = (
    df_padron["productos"]
    .str.replace(" Y ", ", ")
    .str.split(", ")
    .apply(lambda v: v if isinstance(v, list) else [])
    .tolist()
)


productos = set()

for lista_productos in productos_raw:
    for producto in lista_productos:
        productos.add(producto)

df_productos = pd.DataFrame({"producto": list(productos)})
df_productos.sort_values(by="producto")

# Creamos tabla intermedia
operadores = list(
    df_padron.loc[:, ("establecimiento", "razon_social")].itertuples(index=False)
)
mapping = []

for i in range(len(operadores)):
    for producto in productos_raw[i]:
        mapping.append(
            {
                "establecimiento": operadores[i].establecimiento,
                "razon_social": operadores[i].razon_social,
                "producto": producto,
            }
        )

producto_por_operador = pd.DataFrame(mapping)
producto_por_operador.sort_values(by=["razon_social", "establecimiento"])
# %%----------------------------------------------------------------

"""Luego, observamos la fuente Salarios"""

print("Cantidad de registros de salario: ", ORIGINAL_SALARIOS_COUNT)
counts = df_salarios["w_median"].value_counts(dropna=False)
print(counts)
print(
    "Porcentaje de valores desconocidos: ",
    str(round(counts[np.nan] * 100 / ORIGINAL_SALARIOS_COUNT, 2)) + "%",
)

# %%----------------------------------------------------------------

"""
Existe una gran cantidad de valores NaN en la columna w_median. 
Descartaremos las filas para el análisis ya que no nos aportarán información, 
y no tenemos forma actualmente de conseguirla. Además de que seguimos teniendo 
una cantidad considerable para nuesto análisis.
"""

df_salarios.dropna(inplace=True)

# %%----------------------------------------------------------------

"""
Ahora, ya estamos en condiciones de dividir las tablas, para conseguir un diseño 3FN, 
respetando las dependencias funcionales que se encuentran en el informe.
"""

# Tabla Operador
consulta_operador = """
                SELECT DISTINCT
                  razon_social,
                  establecimiento,
                  localidad_id,
                  clae2,
                  Certificadora_id as certificadora_id,
                  categoria_id,
                  rubro,
                FROM df_padron
                ORDER BY localidad_id
            """

df_operador = sql ^ consulta_operador
df_operador.head()

# %%----------------------------------------------------------------

# Tabla Certificadora
consulta_certificadora = """
                SELECT DISTINCT
                  Certificadora_id as certificadora_id,
                  certificadora_deno
                FROM df_padron
            """

df_certificadora = sql ^ consulta_certificadora

# Tabla Categoria
consulta_categoria = """
                SELECT DISTINCT
                  categoria_id,
                  categoria_desc
                FROM df_padron
            """

df_categoria = sql ^ consulta_categoria

# %%----------------------------------------------------------------

"""
Para las siguientes tablas, de divisiones políticas, vale aclarar que la misma 
información se encuentra en la fuente Diccionario Departamentos. Sin embargo, como 
ésta fue la utilizada principalmente para matchear los departamentos y localidades 
del Padrón, crearemos las tablas a partir de ésta.
"""

# Tabla Provincia


consulta_provincia = """
                SELECT DISTINCT
                  provincia_id as id,
                  provincia_nombre as nombre
                FROM df_localidades
            """

df_provincia = sql ^ consulta_provincia

# Tabla Departamento
consulta_departamento = """
                SELECT DISTINCT
                  departamento_id as id,
                  departamento_nombre as nombre,
                  provincia_id
                FROM df_localidades
                WHERE departamento_id IS NOT NULL AND
                 departamento_nombre IS NOT NULL
            """

df_departamento = sql ^ consulta_departamento

# Tabla Municipio
consulta_municipio = """
                SELECT DISTINCT
                  municipio_id as id,
                  municipio_nombre as nombre,
                  departamento_id
                FROM df_localidades
                WHERE municipio_id IS NOT NULL and municipio_nombre IS NOT NULL
            """

df_municipio = sql ^ consulta_municipio


# Tabla Localidad
consulta_localidad = """
                SELECT DISTINCT
                  id,
                  nombre,
                  centroide_lon,
                  centroide_lat,
                  categoria,
                  fuente,
                  municipio_id,
                  departamento_id
                FROM df_localidades
                WHERE nombre IS NOT NULL AND departamento_id IS NOT NULL
            """

df_localidad = sql ^ consulta_localidad

# Tabla Sector

# Observamos que la letra "OTROS" no posee valor en la columna "letra".
# Le asignamos un valor artificial "OTROS"
df_clae2.loc[df_clae2["letra_desc"] == "OTROS", "letra"] = "OTROS"

consulta_sector = """
                SELECT DISTINCT
                  clae2,
                  clae2_desc,
                  letra as letra_id
                FROM df_clae2
            """

df_sector = sql ^ consulta_sector

# Tabla Letra
consulta_letra = """
                SELECT DISTINCT
                  letra as letra_id,
                  letra_desc
                FROM df_clae2
            """

df_letra = sql ^ consulta_letra

# Tabla RegistroSalarial

consulta_registro_salarial = """
                SELECT DISTINCT
                  fecha,
                  codigo_departamento_indec as departamento_id,
                  clae2,
                  w_median
                FROM df_salarios
            """

df_registro_salarial = sql ^ consulta_registro_salarial


# %%----------------------------------------------------------------

"""
Exportación del modelo
"""

MODEL_ROOT_PATH = "./TablasLimpias/"

df_registro_salarial.to_csv(f"{MODEL_ROOT_PATH}registro_salarial.csv", index=False)
df_letra.to_csv(f"{MODEL_ROOT_PATH}letra.csv", index=False)
df_sector.to_csv(f"{MODEL_ROOT_PATH}sector.csv", index=False)
df_localidad.to_csv(f"{MODEL_ROOT_PATH}localidad.csv", index=False)
df_municipio.to_csv(f"{MODEL_ROOT_PATH}municipio.csv", index=False)
df_departamento.to_csv(f"{MODEL_ROOT_PATH}departamento.csv", index=False)
df_provincia.to_csv(f"{MODEL_ROOT_PATH}provincia.csv", index=False)
df_categoria.to_csv(f"{MODEL_ROOT_PATH}categoria.csv", index=False)
df_certificadora.to_csv(f"{MODEL_ROOT_PATH}certificadora.csv", index=False)
df_operador.to_csv(f"{MODEL_ROOT_PATH}operador.csv", index=False)
df_productos.to_csv(f"{MODEL_ROOT_PATH}producto.csv", index=False)
producto_por_operador.to_csv(f"{MODEL_ROOT_PATH}producto_por_operador.csv", index=False)
