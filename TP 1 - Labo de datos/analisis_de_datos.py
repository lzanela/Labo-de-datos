"""

Trabajo Práctico Nº1

Materia Laboratorio de Datos, FCEyN UBA
1er Cuatrimestre 2023

Grupo: Pythonisos

Integrantes: Nicolás Rozenberg, Joaquín Viera, Luca Zanela. 

Sección: Análisis de los Datos.

"""

#%%----------------------------------------------------------------
import pandas as pd
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_ROOT_PATH = "./TablasLimpias/"

categoria_csv = f"{MODEL_ROOT_PATH}categoria.csv"
registro_salarial_csv = f"{MODEL_ROOT_PATH}registro_salarial.csv"
sector_csv = f"{MODEL_ROOT_PATH}sector.csv"
provincia_csv = f"{MODEL_ROOT_PATH}provincia.csv"
producto_por_operador_csv = f"{MODEL_ROOT_PATH}producto_por_operador.csv"
producto_csv = f"{MODEL_ROOT_PATH}producto.csv"
operador_csv = f"{MODEL_ROOT_PATH}operador.csv"
municipio_csv = f"{MODEL_ROOT_PATH}municipio.csv"
localidad_csv = f"{MODEL_ROOT_PATH}localidad.csv"
letra_csv = f"{MODEL_ROOT_PATH}letra.csv"
departamento_csv = f"{MODEL_ROOT_PATH}departamento.csv"
certificadora_csv = f"{MODEL_ROOT_PATH}certificadora.csv"

# Utilizamos los datasets limpios

df_salarios = pd.read_csv(registro_salarial_csv, encoding="utf-8")
df_sector = pd.read_csv(sector_csv, encoding="utf-8")
df_prod_operador = pd.read_csv(producto_por_operador_csv, encoding="utf-8")
df_producto = pd.read_csv(producto_csv, encoding="utf-8")
df_operador = pd.read_csv(operador_csv, encoding="utf-8")
df_municipio = pd.read_csv(municipio_csv, encoding="utf-8")
df_localidad = pd.read_csv(localidad_csv, encoding="utf-8")
df_letra = pd.read_csv(letra_csv, encoding="utf-8")
df_departamento = pd.read_csv(departamento_csv, encoding="utf-8")
df_certificadora = pd.read_csv(certificadora_csv, encoding="utf-8")
df_provincia = pd.read_csv(provincia_csv, encoding="utf-8")
df_categoria = pd.read_csv(categoria_csv, encoding="utf-8")


#%%----------------------------------------------------------------
"""

Responder las siguientes consultas a través de consultas SQL:

i) ¿Existen provincias que no presentan Operadores Orgánicos Certificados?
¿En caso de que sí, cuántas y cuáles son?
"""

consultaSQL = """
                   SELECT DISTINCT
                    count(*) AS cant_operadores, 
                    prov.nombre AS provincia
                   FROM df_operador oper
                   INNER JOIN df_localidad loc
                   ON oper.localidad_id=loc.id
                   INNER JOIN df_departamento depto
                   ON depto.id = loc.departamento_id
                   RIGHT OUTER JOIN df_provincia prov
                   ON prov.id = depto.provincia_id 
                   GROUP BY provincia
              """
operadores_por_provincia = sql ^ consultaSQL


print("El listado de cantidad de operadores por provincia")
print(operadores_por_provincia)
#%%----------------------------------------------------------------
"""
Como es conocido, hay 24 provincias en Argentina, que es el tamaño del dataframe de 
operadores por provincia conseguido. Por lo tanto, no existen provincias sin operadores. 
Lo comprobamos.
"""

consultaSQL = """
                   SELECT DISTINCT
                    count(*) AS cant_operadores, 
                    prov.nombre AS provincia
                   FROM df_operador oper
                   INNER JOIN df_localidad loc
                   ON oper.localidad_id=loc.id
                   INNER JOIN df_departamento depto
                   ON depto.id = loc.departamento_id
                   RIGHT OUTER JOIN df_provincia prov
                   ON prov.id = depto.provincia_id 
                   GROUP BY provincia
                   HAVING cant_operadores = 0
              """

print("El listado de provincias sin operadores")
print(sql ^ consultaSQL)
#%%----------------------------------------------------------------
"""
ii) ¿Existen departamentos que no presentan Operadores Orgánicos
Certificados? ¿En caso de que sí, cuántos y cuáles son?

"""

consultaSQL = """
                   SELECT DISTINCT
                    count(oper.id) AS cant_operadores, 
                    depto.nombre AS departamento
                   FROM df_operador oper
                   INNER JOIN df_localidad loc
                   ON oper.localidad_id=loc.id
                   RIGHT OUTER JOIN df_departamento depto
                   ON depto.id = loc.departamento_id
                   GROUP BY departamento
                   HAVING cant_operadores = 0
              """

operadores_por_departamento = sql ^ consultaSQL

print(
    f"La cantidad de departamentos sin operadores es {operadores_por_departamento.shape[0]}"
)

print("El listado de departamentos sin operadores es")
print(operadores_por_departamento["departamento"])

departamentos_sin_operadores = list(operadores_por_departamento["departamento"])
departamentos_sin_operadores = ', '.join(departamentos_sin_operadores)
print(departamentos_sin_operadores)
#%%----------------------------------------------------------------
"""iii) ¿Cuál es la actividad que más operadores tiene?"""

consultaSQL = """
                   SELECT
                    MAX(cant_operadores) as maximo_operadores,
                    ANY_VALUE(descripcion) as descripcion,
                    ANY_VALUE(clae2) AS clae2
                   FROM (
                     SELECT 
                      count(oper.id) as cant_operadores,
                      sector.clae2_desc as descripcion,
                      sector.clae2 AS clae2
                     FROM df_operador oper
                     RIGHT OUTER JOIN df_sector sector
                     ON oper.clae2 = sector.clae2
                     GROUP BY sector.clae2, sector.clae2_desc  
                   )
              """

actividad_con_mas_operadores = sql ^ consultaSQL

actividad_con_mas_operadores
#%%----------------------------------------------------------------
"""iv) ¿Cuál fue el salario promedio de esa actividad en 2022? (si hay varios
registros de salario, mostrar el más actual de ese año)
"""

consultaSQL = """
                   SELECT
                    ROUND(AVG(salario), 2) as promedio,
                   FROM (
                     SELECT 
                      w_median AS salario,
                     FROM df_salarios reg_salarial
                     WHERE clae2 = 10 AND fecha LIKE '2022-12-01'
                   )
              """

prom_salarial_2022 = sql ^ consultaSQL

prom_salarial_2022
#%%----------------------------------------------------------------
"""
v) ¿Cuál es el promedio anual de los salarios en Argentina y cual es su
desvío?, ¿Y a nivel provincial? ¿Se les ocurre una forma de que sean
comparables a lo largo de los años? ¿Necesitarían utilizar alguna fuente de
datos externa secundaria? ¿Cuál?

En respuesta a las últimas 3 preguntas, podríamos decir que una manera de que 
los promedios salariales sean comparables a lo largo de los años es incorporando 
al análisis una tabla que indique la inflación interanual para cada uno de los años 
considerados. Con esa database a mano, se podría "actualizar" todos los salarios 
anteriores a 2022 multiplicándolos sucesivamente por las inflaciones interanuales 
que haya habido en los años comprendidos entre el año considerado y 2022. De esta manera, 
los sueldos serían, en algún sentido, equiparables a lo largo de los años. Igualmente, 
este análisis tampoco es del todo completo, dado que, lamentablemente, las actualizaciones 
salariales no suelen seguir la inflación. Algo que se podría hacer para verificar 
efectivamente si esto sucede o no, es conseguir datos históricos de la inflación interanual, 
y analizar la correlación que existe entre la proporción de aumento salarial interanual y 
la inflación interanual: una correlación positiva indicaría un aumento salarial por encima 
de la inflación, mientras que una correlación negativa correspondería a un aumento salarial 
por debajo de la inflación.
"""

consultaSQL = """
                   SELECT
                    ROUND(AVG(w_median), 2) as promedio,
                    CASE
                        WHEN fecha LIKE '2014-%' THEN '2014'
                        WHEN fecha LIKE '2015-%' THEN '2015'
                        WHEN fecha LIKE '2016-%' THEN '2016'
                        WHEN fecha LIKE '2017-%' THEN '2017'
                        WHEN fecha LIKE '2018-%' THEN '2018'
                        WHEN fecha LIKE '2019-%' THEN '2019'
                        WHEN fecha LIKE '2020-%' THEN '2020'
                        WHEN fecha LIKE '2021-%' THEN '2021'
                        WHEN fecha LIKE '2022-%' THEN '2022'
                    END AS anio
                   FROM df_salarios
                   GROUP BY anio
                   ORDER BY anio
              """

prom_salarial_nacional = sql ^ consultaSQL

prom_salarial_nacional
#%%----------------------------------------------------------------
consultaSQL = """
                   SELECT
                    ROUND(STDDEV(promedio), 2) as desvio,
                   FROM prom_salarial_nacional
              """
desvio_promedio_nacional = sql ^ consultaSQL

desvio_promedio_nacional

consultaSQL = """
                   SELECT
                    ROUND(AVG(w_median), 2) as promedio,
                    CASE
                        WHEN fecha LIKE '2014-%' THEN '2014'
                        WHEN fecha LIKE '2015-%' THEN '2015'
                        WHEN fecha LIKE '2016-%' THEN '2016'
                        WHEN fecha LIKE '2017-%' THEN '2017'
                        WHEN fecha LIKE '2018-%' THEN '2018'
                        WHEN fecha LIKE '2019-%' THEN '2019'
                        WHEN fecha LIKE '2020-%' THEN '2020'
                        WHEN fecha LIKE '2021-%' THEN '2021'
                        WHEN fecha LIKE '2022-%' THEN '2022'
                    END AS anio,
                    prov.nombre as provincia_nombre
                   FROM df_salarios as salarios
                   INNER JOIN df_operador oper
                   ON oper.clae2 = salarios.clae2
                   INNER JOIN df_localidad loc
                   ON oper.localidad_id=loc.id
                   INNER JOIN df_departamento depto
                   ON depto.id = loc.departamento_id
                   RIGHT OUTER JOIN df_provincia prov
                   ON prov.id = depto.provincia_id 
                   GROUP BY anio, prov.nombre
                   ORDER BY anio ASC, prov.nombre ASC
              """

prom_salarial_provincial = sql ^ consultaSQL

prom_salarial_provincial

#%%
consultaSQL = """
                   SELECT
                    ROUND(STDDEV(promedio), 2) as desvio,
                   FROM prom_salarial_provincial
              """
desvio_promedio_provincial = sql ^ consultaSQL

desvio_promedio_provincial

#%%----------------------------------------------------------------
"""

GRAFICOS

Mostrar, utilizando herramientas de visualización, la siguiente información:

i) Cantidad de Operadores por provincia.
"""

consultaSQL = """
                   SELECT DISTINCT 
                    oper.id AS id,
                    oper.localidad_id AS localidad, 
                    loc.departamento_id AS departamento, 
                    oper.clae2,
                    prov.nombre AS provincia
                   FROM df_operador oper
                   RIGHT OUTER JOIN df_localidad loc
                   ON oper.localidad_id=loc.id
                   INNER JOIN df_departamento  dep
                   ON loc.departamento_id=dep.id
                   INNER JOIN df_provincia  prov
                   ON dep.provincia_id=prov.id
              """
operadores_localizados = sql ^ consultaSQL

consultaSQL = """
                   SELECT provincia, count(id) AS cantidad_operadores
                   FROM operadores_localizados
                   GROUP BY provincia
              """
prov_cant_oper = sql ^ consultaSQL

sns.barplot(data=prov_cant_oper, y="provincia", x="cantidad_operadores").set(
    xlabel="Cantidad de operadores", ylabel=" Provincia"
)
plt.show()
plt.close()
#%%----------------------------------------------------------------
"""ii) Boxplot, por cada provincia, donde se pueda observar la cantidad de
productos por operador.
"""

consultaSQL = """
                   SELECT 
                    operador_id AS id, 
                    count(*) AS cantidad_productos
                   FROM df_prod_operador
                   GROUP BY operador_id;
              """
operador_cant_prod = sql ^ consultaSQL

consultaSQL = """
                   SELECT DISTINCT 
                    oper.id, 
                    oper.provincia, 
                    prod.cantidad_productos
                   FROM operadores_localizados oper
                   INNER JOIN operador_cant_prod prod
                   ON oper.id=prod.id
                   ORDER BY oper.id 
              """
operador_prov_cant_prod = sql ^ consultaSQL

sns.boxplot(data=operador_prov_cant_prod, y="provincia", x="cantidad_productos").set(
    xlabel="Cantidad de productos", ylabel=" Provincia"
)
plt.show()
plt.close()
#%%----------------------------------------------------------------
"""
Encontramos con algunos outliers, principalmente por arriba. E incluso algunas 
provincias donde parecería que la distribución es uniforme, como Neuquén. Los mismos 
no nos interesan para nuestro análisis. Por lo tanto, dejaremos de lado los registros 
que estén por fuera del cuantil 0.97 (aproximadamente a 2 desvíos de la media, por arriba).
"""

operador_prov_cant_prod["cantidad_productos"].quantile(0.97)

consultaSQL = """
                   SELECT DISTINCT *
                   FROM operador_prov_cant_prod
                   WHERE cantidad_productos<8
              """
operador_prov_cant_prod = sql ^ consultaSQL

sns.boxplot(data=operador_prov_cant_prod, y="provincia", x="cantidad_productos").set(
    xlabel="Cantidad de productos", ylabel="Provincia"
)
plt.show()
plt.close()
#%%----------------------------------------------------------------

"""
iii) Relación entre cantidad de emprendimientos certificados de cada provincia y 
el salario promedio en dicha provincia (para la actividad) en el año 2022. En caso de 
existir más de un salario promedio para ese año, mostrar el último del año 2022.
"""

consultaSQL = """
                   SELECT 
                    prov.nombre as provincia, 
                    sec.clae2_desc as actividad, 
                    sal.clae2 as clae2, 
                    ROUND(AVG(w_median), 2) as salario_promedio
                   FROM df_salarios sal
                   INNER JOIN df_departamento dep
                   ON sal.departamento_id = dep.id
                   INNER JOIN df_provincia prov
                   ON dep.provincia_id = prov.id
                   INNER JOIN df_sector sec
                   ON sec.clae2 = sal.clae2
                   WHERE sal.fecha = '2022-12-01'
                   GROUP BY prov.nombre, sal.clae2, actividad
              """
prov_clae2_salario = sql ^ consultaSQL

consultaSQL = """
                   SELECT provincia, count(distinct id) AS cantidad_operadores, clae2
                   FROM operadores_localizados
                   GROUP BY provincia, clae2
              """
prov_clae2_cant_oper = sql ^ consultaSQL

consultaSQL = """
                   SELECT 
                    oper.provincia as provincia, 
                    sal.actividad as actividad, 
                    salario_promedio, 
                    cantidad_operadores
                   FROM prov_clae2_salario sal
                   INNER JOIN prov_clae2_cant_oper oper
                   ON sal.provincia = oper.provincia AND sal.clae2 = oper.clae2
              """
cant_operadores_salario_actividad_prov = sql ^ consultaSQL

sns.scatterplot(
    data=cant_operadores_salario_actividad_prov,
    x="salario_promedio",
    y="cantidad_operadores",
    hue="actividad",
).set(xlabel="Salario promedio", ylabel="Cantidad de operadores")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.75))
plt.show()
plt.close()
#%%----------------------------------------------------------------
"""Realizamos un nuevo gráfico descartando los casos que posean más de 100 operadores."""

consultaSQL = """
                   SELECT 
                    oper.provincia as provincia, 
                    sal.actividad as actividad, 
                    salario_promedio, 
                    cantidad_operadores
                   FROM prov_clae2_salario sal
                   INNER JOIN prov_clae2_cant_oper oper
                   ON sal.provincia = oper.provincia AND sal.clae2 = oper.clae2
                   WHERE cantidad_operadores < 100
              """
cant_operadores_salario_actividad_prov = sql ^ consultaSQL

sns.scatterplot(
    data=cant_operadores_salario_actividad_prov,
    x="salario_promedio",
    y="cantidad_operadores",
    hue="actividad",
).set(xlabel="Salario promedio", ylabel="Cantidad de operadores")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.75))
plt.show()
plt.close()

#%%----------------------------------------------------------------
"""
En el gráfico de arriba, cada punto representa la relación entre salario promedio y 
cantidad de operadores para cada par `provincia-actividad`.
"""

consultaSQL = """
                   SELECT
                     COUNT(DISTINCT oper.id) as cantidad_operadores,
                     oper.provincia AS provincia,
                     ROUND(AVG(salarios.w_median), 2) AS salario_promedio
                   FROM df_salarios salarios
                   INNER JOIN operadores_localizados oper
                   ON oper.clae2 = salarios.clae2
                   WHERE salarios.fecha = '2022-12-01'
                   GROUP BY provincia
                   ORDER BY provincia
              """
cant_operadores_salario_prov = sql ^ consultaSQL

sns.scatterplot(
    data=cant_operadores_salario_prov,
    x="salario_promedio",
    y="cantidad_operadores",
    hue="provincia",
).set(xlabel="Salario promedio", ylabel="Cantidad de operadores")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05))
plt.show()
plt.close()
#%%----------------------------------------------------------------
"""Ahora por actividad"""

consultaSQL = """
                   SELECT
                     COUNT(DISTINCT oper.id) as cantidad_operadores,
                     sector.clae2_desc AS actividad,
                     ROUND(AVG(salarios.w_median), 2) AS salario_promedio
                   FROM df_salarios salarios
                   INNER JOIN operadores_localizados oper
                   ON oper.clae2 = salarios.clae2
                   INNER JOIN df_sector sector
                   ON oper.clae2 = sector.clae2
                   WHERE salarios.fecha = '2022-12-01'
                   GROUP BY actividad
              """
cant_operadores_salario_prov = sql ^ consultaSQL

sns.scatterplot(
    data=cant_operadores_salario_prov,
    x="salario_promedio",
    y="cantidad_operadores",
    hue="actividad",
).set(xlabel="Salario promedio", ylabel="Cantidad de operadores")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05))
plt.show()
plt.close()
#%%----------------------------------------------------------------
"""
Este conjunto de gráficos nos permite observar que, en términos generales, a medida 
que disminuye la cantidad de operadores, el salario promedio aumenta. Da una idea de 
que cuantos menos operadores haya para una actividad, más dinero obtendrán.

iv) ¿Cuál es la distribución de los salarios promedio en Argentina? Realicen un 
violinplot de los salarios promedio por provincia. Grafiquen el último ingreso 
medio por provincia.
"""

consultaSQL = """
                   SELECT 
                    sal.fecha AS fecha, 
                    prov.nombre AS provincia,
                    sal.w_median AS salario, 
                    sector.clae2_desc AS actividad,
                    prov.nombre AS provincia
                   FROM df_salarios sal
                   INNER JOIN df_sector sector
                   ON sector.clae2 = sal.clae2
                   INNER JOIN df_departamento dep
                   ON sal.departamento_id = dep.id
                   INNER JOIN df_provincia prov
                   ON dep.provincia_id=prov.id
              """

salarios_prov = sql ^ consultaSQL

# ¿para qué fecha estan todas las provincias?

consultaSQL = """
                   SELECT  DISTINCT fecha, count(DISTINCT provincia) AS cant_provincia
                   FROM salarios_prov
                   GROUP BY fecha
                   HAVING cant_provincia = 24
                   ORDER BY fecha DESC
              """

provs_por_fecha = sql ^ consultaSQL
provs_por_fecha
#%%----------------------------------------------------------------
"""
Resulta que para la fecha 2022-12-01 las 24 provincias cargaron datos, entonces usaremos 
esa fecha. También, para estudiar la distribución, nos interesaría descartar los salarios 
que sean anómalos.
"""

df_salarios.loc[df_salarios["fecha"] == "2022-12-01", "w_median"].quantile(0.97)

consultaSQL = """
                   SELECT provincia, salario, actividad
                   FROM salarios_prov
                   WHERE salarios_prov.fecha='2022-12-01' AND salarios_prov.salario<716000
              """

ult_sal_prov = sql ^ consultaSQL

ult_sal_prov

sns.violinplot(data=ult_sal_prov, x="salario", y="provincia")
plt.show()
plt.close()
#%%----------------------------------------------------------------
consultaSQL = """
                   SELECT provincia, ROUND(AVG(salario), 2) AS promedio_provincia
                   FROM ult_sal_prov
                   GROUP BY provincia
              """

prov_sal_prom = sql ^ consultaSQL

sns.barplot(data=prov_sal_prom, x="promedio_provincia", y="provincia")
plt.show()
plt.close()
