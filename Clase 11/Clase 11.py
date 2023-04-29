# -*- coding: utf-8 -*-
import pandas as pd
from inline_sql import sql, sql_val

datosDengue = pd.read_csv("DatosDengueYZikaOriginal.csv")

#%% Implementación de la técnica GQM

"""
# Goal: determinar si hay consistencia entre 'grupo_edad_desc' y 'grupo_edad_id'.
# Question: ¿hay una correspondencia biunívoca entre las columnas 'grupo_edad_desc' y 'grupo_edad_id'?
# Métrica 1: chequear si la cantidad de valores distintos de cada columna coincide
"""

# Métrica 1

consultaIdDistintos = """
                        SELECT DISTINCT count()
                        FROM (SELECT DISTINCT grupo_edad_id
                              FROM datosDengue)
                      """
consultaDescDistintos = """
                        SELECT DISTINCT count()
                        FROM (SELECT DISTINCT grupo_edad_desc
                              FROM datosDengue)
                        """

cant_distinct_id = sql ^ consultaIdDistintos
cant_distinct_desc = sql ^ consultaDescDistintos

metrica1 = cant_distinct_id == cant_distinct_desc
print(metrica1)
