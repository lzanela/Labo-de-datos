import pandas as pd
from inline_sql import sql, sql_val

padrones = pd.read_csv("padron-de-operadores-organicos-certificados-MODIFICADO1.csv", encoding = "utf-8")

consulta_registros =  """
                SELECT DISTINCT count()
                FROM padrones
            """
            
registros = sql ^ consulta_registros

print(registros)

#%% Ejercicio j) i)
consulta_cant_de_ops =  """
                SELECT DISTINCT provincia, count(*) AS cantidad_de_operadores
                FROM padrones
                GROUP BY provincia
            """

cant_de_ops = sql ^ consulta_cant_de_ops
print(cant_de_ops)