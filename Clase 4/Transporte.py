import numpy as np
import pandas as pd

archivo = 'EncuestaMovilidadRespuestas.csv'
df = pd.read_csv(archivo)

transportes_usados = df[df.columns[-1]].value_counts()
mas_usado = (transportes_usados.idxmax(), transportes_usados.max())