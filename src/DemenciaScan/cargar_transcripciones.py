"""
Este script carga transcripciones desde archivos .txt en una carpeta,
calcula métricas lingüísticas como riqueza léxica y número de palabras,
y guarda los resultados en un archivo CSV.

Dependencias: os, pandas, re
"""

import os
import pandas as pd
import re

def cargar_transcripciones_desde_carpeta(ruta_carpeta):
    """
    Carga transcripciones desde archivos .txt en la carpeta especificada.

    Args:
        ruta_carpeta (str): Ruta a la carpeta que contiene los archivos .txt.

    Returns:
        pd.DataFrame: DataFrame con columnas 'id' (nombre del archivo sin extensión) y 'texto' (contenido del archivo).
    """
    datos = []
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith(".txt"):
            ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                texto = f.read()
            datos.append({
                "id": nombre_archivo.replace(".txt", ""),
                "texto": texto
            })
    return pd.DataFrame(datos)

# Ruta a la carpeta de nuevas transcripciones para entrenar LLM
ruta_carpeta = r"DemenciaScan/TranscripcionesParaEntrenarLLM"

# Cargar los datos desde la carpeta especificada
df = cargar_transcripciones_desde_carpeta(ruta_carpeta)

# Agregar columnas para los nuevos campos (dejados como None por ahora)
df['edad'] = None
df['nivel_educacion'] = None
df['calidad_sueno'] = None
df['habitos_diarios'] = None
df['nivel_entorno_social'] = None
df['diagnostico_alzheimer'] = None

# Mostrar las primeras filas del DataFrame actualizado
print(df.head())

def riqueza_lexica(texto):
    """
    Calcula la riqueza léxica del texto como la proporción de palabras únicas sobre el total de palabras.

    Args:
        texto (str): Texto original.

    Returns:
        float: Riqueza léxica (0 si no hay palabras).
    """
    palabras = texto.split()
    return len(set(palabras)) / len(palabras) if len(palabras) > 0 else 0

# Calcular número de palabras y riqueza léxica
df['num_palabras'] = df['texto'].apply(lambda x: len(x.split()))
df['riqueza_lexica'] = df['texto'].apply(riqueza_lexica)

# Mostrar estadísticas descriptivas de las métricas
print(df[['num_palabras', 'riqueza_lexica']].describe())

# Guardar el DataFrame procesado en un archivo CSV
df.to_csv("transcripciones_procesadas.csv", index=False)
print("Archivo guardado: transcripciones_procesadas.csv")
