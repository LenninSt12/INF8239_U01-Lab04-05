import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
text_col = os.getenv("TEXT_COLUMN")
target_col = os.getenv("TARGET_COLUMN")

df = pd.read_csv(dataset_path)

print("--- REPORTE DE AUDITORÍA ---")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print(f"Nulos en texto ('{text_col}'): {df[text_col].isnull().sum()}")
print(f"Duplicados exactos: {df.duplicated().sum()}")

print(f"\nDistribución de clases ('{target_col}' - Top 5):")
print(df[target_col].value_counts().head(5))
print(f"Total de clases únicas: {df[target_col].nunique()}")

textos_validos = df[text_col].dropna().astype(str)
longitudes = textos_validos.apply(len)
print(f"\nLongitud de textos (caracteres):")
print(f"p50 (Mediana): {np.percentile(longitudes, 50):.0f}")
print(f"p90: {np.percentile(longitudes, 90):.0f}")
print(f"p99: {np.percentile(longitudes, 99):.0f}")