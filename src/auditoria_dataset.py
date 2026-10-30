import kagglehub
import pandas as pd
import os
import shutil

# 1. Trazabilidad y Descarga
print("Descargando corpus público desde Kaggle...")
fuente_original = "kashifsahil/16000-movies-1910-2024-metacritic"
path_kaggle = kagglehub.dataset_download(fuente_original)

# 2. Configuración de Rutas Relativas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_datos = os.path.join(BASE_DIR, "datos")
archivo_csv = [f for f in os.listdir(path_kaggle) if f.endswith('.csv')][0]
ruta_local_csv = os.path.join(ruta_datos, archivo_csv)
shutil.copy(os.path.join(path_kaggle, archivo_csv), ruta_local_csv)
print(f"Dataset guardado en: {ruta_local_csv}")

# 3. Auditoría Inicial
df = pd.read_csv(ruta_local_csv)
print(f"\nTotal de registros originales: {len(df)}")

duplicados = df.duplicated().sum()
if duplicados > 0:
    print(f"Se encontraron y eliminaron {duplicados} registros duplicados.")
    df = df.drop_duplicates()

print(f"Total de registros válidos: {len(df)}")
print("\nColumnas disponibles:")
print(df.columns.tolist())