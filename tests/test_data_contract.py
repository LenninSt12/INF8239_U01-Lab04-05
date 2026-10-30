import os
import pandas as pd
import pytest
from dotenv import load_dotenv

# Cargar configuración
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH", "data/raw/16k_Movies.csv")
TEXT_COL = os.getenv("TEXT_COLUMN", "Description")
TARGET_COL = os.getenv("TARGET_COLUMN", "Genres")

@pytest.fixture(scope="module")
def df():
    assert os.path.exists(DATASET_PATH), f"El archivo {DATASET_PATH} no fue encontrado."
    return pd.read_csv(DATASET_PATH)

def test_columnas_requeridas_existen(df):
    columnas_esperadas = [TEXT_COL, TARGET_COL]
    for col in columnas_esperadas:
        assert col in df.columns, f"El contrato exige la columna: {col}"

def test_sin_nulos_en_texto_predictivo(df):
    nulos = df[TEXT_COL].isnull().sum()
    assert nulos == 0, f"Contrato violado: Hay {nulos} valores nulos en {TEXT_COL}."

def test_volumen_minimo_datos(df):
    minimo_requerido = 1000
    assert len(df) >= minimo_requerido, f"Contrato violado: Solo hay {len(df)} registros."

def test_tipo_datos_texto(df):
    muestra = df[TEXT_COL].dropna().iloc[0]
    assert isinstance(muestra, str), "Contrato violado: La variable predictora no es texto."