import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Clasificador de Géneros Cinematográficos",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Clasificador de Películas por Sinopsis")
st.markdown("Laboratorio 5 - Modelo predictivo de texto basado en **Complement Naive Bayes**.")

model_path = "models/text_model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("No se encontró el modelo entrenado en `models/text_model.joblib`. Ejecute primero el script de entrenamiento.")
else:
    st.sidebar.header("Acerca del Modelo")
    st.sidebar.info(
        "Este modelo utiliza TF-IDF y Complement Naive Bayes para clasificar "
        "el género principal de una película a partir de su sinopsis en inglés.\n\n"
        "⚠️ **Alcance y Limitaciones:** El corpus está limitado a cine occidental y sinopsis en inglés. "
        "Los resultados son predicciones estadísticas condicionadas por los datos, no verdades absolutas."
    )

    user_input = st.text_area(
        "Ingrese la sinopsis de la película (en inglés recomendado para mayor precisión):",
        placeholder="Ej: A brave hero fights against insurmountable odds to save the city..."
    )

    if st.button("Predecir Género"):
        if user_input.strip() == "":
            st.warning("Por favor, ingrese un texto válido para realizar la predicción.")
        else:
            prediction = model.predict([user_input])
            st.success(f"**Género Predicho:** {prediction[0]}")
            st.caption("Nota: La etiqueta mostrada es una estimación estadística basada en patrones de texto y no sustituye una evaluación profunda.")
