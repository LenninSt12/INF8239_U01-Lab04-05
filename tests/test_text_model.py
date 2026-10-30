import os
import joblib
import pytest

def test_model_file_exists():
    assert os.path.exists("models/text_model.joblib"), "El archivo del modelo joblib no existe."

def test_model_prediction():
    model_path = "models/text_model.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Modelo no encontrado aún.")
    
    model = joblib.load(model_path)
    # Probar con una sinopsis de ejemplo en inglés (acorde al corpus del dataset)
    sample_text = ["A thrilling action movie with brave heroes and intense battles."]
    prediction = model.predict(sample_text)
    
    assert len(prediction) == 1, "La predicción debe devolver una etiqueta para la muestra."
    assert isinstance(prediction[0], str), "La etiqueta predicha debe ser una cadena de texto."
