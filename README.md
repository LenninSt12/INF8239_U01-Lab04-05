#  Clasificación de Texto Cinematográfico (Laboratorios 04 y 05)

Este repositorio contiene el pipeline completo de Machine Learning (Ingeniería de Datos, Auditoría, Entrenamiento y Despliegue) para predecir el género principal de películas basándose en sus sinopsis.

**Autor:** Lennen Santos Mejía  
**Programa:** Maestría en Ciencia de Datos e Inteligencia Artificial (Ciencia de Datos II) - UASD

---

##  Origen y Licencia de los Datos
El dataset utilizado en este proyecto (`16k_Movies.csv`) fue obtenido desde la plataforma Kaggle bajo el título **[16000+ Movies 1910-2024 (Metacritic)]**. 
Los datos se distribuyen explícitamente bajo la licencia **CC0: Public Domain**, lo que garantiza su libre uso, modificación y distribución sin restricciones para esta experimentación académica e investigación predictiva.

---

##  Auditoría y Preprocesamiento

### 1. Justificación de la Variable Objetivo (Género Principal)
En las bases de datos cinematográficas, las películas suelen presentar etiquetas múltiples (ej. *"Action, Adventure, Sci-Fi"*). En este proyecto, se extrajo estrictamente el primer término listado (`x.split(',')[0]`). Esta decisión técnica obedece a la convención de que el primer género listado es el más representativo o dominante de la obra. Al aislar este elemento, transformamos un problema complejo de clasificación multietiqueta (*multi-label*) en uno de clasificación multiclase (*multi-class*), simplificando la convergencia de los modelos lineales sin degradar la intención analítica del benchmark.

### 2. Prevención de Fuga de Datos (Data Leakage)
Durante la auditoría del corpus, se detectaron sinopsis repetidas en el dataset original. Se implementó una purga de duplicados mediante `drop_duplicates(subset=['Description'])` **antes** de la división de datos (Train/Test Split). Esto garantizó que ninguna sinopsis de entrenamiento se filtrara al conjunto de prueba, evitando métricas de evaluación artificialmente infladas.

---

##  Modelado y Análisis de Errores

El modelo seleccionado fue **Complement Naive Bayes** utilizando representación vectorial TF-IDF (F1-Macro de 0.2222), superando al baseline Dummy Classifier y a la Regresión Logística, especialmente en su manejo de datos desbalanceados. Las métricas de todas las clases se exportaron exitosamente en `reports/classification_report_full.txt`.

### Análisis de Errores de Clasificación (20 Casos Analizados)
Tras evaluar las predicciones en el conjunto de prueba (`reports/error_analysis.csv`), se analizaron 20 casos cualitativos donde el modelo falló. Se identificaron cuatro patrones principales que explican las limitaciones del algoritmo TF-IDF:

1. **Ambigüedad léxica (Palabras clave engañosas):** El modelo asocia ciertas palabras con géneros específicos, ignorando el contexto irónico. Una sinopsis de *Comedia* que menciona *"chainsaw"* y *"battle"* fue predicha como *Acción*. Otra de *Fantasía* con términos como *"sinister black hole"* fue lógicamente predicha como *Horror*.
2. **Fronteras difusas (Drama vs. Documental):** El modelo sufre para distinguir narrativas basadas en hechos reales de los documentales puros. Dramas históricos y biografías (ej. Notorious B.I.G., David Bowie) son clasificadas como *Documental* al detectarse terminología periodística u objetiva.
3. **Tono relacional interpretado como Comedia:** Los dramas con temáticas interpersonales ligeras (cenas de amigos, bodas, crisis de pareja) tienden a ser clasificados como *Comedia*, ya que palabras como *"lovers"*, *"wine"* o *"friends reunite"* dominan las comedias del conjunto de entrenamiento.
4. **Subjetividad de la etiqueta real (Ground Truth):** Al forzar el modelo a predecir solo el género principal, se penalizan aciertos semánticos válidos (ej. un thriller de policías etiquetado estrictamente como *Acción* en el dataset original, pero predicho correctamente en esencia como *Crimen* por el modelo).

---

##  Instrucciones de Reproducción Local

Este proyecto utiliza `uv` para garantizar la reproducibilidad exacta del entorno.

**1. Sincronizar el entorno e instalar dependencias:**

uv python install 3.12
uv sync

**2. Ejecutar la validación y el contrato de datos:**

uv run pytest tests/test_data_contract.py -v

**3. Entrenar el modelo (genera reportes y guarda el .joblib):**

uv run python scripts/train_text.py

**4. Lanzar la aplicación interactiva (Streamlit):**

uv run streamlit run app/streamlit_app.py