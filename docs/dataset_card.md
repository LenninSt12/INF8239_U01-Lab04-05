# Ficha del Dataset: Metacritic Movies (1910-2024)

## 1. Metadatos Generales
* **Fuente:** Kaggle (`kashifsahil/16000-movies-1910-2024-metacritic`).
* **Responsable:** Kashif Sahil (Usuario de Kaggle) / Datos extraídos de Metacritic.
* **Fecha / Versión:** 2024 (Última actualización de la fuente original).
* **Licencia:** Open Data / Dominio Público en Kaggle.
* **Uso previsto:** Entrenamiento de modelos de Procesamiento de Lenguaje Natural (NLP) para clasificación de textos (sinopsis a géneros) y análisis exploratorio del cine histórico.

## 2. Diccionario de Datos

| Columna | Tipo | Significado | Unidad / Categorías válidas |
| :--- | :--- | :--- | :--- |
| `Unnamed: 0` | Entero | Índice original del scraping. | Identificador numérico (0 a N). |
| `Title` | Cadena (String) | Título original de la película. | Texto libre. |
| `Release Date` | Cadena (String) | Fecha en que se estrenó la película. | Formato de fecha (ej. "July 16, 2010"). |
| `Description` | Cadena (String) | Sinopsis o resumen de la trama. | Texto libre en inglés (**Variable Predictora**). |
| `Rating` | Flotante | Calificación agregada de la película. | Escala decimal de 0.0 a 10.0. |
| `No of Persons Voted`| Entero | Cantidad de usuarios que calificaron. | Número entero positivo. |
| `Directed by` | Cadena (String) | Director o directores de la obra. | Texto libre (nombres propios). |
| `Written by` | Cadena (String) | Guionista o escritores. | Texto libre (nombres propios). |
| `Duration` | Cadena (String) | Tiempo de duración de la cinta. | Formato texto (ej. "2h 28m"). |
| `Genres` | Cadena (String) | Categorías cinematográficas asignadas. | Clases separadas por comas (**Variable Objetivo**). |

## 3. Ejemplos Anonimizados y Transformaciones

**Ejemplos de la estructura base:**
1. `Title`: "Space Journey", `Description`: "Un astronauta viaja al sol...", `Rating`: 8.5, `Genres`: "Sci-Fi,Adventure"
2. `Title`: "Love in Paris", `Description`: "Dos jóvenes se conocen en la torre...", `Rating`: 6.2, `Genres`: "Romance,Drama"
3. `Title`: "The Last Stand", `Description`: "Un policía retirado debe defender...", `Rating`: 7.0, `Genres`: "Action"
4. `Title`: "Laugh Out Loud", `Description`: "Un comediante intenta triunfar...", `Rating`: 5.5, `Genres`: "Comedy"
5. `Title`: "Dark Secrets", `Description`: "Un detective investiga un culto...", `Rating`: 7.8, `Genres`: "Thriller,Mystery,Drama"

**Transformaciones requeridas para el modelado:**
* **Limpieza:** Se descartará la columna `Unnamed: 0`.
* **Procesamiento de Texto:** La columna `Description` pasará por tokenización, eliminación de *stopwords* y vectorización (TF-IDF).
* **Variable Objetivo:** Dado que `Genres` contiene clases concatenadas (1,663 combinaciones únicas detectadas), se aplicará una transformación para extraer únicamente el **género principal** (la primera palabra antes de la coma) o se utilizará un `MultiLabelBinarizer` para evitar diluir el *accuracy* del clasificador.

## 4. Sesgos, Limitaciones y Usos Prohibidos
* **Sesgos previsibles:** Existe un sesgo geográfico, cultural y lingüístico significativo. El corpus está dominado por cine occidental (especialmente Hollywood) y las sinopsis están exclusivamente en idioma inglés. Las calificaciones (`Rating`) reflejan el sesgo demográfico de la base de usuarios de Metacritic.
* **Población no cubierta:** Producciones independientes fuera del circuito de distribución occidental, cortometrajes, películas en idiomas distintos al inglés (sin traducción oficial), y contenido exclusivo de plataformas cerradas de streaming locales.
* **Usos prohibidos:** Este dataset no debe utilizarse para predecir el éxito financiero (taquilla) de una obra, ya que carece de variables económicas. Tampoco debe usarse para sistemas de recomendación comercial automatizados sin aplicar antes una corrección por el sesgo de popularidad (películas antiguas o menos votadas están subrepresentadas).
## Reproducción del Proyecto
Para reproducir este proyecto, descarga el dataset "16000-movies-1910-2024-metacritic" de Kaggle, colócalo en la ruta local `data/raw/16k_Movies.csv` y define la variable `DATASET_PATH` en tu archivo `.env`.