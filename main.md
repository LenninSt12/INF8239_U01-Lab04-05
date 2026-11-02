::: titlepage
[Universidad Autónoma de Santo Domingo]{.smallcaps}

[Facultad de Ciencias -- Escuela de Computación]{.smallcaps}

**Informe Técnico: Procesamiento, Auditoría y Clasificación de Texto
Cinematográfico**

*Maestría en Ciencia de Datos e Inteligencia Artificial\
Ciencia de Datos II*

**Autor:** Lennen Santos Mejía

**Fecha:** Octubre de 2026
:::

# Introducción y Objetivos

El presente informe documenta el desarrollo completo de los laboratorios
de ingeniería de datos y modelado predictivo de texto (\*\*Lab 04 y Lab
05\*\*). El objetivo central consistió en diseñar, auditar, entrenar,
evaluar y desplegar localmente un pipeline de aprendizaje automático
capaz de clasificar el género principal de películas a partir del
análisis de sus sinopsis textuales utilizando el conjunto de datos
`16k_Movies.csv`.

## Repositorio Oficial del Proyecto

El código fuente completo, scripts de procesamiento, modelos
serializados, pruebas unitarias y reportes de esta investigación se
encuentran alojados y versionados en el siguiente repositorio oficial de
GitHub:

::: center
<https://github.com/LenninSt12/INF8239_U01-Lab04-05>
:::

# Laboratorio 04: Configuración, Auditoría y Contrato de Datos

## Entorno y Reproducibilidad

El proyecto se estructuró bajo un entorno virtual gestionado con `uv`,
garantizando la compatibilidad estricta de versiones para bibliotecas
críticas como `scikit-learn`, `pandas` y `pytest`.

<figure id="fig:env_uv" data-latex-placement="htbp">
<img src="./2. Preparar el entorno local con uv.png"
style="width:90.0%" />
<figcaption>Preparación y sincronización del entorno local con
<code>uv</code>.</figcaption>
</figure>

## Auditoría del Corpus y Contrato de Datos

Mediante la ejecución de scripts automatizados de validación, se
confirmó la integridad del dataset de sinopsis:

- **Dimensiones iniciales:** 16,290 filas y 10 columnas.

- **Valores nulos:** 0 nulos detectados en la columna de texto
  descriptivo (`Description`).

- **Estadísticas de longitud de texto:** Mediana de 351 caracteres,
  percentil 90 de 731 y percentil 99 de 1168 caracteres.

- **Pruebas de contratos:** Las pruebas unitarias ejecutadas mediante
  `pytest` sobre el contrato de datos superaron satisfactoriamente las
  validaciones estructurales de esquemas.

- **Prevención de Fuga de Datos (Data Leakage):** Durante la auditoría
  profunda se detectaron sinopsis repetidas. Se aplicó una purga de
  duplicados mediante `drop_duplicates()` antes del particionado para
  evitar métricas infladas y asegurar la integridad del conjunto de
  prueba.

- **Licencia de los Datos:** El conjunto de datos original
  (`16k_Movies.csv`) proviene de bases de datos cinematográficas de
  acceso público (tipo IMDB/TMDB) y se ampara bajo licencias de uso
  abierto para fines académicos y de investigación (ej. CC0: Public
  Domain o DbCL), permitiendo su análisis predictivo en este
  laboratorio.

<figure id="fig:descargar_verificar" data-latex-placement="htbp">
<img src="./6. Descargar y verificar.png" style="width:90.0%" />
<figcaption>Verificación y estructura inicial del proyecto en Visual
Studio Code.</figcaption>
</figure>

<figure id="fig:auditoria_corpus" data-latex-placement="htbp">
<img src="./6. Descargar y verificar 2.png" style="width:90.0%" />
<figcaption>Reporte de auditoría del corpus y estadísticas descriptivas
de longitud.</figcaption>
</figure>

<figure id="fig:contrato_datos" data-latex-placement="htbp">
<img src="./8. Ejecutar el contrato de datos.png" style="width:90.0%" />
<figcaption>Ejecución exitosa de las pruebas del contrato de datos con
<code>pytest</code>.</figcaption>
</figure>

<figure id="fig:git_lab04" data-latex-placement="htbp">
<img src="./9. Registrar con Git.png" style="width:90.0%" />
<figcaption>Inicialización y control de versiones iniciales en
Git.</figcaption>
</figure>

<figure id="fig:portabilidad_1" data-latex-placement="htbp">
<img src="./10. Comprobar portabilidad local-nube part1.png"
style="width:90.0%" />
<figcaption>Configuración de portabilidad y generación de dependencias
para la nube (Parte 1).</figcaption>
</figure>

<figure id="fig:portabilidad_2" data-latex-placement="htbp">
<img src="./10. Comprobar portabilidad local-nube part2.png"
style="width:90.0%" />
<figcaption>Configuración de portabilidad y generación de dependencias
para la nube (Parte 2).</figcaption>
</figure>

## Justificación de la Variable Objetivo (Género Principal)

En bases de datos cinematográficas, las películas suelen presentar
etiquetas múltiples (ej. \"Action, Adventure, Sci-Fi\"). En este
proyecto, se extrajo estrictamente el primer término listado
(`x.split(’,’)[0]`). Esta decisión técnica obedece a la convención de
que el primer género listado es el más representativo o dominante de la
obra. Al aislar este elemento, transformamos un problema complejo de
clasificación multietiqueta (*multi-label*) en uno de clasificación
multiclase (*multi-class*), simplificando la convergencia de modelos
lineales sin degradar significativamente la intención analítica del
benchmark.

# Laboratorio 05: Entrenamiento y Evaluación de Modelos de Texto

## Estrategia de Modelado y Selección

Se evaluaron tres arquitecturas de clasificación utilizando
representación vectorial basada en `TfidfVectorizer` (con un límite de
5,000 características y eliminación de palabras vacías en inglés):

1.  **Baseline Dummy Classifier:** Estrategia de prioridades base
    ($\text{F1-macro} = 0.0234$).

2.  **Logistic Regression:** Modelo lineal generalizado
    ($\text{F1-macro} = 0.2105$).

3.  **Complement Naive Bayes:** Algoritmo probabilístico optimizado para
    conjuntos desbalanceados ($\text{F1-macro} = 0.2222$, seleccionado
    como el mejor modelo).

El modelo seleccionado se serializó de manera robusta en el archivo
`models/text_model.joblib`.

<figure id="fig:entrenar_pipelines" data-latex-placement="htbp">
<img src="./3. Entrenar los tres pipelines.png" style="width:90.0%" />
<figcaption>Entrenamiento de los tres pipelines y métricas de
desempeño.</figcaption>
</figure>

## Análisis de Métricas y Matriz de Confusión

Al evaluar el reporte de clasificación y la matriz de confusión generada
(`reports/confusion_text.png`), se identificaron los siguientes patrones
críticos:

- **Aciertos robustos:** Las categorías principales del corpus como
  *Comedy* (448 aciertos), *Documentary* (405 aciertos), *Drama* (393
  aciertos) y *Action* (337 aciertos) concentraron la mayor cantidad de
  aciertos debido al volumen masivo de datos de entrenamiento.

- **Clases con mayor dificultad:** Las categorías minoritarias (como
  \*Fantasy\*, \*Adventure\* y \*Animation\*) presentaron un recall
  sumamente bajo o cercano a cero debido al desbalance estructural de
  las clases.

- **Tipo de error más frecuente:** La ambigüedad semántica y el cruce
  temático, evidenciándose una confusión bidireccional masiva entre las
  clases *Drama* y *Comedy* (174 casos y 138 casos cruzados), dado que
  muchas sinopsis integran simultáneamente elementos serios y
  humorísticos.

<figure id="fig:matriz_confusion" data-latex-placement="htbp">
<img src="./3. Entrenar los tres pipelines part2.png"
style="width:90.0%" />
<figcaption>Matriz de confusión detallada (Top 10 Clases) del modelo
Complement Naive Bayes.</figcaption>
</figure>

## Análisis de Errores (*Error Analysis*)

Se exportó el reporte detallado en `reports/error_analysis.csv`. Al
auditar las muestras erróneas, se categorizaron los fallos principales
bajo los siguientes criterios:

- **Ambigüedad de género:** Obras con enfoques híbridos difíciles de
  delimitar en una sola etiqueta principal.

- **Etiquetas discutibles:** Clasificaciones originales en la fuente que
  narrativamente podrían encajar mejor en categorías secundarias.

- **Falta de señal léxica distintiva:** Descripciones cortas que no
  proveen suficientes pesos TF-IDF diferenciadores.

### Análisis Cualitativo Específico (20 Casos Analizados) {#análisis-cualitativo-específico-20-casos-analizados .unnumbered}

Para profundizar en las causas, se analizaron 20 errores de
clasificación documentados en el conjunto de prueba. Se identificaron
cuatro patrones concretos que explican los fallos del modelo TF-IDF:

- **Ambigüedad léxica (Palabras clave engañosas):** El modelo asocia
  ciertas palabras con géneros específicos, ignorando el contexto
  irónico. Una sinopsis de *Comedia* que menciona "chainsaw" y "battle"
  fue predicha como *Acción*. Otra de *Fantasía* con términos como
  "sinister black hole" fue lógicamente predicha como *Horror*.

- **Fronteras difusas (Drama vs. Documental):** El modelo sufre para
  distinguir narrativas basadas en hechos reales de los documentales
  puros. Dramas históricos y biografías (ej. Notorious B.I.G., David
  Bowie) son clasificadas como *Documental* al detectarse terminología
  periodística u objetiva.

- **Tono relacional interpretado como Comedia:** Los dramas con
  temáticas interpersonales ligeras (cenas de amigos, bodas, crisis de
  pareja) tienden a ser clasificados como *Comedia*, ya que palabras
  como "lovers", "wine" o "friends reunite" dominan las comedias del
  conjunto de entrenamiento.

- **Subjetividad de la etiqueta real (Ground Truth):** Al forzar el
  modelo a predecir solo el género principal, se penalizan aciertos
  semánticos válidos (ej. un thriller de policías etiquetado
  estrictamente como *Acción* en el dataset original, pero predicho
  correctamente en esencia como *Crimen* por el modelo).

<figure id="fig:analizar_errores" data-latex-placement="htbp">
<img src="./6. Analizar errores.png" style="width:90.0%" />
<figcaption>Generación del reporte de análisis de errores en la
terminal.</figcaption>
</figure>

# Pruebas Unitarias y Despliegue Local (*Streamlit*)

- **Pruebas Automatizadas:** Se ejecutaron con éxito las pruebas
  unitarias con `pytest` en la suite `tests/test_text_model.py`,
  validando la existencia del artefacto y la consistencia en las
  predicciones del pipeline.

- **Despliegue Interactivo:** Se implementó una aplicación web local en
  `app/streamlit_app.py` ejecutada mediante `streamlit`, la cual permite
  ingresar sinopsis arbitrarias, consultar el género predicho y
  visualizar de forma transparente las advertencias sobre el alcance y
  limitaciones del corpus occidental en inglés.

<figure id="fig:pruebas_unitarias" data-latex-placement="htbp">
<img src="./7. Ejecutar pruebas y revisar el modelo guardado.png"
style="width:90.0%" />
<figcaption>Ejecución de pruebas unitarias con <code>pytest</code> y
validación del modelo serializado.</figcaption>
</figure>

<figure id="fig:streamlit_local" data-latex-placement="htbp">
<img src="./8. Ejecutar la aplicación local.png" style="width:90.0%" />
<figcaption>Interfaz de usuario de la aplicación local desplegada con
Streamlit.</figcaption>
</figure>

<figure id="fig:nube_1" data-latex-placement="htbp">
<img src="./9. Preparar para publicación en la nube part1.png"
style="width:90.0%" />
<figcaption>Preparación de dependencias y validación Git para la nube
(Parte 1).</figcaption>
</figure>

<figure id="fig:nube_2" data-latex-placement="htbp">
<img src="./9. Preparar para publicación en la nube part2.png"
style="width:90.0%" />
<figcaption>Generación de archivo de dependencias portables (Parte
2).</figcaption>
</figure>

# Conclusión e Interpretación Final

> *«El modelo **Complement Naive Bayes** obtuvo el mejor desempeño
> general en términos de F1-macro. Sin embargo, la clase *Adventure*
> (junto con otras categorías minoritarias) presentó el menor recall,
> con un volumen elevado de falsos negativos. En este contexto
> operativo, dicho error implica que el clasificador carece de capacidad
> suficiente para generalizar patrones en categorías con baja
> representación de muestras. Por consiguiente, proponemos aplicar
> técnicas de re-muestreo, balanceo de clases o incorporación de
> embeddings avanzados antes de liberar el sistema a un entorno de
> producción masiva».*

::: center

------------------------------------------------------------------------

\
**Lennen Santos Mejía**\
Maestría en Ciencia de Datos e Inteligencia Artificial
:::
