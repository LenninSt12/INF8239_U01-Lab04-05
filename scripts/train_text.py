import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix

load_dotenv()
dataset_path = os.getenv("DATASET_PATH", "data/raw/16k_Movies.csv")
text_col = os.getenv("TEXT_COLUMN", "Description")
target_col = os.getenv("TARGET_COLUMN", "Genres")

print("Cargando dataset...")
df = pd.read_csv(dataset_path)

df = df.dropna(subset=[text_col, target_col]).copy()
df['Primary_Genre'] = df[target_col].astype(str).apply(lambda x: x.split(',')[0].strip() if pd.notnull(x) else "Unknown")

class_counts = df['Primary_Genre'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['Primary_Genre'].isin(valid_classes)].copy()

X = df[text_col].astype(str)
y = df['Primary_Genre']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipelines = {
    "Baseline_Dummy": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', DummyClassifier(strategy='prior', random_state=42))
    ]),
    "Complement_NB": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', ComplementNB())
    ]),
    "Logistic_Regression": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
}

resultados = []
best_f1 = -1
best_model = None
best_name = ""

print("\n--- ENTRENANDO MODELOS ---")
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    resultados.append({"Model": name, "F1_Macro": f1_macro})
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_model = pipeline
        best_name = name

os.makedirs("reports", exist_ok=True)
pd.DataFrame(resultados).to_csv("reports/text_metrics.csv", index=False)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/text_model.joblib")

y_pred_best = best_model.predict(X_test)

# Generar el archivo error_analysis.csv con las predicciones erróneas
df_errors = pd.DataFrame({
    "Description": X_test,
    "True_Genre": y_test,
    "Predicted_Genre": y_pred_best
})
df_errors = df_errors[df_errors["True_Genre"] != df_errors["Predicted_Genre"]]
df_errors.head(50).to_csv("reports/error_analysis.csv", index=False)
print("Archivo reports/error_analysis.csv generado con éxito.")
