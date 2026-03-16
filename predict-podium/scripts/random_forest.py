from pathlib import Path
# Data Processing
import pandas as pd
import numpy as np
from preprocessing import preprocessing_data

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
DIR.mkdir(exist_ok=True)

X_train, X_test, y_train, y_test = preprocessing_data("podium")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def plot_decision_tree():
    for i in range(10):
        plt.figure(figsize=(12, 6))
        plot_tree(
            rf.estimators_[i],
            feature_names=X_train.columns,
            filled=True,
            max_depth=2,
            impurity=False,
            proportion=True
        )
        plt.tight_layout()
        plt.savefig(DIR / "plots" / "decision trees" / f"tree_{i}.png", dpi=200)
        # oder plt.show() im Jupyter/GUI
        plt.close()


# ─── Interaktive Prediction ──────────────────────────────────────────────────

def predict_position():
    print("\n── F1 Position Predictor ──")
    print("Features eingeben (Enter = überspringen / Standardwert):\n")

    # Gleiche Spalten wie X_train — passe die Namen an dein preprocessing an!
    inputs = {}

    feature_defaults = {
        "GridPosition": ("Startplatz (1–20)", 10),
        "Q_best_sec": ("Beste Quali-Zeit in Sekunden (z. B. 82.4)", 85.0),
        "median_laptime": ("Median Rundenzeit in Sekunden (z. B. 91.2)", 92.0),
        "box": ("Anzahl Boxenstopps (0–4)", 1),
        "rainfall": ("Regen? (0 = nein, 1 = ja)", 0),
        "year": ("Saison (z. B. 2024)", 2024),
    }

    for col, (label, default) in feature_defaults.items():
        if col not in X_train.columns:
            continue  # Feature nicht im Modell → überspringen
        raw = input(f"  {label} [{default}]: ").strip()
        inputs[col] = float(raw) if raw else default

    # Fehlende Spalten mit 0 auffüllen (für One-Hot-encoded Spalten etc.)
    input_df = pd.DataFrame([inputs])
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    prediction = rf.predict(input_df)[0]
    proba = rf.predict_proba(input_df)[0]
    confidence = proba.max() * 100

    print(f"\n  Vorhergesagter Platz : {prediction}")
    print(f"  Konfidenz            : {confidence:.1f}%")

    # Top-3 wahrscheinlichste Plätze anzeigen
    top3_idx = proba.argsort()[-3:][::-1]
    print("\n  Top-3 Wahrscheinlichkeiten:")
    for idx in top3_idx:
        print(f"    Platz {rf.classes_[idx]:>2}  →  {proba[idx] * 100:.1f}%")


predict_position()