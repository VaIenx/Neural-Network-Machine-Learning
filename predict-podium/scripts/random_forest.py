
from pathlib import Path

# Data Processing
import pandas as pd
import numpy as np
from contourpy.util import data

from preprocessing import preprocessing_data

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
import warnings
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
DIR = Path(__file__).resolve().parents[1]
DIR.mkdir(exist_ok=True)

class random_forest:
    def __init__(self, *data, n_tree=100, verbose=1):
        super().__init__()
        self.X_train = data[0]
        self.X_test = data[1]
        self.y_train = data[2]
        self.y_test = data[3]
        self.n_tree = n_tree
        self.verbose = verbose

        self.y_pred = None
        self.accuracy = None
        self.random_forest = RandomForestClassifier(
            n_estimators=1,
            warm_start=True,  # ← key: baut auf vorherigen Bäumen auf
            verbose=0,
            n_jobs=-1
        )

        self.loss_list = []
        self.val_loss_list = []

    def run(self):
        for i in tqdm(range(1, self.n_tree + 1), desc="Training Forest", unit="tree", dynamic_ncols=False):
            self.random_forest.n_estimators = i
            self.random_forest.fit(self.X_train, self.y_train)

        self.y_pred = self.random_forest.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Accuracy:", self.accuracy)

    def plot_decision_tree(self):
        for i in range(10):
            plt.figure(figsize=(12, 6))
            plot_tree(
                self.random_forest.estimators_[i],
                feature_names=self.X_train.columns,
                filled=True,
                max_depth=2,
                impurity=False,
                proportion=True
            )
            plt.tight_layout()
            plt.savefig(DIR / "plots" / "decision trees" / f"tree_{i}.png", dpi=200)
            plt.close()


    # ─── Interaktive Prediction ──────────────────────────────────────────────────

    def predict_position(self):
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
            if col not in self.X_train.columns:
                continue  # Feature nicht im Modell → überspringen
            raw = input(f"  {label} [{default}]: ").strip()
            inputs[col] = float(raw) if raw else default

        # Fehlende Spalten mit 0 auffüllen (für One-Hot-encoded Spalten etc.)
        input_df = pd.DataFrame([inputs])
        input_df = input_df.reindex(columns=self.X_train.columns, fill_value=0)

        prediction = self.random_forest.predict(input_df)[0]
        proba = self.random_forest.predict_proba(input_df)[0]
        confidence = proba.max() * 100
        print("")
        print(f"####################################################")
        print("")
        print(f"\n  Vorhergesagter Platz : {prediction}")
        print(f"  Konfidenz            : {confidence:.1f}%")

        # Top-3 wahrscheinlichste Plätze anzeigen
        top3_idx = proba.argsort()[-3:][::-1]
        print("\n  Top-3 Wahrscheinlichkeiten:")
        for idx in top3_idx:
            print(f"    Platz {self.random_forest.classes_[idx]:>2}  →  {proba[idx] * 100:.1f}%")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, scaler = preprocessing_data(verbose=1)
    random_forest = random_forest( X_train, X_test, y_train, y_test, n_tree=100, verbose=2)
    random_forest.run()

