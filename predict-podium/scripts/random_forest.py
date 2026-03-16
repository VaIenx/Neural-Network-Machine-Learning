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
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphvi



DIR = Path(__file__).resolve().parents[1]

print(DIR)
df = pd.read_csv(str(DIR / "DATA.csv"))

