from pathlib import Path
import warnings
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
DIR = Path(__file__).resolve().parents[1]


def preprocessing_data(target="podium", verbose=1): # ← "podium" oder "position" | verbose: 0=nichts, 1=progressbar, 2=progressbar+tabellen

    steps = 7
    bar = tqdm(total=steps, desc="Preprocessing", unit="step", disable=verbose == 0)

    df = pd.read_csv(f'{DIR}/DATA.csv')
    bar.set_postfix_str("CSV geladen"); bar.update(1)

    # Zielvariable je nach Modus setzen
    if target == "podium":
        df['podium'] = (df['Position'] <= 3).astype(int)  # Wenn podium dann 1
        df = df.drop(columns=['Abbreviation', 'Position', 'year', 'race'])  # spalten rauswerfen damit das NN nicht cheated
    elif target == "position":
        df = df[df['Position'] >= 1]  # fehlerhafte 0-Werte entfernen
        df = df.drop(columns=['Abbreviation', 'year', 'race'])  # spalten rauswerfen damit das NN nicht cheated
    bar.set_postfix_str(f"Target '{target}' gesetzt"); bar.update(1)

    df['Status'] = (df['Status'] == 'Finished').astype(int)  # Wenn finished dann 1
    df['rainfall'] = df['rainfall'].astype(int)  # regen statt Bool als 0/1
    bar.set_postfix_str("Bool-Spalten kodiert"); bar.update(1)

    le = LabelEncoder()  # codiert die Namen in zahlen
    df['TeamName'] = le.fit_transform(df['TeamName'])
    bar.set_postfix_str("TeamName Label-encoded"); bar.update(1)

    scaler = StandardScaler()  # skaliert die Zahlen, damit alle gleich viel wert sind
    df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']] = scaler.fit_transform(
        df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']]
    )
    bar.set_postfix_str("Features skaliert"); bar.update(1)

    # SPLITTING
    if target == "podium":
        X = df.drop(columns=['podium'])
        y = df['podium']
    elif target == "position":
        X = df.drop(columns=['Position'])
        y = df['Position'].astype(int)
    bar.set_postfix_str("X/y gesplittet"); bar.update(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # splittet zu 80% training und 20% test mit Seed 42
    bar.set_postfix_str("Train/Test-Split fertig"); bar.update(1)

    bar.close()

    if verbose >= 2:
        print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
        print(df.head())
        print(df.dtypes)

    return X_train, X_test, y_train, y_test, scaler, le  # ← scaler + le mitgeben für User-Input


def preprocessing_data_for_rf(target="podium", verbose=1):

    steps = 6  # ← eine weniger, Status-Schritt fällt weg
    bar = tqdm(total=steps, desc="Preprocessing", unit="step", disable=verbose == 0)

    df = pd.read_csv(f'{DIR}/DATA.csv')
    bar.set_postfix_str("CSV geladen"); bar.update(1)

    if target == "podium":
        df['podium'] = (df['Position'] <= 3).astype(int)
        df = df.drop(columns=['Abbreviation', 'Position', 'Status', 'year', 'race'])  # ← Status raus
    elif target == "position":
        df = df[df['Position'] >= 1]
        df = df.drop(columns=['Abbreviation', 'Status', 'year', 'race'])  # ← Status raus
        df['Position'] = pd.cut(                                           # ← Bins statt exakte Plätze
            df['Position'],
            bins=[0, 3, 6, 10, 20],
            labels=["Podium", "Top6", "Top10", "Hinten"]
        )
    bar.set_postfix_str(f"Target '{target}' gesetzt"); bar.update(1)

    df['rainfall'] = df['rainfall'].astype(int)
    bar.set_postfix_str("Bool-Spalten kodiert"); bar.update(1)

    le = LabelEncoder()
    df['TeamName'] = le.fit_transform(df['TeamName'])
    bar.set_postfix_str("TeamName Label-encoded"); bar.update(1)

    scaler = StandardScaler()
    df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']] = scaler.fit_transform(
        df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']]
    )
    bar.set_postfix_str("Features skaliert"); bar.update(1)

    if target == "podium":
        X = df.drop(columns=['podium'])
        y = df['podium']
    elif target == "position":
        X = df.drop(columns=['Position'])
        y = df['Position']  # ← kein .astype(int) mehr, sind jetzt Strings
    bar.set_postfix_str("Train/Test-Split fertig"); bar.update(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bar.close()

    if verbose >= 2:
        print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
        print(df.head())
        print(df.dtypes)

    return X_train, X_test, y_train, y_test, scaler, le