# Dokumentation: Vorhersage der Reifenmischung (S/M/H) mittels Neuronaler Netze

---

## 1. Datenakquise und Initialisierung

In diesem Schritt wird die Verbindung zur FastF1 API hergestellt. Um die Ladezeiten bei wiederholten Zugriffen zu minimieren, wird ein lokaler Cache-Ordner verwendet.

```python
import fastf1, os
import pandas as pd

# Sicherstellen, dass das Cache-Verzeichnis existiert
if not os.path.exists('cache'):
    os.makedirs('cache')

# Cache aktivieren
fastf1.Cache.enable_cache('cache')

# Laden der Session (Beispiel: GP Frankreich 2021, Rennen)
session = fastf1.get_session(2021, 'France', 'R')
session.load()
```

---

## 2. Konvertierung in Pandas DataFrame

Die Rohdaten der Runden (`laps`) werden in ein Pandas DataFrame umgewandelt. Dies ermöglicht eine effiziente Tabellenansicht und Filterung der Daten für das spätere Machine Learning Modell.

```python
# Extraktion der Runden-Daten
laps = session.laps

# Umwandlung in ein strukturiertes DataFrame
f1_data = pd.DataFrame(laps)

# Erste Überprüfung der Datenstruktur
print(f1_data.head())
print(f"\nDatensatz-Größe: {f1_data.shape}")
```

---

## ⚠️ Änderung des Projektziels

> **Hinweis zur Transparenz:** Die ursprüngliche Zielsetzung dieser Dokumentation war die **Klassifikation des Reifentyps** (Soft / Medium / Hard). Im Verlauf der Analyse hat sich gezeigt, dass dieses Ziel für ein erstes Neuronales Netz problematisch ist:
>
> - Medium- und Hard-Reifen unterscheiden sich im Fahrverhalten kaum messbar
> - Das Modell würde nicht an der Aufgabe scheitern, sondern daran dass **kein ausreichendes Signal in den Daten vorhanden ist**
> - Zusätzlich besteht ein **Datenleck-Risiko**: `TyreLife` ist eine direkte Ableitung des Reifentyps und würde die Klassifikation trivial machen
>
> **Das neue Ziel ist Regression:** Das Modell sagt die **Rundenzeit in Sekunden** vorher, basierend auf Reifenzustand, Reifentyp und Rennsituation. Dies ist strategisch relevanter, technisch sauberer und als Einstieg in Neuronale Netze besser geeignet.

---

## 3. Datenaggregation über mehrere Rennen

Ein einzelnes Rennen liefert zu wenig Daten (~50 Runden pro Fahrer). Deshalb werden **5 Grands Prix der Saison 2021** geladen und zusammengeführt. Die Auswahl deckt verschiedene Streckentypen ab, damit das Modell nicht nur auf eine Streckencharakteristik optimiert wird.

```python
import fastf1, os
import pandas as pd
from tqdm import tqdm

if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

races_to_load = [
    (2021, 'Belgium'),
    (2021, 'France'),
    (2021, 'Austria'),
    (2021, 'Silverstone'),
    (2021, 'Abu Dhabi')
]

teams = ['Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren']
all_laps_data = []

for year, gp in races_to_load:
    print(f"\n--- Lade {gp} {year} ---")
    try:
        session = fastf1.get_session(year, gp, 'R')
        session.load()

        # Filtern: Nur relevante Teams, keine Boxenrunden, nur Slick-Reifen
        laps = session.laps.pick_teams(teams).pick_wo_box().pick_compounds(['SOFT', 'MEDIUM', 'HARD'])

        if len(laps) == 0:
            print(f"Keine Daten für {gp}. Überspringe...")
            continue

        all_laps_data.append(laps)

    except Exception as e:
        print(f"Fehler beim Laden von {gp}: {e}")
```

**Warum diese Filter?**
- `pick_teams()` – Nur die 4 konkurrenzfähigen Top-Teams, um konsistente Datenqualität zu gewährleisten
- `pick_wo_box()` – Boxenrunden haben künstlich lange Rundenzeiten und würden das Modell verfälschen
- `pick_compounds(['SOFT', 'MEDIUM', 'HARD'])` – Regenreifen folgen einer völlig anderen Physik und gehören nicht in dieses Modell

---

## 4. Telemetrie-Extraktion und TrackStatus-Filter

Pro Runde werden zusätzliche Telemetriedaten geladen: maximale und minimale Geschwindigkeit. Außerdem wird geprüft ob die Runde unter normalen Rennbedingungen gefahren wurde (`TrackStatus == '1'`). Runden unter Safety Car oder Virtual Safety Car sind künstlich langsam und würden das Modell verzerren.

```python
for year, gp in races_to_load:
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    laps = session.laps.pick_teams(teams).pick_wo_box().pick_compounds(['SOFT', 'MEDIUM', 'HARD'])

    max_speeds, min_speeds, valid_indices = [], [], []

    for index, lap in tqdm(laps.iterrows(), total=len(laps), desc=f"Telemetrie {gp}"):
        # Nur Runden unter grüner Flagge (kein Safety Car, kein VSC, kein Unfall)
        if str(lap['TrackStatus']) == '1':
            try:
                car_data = lap.get_car_data()
                if len(car_data) > 0:
                    max_speeds.append(car_data['Speed'].max())
                    min_speeds.append(car_data['Speed'].min())
                    valid_indices.append(index)
            except Exception as e:
                # Runde überspringen, aber Grund festhalten
                print(f"Runde {index} übersprungen: {e}")
                continue
```

**TrackStatus Übersicht:**

| Status | Bedeutung |
|---|---|
| `1` | Normales Rennen (grüne Flagge) ✅ |
| `2` | Gelbe Flagge (Gefahr auf der Strecke) ❌ |
| `4` | Safety Car ❌ |
| `5` | Rote Flagge (Rennen unterbrochen) ❌ |
| `6` | Virtual Safety Car (VSC) ❌ |

---

## 5. Datenbereinigung und Export

Die gefilterten Runden werden zusammengeführt, auf die relevanten Spalten reduziert und als CSV gespeichert. Die Rundenzeit wird von einem Zeitdelta-Format in Sekunden (float) umgewandelt, da Neuronale Netze ausschließlich mit Zahlen arbeiten.

```python
if all_laps_data:
    final_df = pd.concat(all_laps_data, ignore_index=True)

    # Nur relevante Spalten behalten
    keep_cols = ['LapTime', 'LapNumber', 'TyreLife', 'Compound', 'Team', 'GP']
    final_df = final_df[keep_cols]

    # LapTime von timedelta in Sekunden umwandeln
    # Beispiel: 0:01:31.405 → 91.405
    final_df['LapTime'] = pd.to_timedelta(final_df['LapTime']).dt.total_seconds()

    # Zeilen mit fehlenden Werten entfernen
    final_df = final_df.dropna()

    final_df.to_csv('DATA.csv', index=False)
    print(f"ERFOLG: {len(final_df)} Runden gespeichert.")
else:
    print("Keine Daten extrahiert.")
```

Nach diesem Schritt sieht `DATA.csv` beispielhaft so aus:

| LapTime | LapNumber | TyreLife | Compound | Team | GP |
|---|---|---|---|---|---|
| 91.405 | 5 | 4 | SOFT | Mercedes | France |
| 92.117 | 6 | 5 | SOFT | Mercedes | France |
| 93.840 | 7 | 6 | SOFT | Mercedes | France |

---

## 6. Preprocessing für das Neuronale Netz

Rohdaten können nicht direkt in ein Neuronales Netz gegeben werden. Zwei Schritte sind zwingend notwendig:

### 6.1 One-Hot-Encoding

Kategorische Spalten (`Compound`, `Team`) müssen in Zahlen umgewandelt werden. Das geschieht über One-Hot-Encoding: Jede Kategorie bekommt eine eigene Spalte mit 0 oder 1.

```
SOFT   → [1, 0, 0]
MEDIUM → [0, 1, 0]
HARD   → [0, 0, 1]
```

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('DATA.csv')

# One-Hot-Encoding für kategorische Spalten
df = pd.get_dummies(df, columns=['Compound', 'Team'], drop_first=False)

# GP-Spalte entfernen (nur zur Übersicht gedacht, kein sinnvolles Feature)
df = df.drop(columns=['GP'])
```

### 6.2 Normalisierung

Zahlenwerte wie `LapTime` (~90s) und `TyreLife` (~1–40) haben sehr unterschiedliche Größenordnungen. Neuronale Netze lernen deutlich besser wenn alle Inputs auf einem ähnlichen Wertebereich liegen (typischerweise 0–1 oder -1–1).

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Features (X) und Ziel (y) trennen
X = df.drop(columns=['LapTime'])
y = df['LapTime']

# Daten aufteilen: 80% Training, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisierung: Mittelwert 0, Standardabweichung 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit_transform nur auf Trainingsdaten!
X_test = scaler.transform(X_test)        # transform (nicht fit) auf Testdaten
```

> **Wichtig:** `fit_transform` darf nur auf den **Trainingsdaten** aufgerufen werden. Würde man den Scaler auf allen Daten fitten, würde Information aus den Testdaten ins Training einfließen – das verfälscht die Bewertung des Modells.

---

## 7. Neuronales Netz mit Keras

Jetzt wird das eigentliche Modell gebaut. Die Architektur ist bewusst einfach gehalten: zwei versteckte Schichten mit ReLU-Aktivierung, ein einzelner Output-Neuron für die vorhergesagte Rundenzeit.

```python
import tensorflow as tf
from tensorflow import keras

# Anzahl der Input-Features (nach One-Hot-Encoding)
input_dim = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(64, activation='relu'),   # Versteckte Schicht 1
    keras.layers.Dense(32, activation='relu'),   # Versteckte Schicht 2
    keras.layers.Dense(1)                        # Output: eine Zahl (LapTime)
])

# Modell konfigurieren
model.compile(
    optimizer='adam',          # Lernalgorithmus
    loss='mse',                # Mean Squared Error – Standardverlust für Regression
    metrics=['mae']            # Mean Absolute Error – intuitiver zu interpretieren
)

model.summary()
```

**Warum diese Architektur?**
- `Dense(64)` und `Dense(32)` – einfach genug um Overfitting zu vermeiden, komplex genug um Muster zu lernen
- `relu` – Standardaktivierung für versteckte Schichten, funktioniert zuverlässig
- Kein `activation` im Output-Layer – bei Regression soll der Wert unbegrenzt sein

---

## 8. Training

```python
history = model.fit(
    X_train, y_train,
    epochs=100,           # 100 Durchläufe durch die Trainingsdaten
    batch_size=32,        # 32 Datenpunkte pro Gewichtsanpassung
    validation_split=0.1, # 10% der Trainingsdaten zur Validierung während des Trainings
    verbose=1
)
```

**Was passiert hier?**
- Das Netz sieht in jeder **Epoch** alle Trainingsdaten einmal
- Nach jeweils 32 Datenpunkten (**Batch**) werden die Gewichte angepasst
- Der **Validation Split** zeigt ob das Modell generalisiert oder nur auswendig lernt

---

## 9. Evaluation

```python
from sklearn.metrics import mean_absolute_error

# Vorhersagen auf den Testdaten
y_pred = model.predict(X_test).flatten()

# MAE berechnen
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.3f} Sekunden")
```

**Wie gut ist das Modell?**

| MAE | Bewertung |
|---|---|
| < 1.0s | Sehr gut |
| 1.0 – 2.0s | Akzeptabel für Stufe 1 |
| > 2.0s | Modell lernt kaum etwas |

---

## 10. Ausblick: Nächste Schritte

Das hier beschriebene Modell ist **Stufe 1** eines dreistufigen Plans:

| Stufe | Features | Ziel |
|---|---|---|
| **Stufe 1** ← Aktuell | TyreLife, Compound, LapNumber | Erstes funktionierendes NN |
| **Stufe 2** | + Team, MeanSpeed | Genauigkeit verbessern |
| **Stufe 3** | + Wetterdaten, Throttle, Brake | Feintuning |

Sobald Stufe 1 einen MAE unter 2.0s erreicht, macht es Sinn weitere Features ergänzen und zu messen ob sie wirklich helfen.