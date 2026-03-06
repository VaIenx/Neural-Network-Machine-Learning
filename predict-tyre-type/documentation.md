## Dokumentation: Vorhersage der Reifenmischung (S/M/H) mittels Neuronaler Netze

### 1. Datenakquise und Initialisierung

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

### 2. Konvertierung in Pandas DataFrame

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
