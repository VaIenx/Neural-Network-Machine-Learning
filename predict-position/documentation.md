# FastF1Collector – Technische Dokumentation

---

## 1. Überblick

Die Klasse `FastF1Collector` automatisiert die Datenerhebung aus der FastF1-API. Sie lädt Rennergebnisse für konfigurierte Grands Prix und Jahre, reichert die Rohdaten um berechnete Kennzahlen an und stellt das Ergebnis als strukturierten Pandas DataFrame bereit.

---

## 2. Abhängigkeiten

Das Modul setzt folgende Bibliotheken voraus:

- `fastf1` – Zugriff auf die offizielle Formel-1-Telemetrie- und Ergebnis-API
- `pandas` – Datenstrukturierung und -verarbeitung als DataFrame

> **Hinweis:** Der Cache wird global vor der Klasseninstanziierung aktiviert: `fastf1.Cache.enable_cache('./cache')`

---

## 3. Klassenattribute

Die Konfiguration erfolgt direkt im Konstruktor über drei private bzw. geschützte Attribute.

| Attribut | Typ | Beschreibung |
|---|---|---|
| `__teams` | `dict` | Mapping von Teamname auf numerischen Schlüssel. Aktuell: McLaren=1, Ferrari=2, Mercedes=3, Red Bull Racing=4. |
| `_races` | `list[str]` | Liste der zu ladenden Grand-Prix-Namen (FastF1-Bezeichner, z. B. `'France'`). |
| `_years` | `list[int]` | Liste der Saisons, für die jeder Kurs geladen wird. |
| `_df` | `pd.DataFrame` | Interner Sammelbehälter. Wird leer initialisiert und durch `create_DataFrame()` befüllt. |

> **Hinweis:** Das doppelte Unterstrich-Präfix bei `__teams` erzeugt Name-Mangling in Python. Auf das Attribut kann von außen nur über `_FastF1Collector__teams` zugegriffen werden.

---

## 4. Methoden

### 4.1 `__init__(self)`

Initialisiert alle Konfigurationsattribute und ruft unmittelbar `create_DataFrame()` auf. Nach der Instanziierung ist `_df` damit bereits befüllt und auf der Konsole ausgegeben.
```python
collector = FastF1Collector()
# → Konstruktor lädt automatisch alle konfigurierten Rennen
```

---

### 4.2 `get_session_data(self, year, race) → pd.DataFrame`

Lädt eine einzelne Rennsession über die FastF1-API und gibt einen DataFrame mit folgenden Spalten zurück:

| Spalte | Quelle | Bedeutung |
|---|---|---|
| `Abbreviation` | `session.results` | Dreistelliges Fahrerkürzel (z. B. HAM, VER) |
| `TeamName` | `session.results` | Vollständiger Teamname als Zeichenkette |
| `GridPosition` | `session.results` | Startposition im Rennen |
| `Position` | `session.results` | Zielposition / Endplatzierung |
| `box` | `session.laps` | Anzahl der Boxenstopps (max. Stint − 1) |
| `year` | Parameter | Übergabewert `year` |
| `race` | Parameter | Übergabewert `race` |

> **Hinweis:** Die Berechnung der Boxenstopps über `session.laps['Stint'].max() - 1` ist eine Näherung: Sie setzt voraus, dass jeder Stint durch genau einen Stopp getrennt wird und kein Stint durch technische Ausfälle vorzeitig endet.
```python
# Beispielaufruf
df = collector.get_session_data(2021, 'France')
```

---

### 4.3 `append_data_to_DataFrame(self, data)`

Hängt einen neuen DataFrame an den internen Sammelbehälter `_df` an. Die Methode nutzt `pd.concat` mit `ignore_index=True`, um einen lückenlosen Index zu gewährleisten.
```python
self._df = pd.concat([self._df, data], ignore_index=True)
```

> **Hinweis:** `pd.concat` erzeugt stets eine neue Kopie. Bei sehr großen Datensätzen kann dies zu erhöhtem Speicherbedarf führen. In diesem Fall bietet sich eine Liste als Zwischenpuffer an, die am Ende einmalig konkateniert wird.

---

### 4.4 `create_DataFrame(self)`

Iteriert über alle Kombinationen aus `_years` und `_races`, ruft für jede Kombination `get_session_data()` auf und übergibt das Ergebnis an `append_data_to_DataFrame()`. Nach Abschluss der Schleife wird `_df` alphabetisch nach Fahrerkürzel sortiert und der Index zurückgesetzt.
```python
for year in self._years:
    for race in self._races:
        data = self.get_session_data(year, race)
        self.append_data_to_DataFrame(data)

self._df = self._df.sort_values('Abbreviation').reset_index(drop=True)
print(self._df)
```

---

## 5. Datenfluss

| Schritt | Aktion | Ergebnis |
|---|---|---|
| 1 | Instanziierung: `FastF1Collector()` | Attribute gesetzt, `create_DataFrame()` gestartet |
| 2 | Äußere Schleife über `_years` | Iteration pro Saison |
| 3 | Innere Schleife über `_races` | Iteration pro Grand Prix |
| 4 | `get_session_data(year, race)` | Einzelner DataFrame für diese Session |
| 5 | `append_data_to_DataFrame(data)` | Daten in `_df` integriert |
| 6 | Sortierung & Reset | Finaler, sortierter DataFrame |

---

## 6. Konfiguration

Alle inhaltlichen Parameter werden direkt im Konstruktor angepasst.

#### Weitere Rennen hinzufügen
```python
self._races = ['France', 'Monaco', 'Silverstone']
```

#### Weitere Saisons hinzufügen
```python
self._years = [2021, 2022, 2023]
```

#### Teams erweitern
```python
self.__teams['Alpine'] = 5
```

---

## 7. Bekannte Einschränkungen

- **Boxenstopp-Berechnung:** `session.laps['Stint'].max() - 1` liefert einen falschen Wert, wenn ein Fahrer das Rennen nicht beendet oder mehrere Stints ohne echten Stopp protokolliert werden.
- **Kein Fehlerhandling:** Schlägt `session.load()` fehl (z. B. Netzwerkfehler), bricht die gesamte Initialisierung ab.
- **`__teams` wird nicht genutzt:** Das Dictionary ist definiert, fließt aber nicht in die erzeugten Daten ein.
- **Fehlende Normalisierung:** `TeamName` bleibt als Klartext erhalten. Für ML-Anwendungen ist One-Hot-Encoding oder Label-Encoding erforderlich.

---

## 8. Erweiterungsideen

- **Fehlerbehandlung:** `try/except` um `session.load()` mit Logging statt Hard Crash.
- **`__teams` aktivieren:** `TeamName` mit dem Mapping in eine numerische Spalte umwandeln.
- **Qualifyingdaten:** Analoge Methode `get_qualifying_data()` für Rundenzeiten aus Q1–Q3.
- **Telemetrie:** Maximale und mittlere Geschwindigkeit pro Stint über `get_car_data()` ergänzen.
- **Property-Accessor:** Öffentliche read-only-Property `df`, die `_df` zurückgibt, statt direktem Attributzugriff.