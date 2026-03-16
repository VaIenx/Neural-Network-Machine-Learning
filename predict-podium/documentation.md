# 🏎️ F1 Podium Predictor — Neuronales Netz zur Podiumsvorhersage

> Schulprojekt im Fach Informationstechnik  
> Thema: Maschinelles Lernen / Neuronale Netze

---

## Inhaltsverzeichnis

1. [Projektziel](#1-projektziel)
2. [Daten & Feature-Engineering](#2-daten--feature-engineering)
3. [Modellauswahl](#3-modellauswahl)
4. [Risiken & Limitierungen](#4-risiken--limitierungen)

---

## 1. Projektziel

Ziel dieses Projekts ist die Entwicklung eines binären Klassifikationsmodells, das auf Basis historischer Formel-1-Daten vorhersagt, ob ein Fahrer in einem Rennen eine **Podiumsplatzierung (P1–P3)** erreicht.

Das Modell soll lernen, aus einer Kombination aus Qualifyingdaten, Rennstatistiken und Wetterbedingungen ein verlässliches Signal für die Zielvariable `podium ∈ {0, 1}` zu extrahieren.

### Zielvariable

```
podium = 1  →  Fahrer belegt Platz 1, 2 oder 3
podium = 0  →  Fahrer belegt Platz 4 oder schlechter / DNF
```

---

## 2. Daten & Feature-Engineering

### Datenquelle

Die Rohdaten werden über die Python-Bibliothek **[FastF1](https://theoehrly.github.io/Fast-F1/)** bezogen, welche offizielle Telemetrie-, Ergebnis- und Wetterdaten der Formel-1-Weltmeisterschaft bereitstellt. Erfasst wurden die Saisons **2022, 2023 und 2024**.

### Begründung der Saisonauswahl

Die Saison 2022 markiert eine der größten Regelreformen in der Geschichte der Formel 1: Mit der Einführung der **Ground-Effect-Aerodynamik** wurden die technischen Grundlagen der Fahrzeuge fundamental neu definiert. Dadurch verschoben sich die Kräfteverhältnisse zwischen den Teams erheblich — Fahrer und Konstrukteure, die unter dem alten Reglement dominant waren, fanden sich teils im Mittelfeld wieder.

Würde man Saisons **vor und nach 2022** mischen (z. B. 2021–2023), entstünden widersprüchliche Muster im Datensatz: Dieselben Features (`TeamName`, `median_laptime`, `GridPosition`) hätten je nach Ära eine grundlegend andere Aussagekraft. Ein Modell, das darauf trainiert wird, lernt möglicherweise **kein allgemeingültiges Muster**, sondern einen Mittelwert zweier inkompatibler Ären.

Die bewusste Beschränkung auf **2022–2024** stellt sicher, dass alle Trainingsdaten unter einheitlichen sportlichen und technischen Rahmenbedingungen entstanden sind. Dies verbessert die **interne Konsistenz** des Datensatzes und erhöht die Wahrscheinlichkeit, dass gelernte Muster auf neue Rennen der gleichen Ära übertragbar sind.

### Erhobene Features

| Feature | Beschreibung | Quelle |
|---|---|---|
| `Abbreviation` | Fahrerkürzel (z. B. `VER`, `HAM`) | `session.results` |
| `TeamName` | Konstrukteur / Team | `session.results` |
| `GridPosition` | Startplatz im Rennen | `session.results` |
| `Position` | Finale Rennposition | `session.results` |
| `Status` | `Finished` oder `DNF` (klassifiziert nach Ausfallursache) | `session.results` |
| `box` | Anzahl der Boxenstopps (Stints − 1) | `session.laps` |
| `median_laptime` | Median der bereinigten Rundenzeiten in Sekunden (ohne In-/Outlap) | `session.laps` |
| `Q_best_sec` | Beste Qualifyingzeit in Sekunden (Q3 → Q2 → Q1, fallback) | Qualifying-Session |
| `year` | Saison | abgeleitet |
| `race` | Rennen / Grand Prix | abgeleitet |
| `rainfall` | Ob es während des Rennens geregnet hat (`True`/`False`) | `session.weather_data` |

### Datenaufbereitung

- **DNF-Klassifikation:** Der `Status`-String wird regelbasiert in `Finished` oder `DNF` überführt. Schlüsselwörter wie `Crash`, `Engine`, `Gearbox` etc. lösen die DNF-Kennzeichnung aus.
- **Fehlende Werte:** Zeilen mit `NaN`-Werten werden vor dem Speichern entfernt (`.dropna()`).
- **Zielvariable:** Wird im Preprocessing aus `Position ∈ {1, 2, 3}` abgeleitet und ist **nicht** Teil der Rohdaten.

### Datenpipeline

```
FastF1 API
    │
    ├── Race Session     →  Ergebnisse, Stint-Daten, Rundenzeiten, Wetter
    └── Qualifying Session →  Q1/Q2/Q3-Zeiten
          │
          ▼
    FastF1Collector.create_DataFrame()
          │
          ▼
    DATA.csv  (eine Zeile = ein Fahrer pro Rennen)
```

---

## 3. Modellauswahl

### Problemtyp

Binäre Klassifikation auf tabellarischen, strukturierten Daten.

### Kandidatenmodelle

| Modell | Begründung |
|---|---|
| **Feedforward Neural Network (MLP)** | Primäres Modell des Projekts; geeignet für nicht-lineare Zusammenhänge zwischen numerischen und kategorischen Features |
| **Random Forest** | Starke Baseline für tabellarische Daten; robust gegenüber Ausreißern und irrelevanten Features |
| **Gradient Boosting (XGBoost / LightGBM)** | State-of-the-Art für strukturierte Daten; gut interpretierbar via Feature Importance |
| **Logistische Regression** | Einfache, interpretierbare Baseline; liefert kalibrierte Wahrscheinlichkeiten |

### Empfohlenes Modell

Für den Einstieg eignet sich ein **MLP mit 2–3 Hidden Layers** (z. B. `[64, 32, 16]` Neuronen, ReLU-Aktivierung, Dropout zur Regularisierung, Sigmoid-Ausgabe). Als Verlustfunktion wird **Binary Cross-Entropy** verwendet.

Da pro Rennen nur 3 von 20 Fahrern aufs Podium fahren, ist das Dataset **unbalanciert** (ca. 15 % positive Klasse). Gegenmaßnahmen:

- Class Weights (`class_weight='balanced'`)
- Oversampling (SMOTE) oder Undersampling
- Evaluation über **F1-Score, Precision, Recall** statt Accuracy

---

## 4. Risiken & Limitierungen

### 4.1 Datenbias

| Risiko | Beschreibung |
|---|---|
| **Konstrukteursbias** | Red Bull dominierte 2022–2024 die Podiumswertung extrem (v. a. 2023 mit 21 von 22 Siegen). Das Modell könnte lernen, `TeamName = Red Bull Racing` als nahezu hinreichende Bedingung für ein Podium zu werten, ohne strukturelle Muster zu erfassen. |
| **Gridpositions-Bias** | Startplatz 1–3 korreliert stark mit Podiumsergebnissen. Das Modell könnte den Qualifying-Ausgang nahezu deterministisch lernen und echte Renndynamik ignorieren. |
| **Strecken-Bias** | Bestimmte Kurse (z. B. Monaco) begünstigen strukturell Überholmanöver, andere nicht. Dieser Kontext ist in den Features nicht explizit kodiert. |

### 4.2 Overfitting

- Der Datensatz umfasst ca. 3 Saisons × ~23 Rennen × 20 Fahrer ≈ **~1.380 Datenpunkte** — ein relativ kleiner Datensatz für neuronale Netze.
- Ein MLP kann bei dieser Datenmenge schnell overfitten. Gegenmaßnahmen: **Dropout**, **Early Stopping**, **Kreuzvalidierung** (z. B. Leave-One-Season-Out).
- Besonders kritisch: Wenn Fahrerkürzel (`Abbreviation`) oder Teamnamen als kategorische Features direkt kodiert werden, lernt das Modell möglicherweise Identitäten statt Muster.

### 4.3 Datenqualität & -vollständigkeit

| Problem | Ursache |
|---|---|
| Fehlende Qualifyingzeiten | Fahrer, die Q1 nicht abschließen, haben kein vollständiges `Q_best_sec`. Wird durch `.dropna()` entfernt. |
| DNF-Klassifikation unvollständig | Die regelbasierte Keyword-Liste deckt nicht alle Ausfallgründe ab. Neue oder unbekannte Statustexte werden fälschlich als `Finished` klassifiziert. |
| Wetterfeature grob | `rainfall` ist ein binäres Flag über die gesamte Session — Intensität, Zeitpunkt und Streckenbereiche sind nicht erfasst. |
| API-Verfügbarkeit | FastF1 ist von den offiziellen F1-Servern abhängig. Fehlende Caches oder API-Änderungen können die Reproduzierbarkeit einschränken. |

### 4.4 Generalisierbarkeit

Das Modell wird auf Daten von 2022–2024 trainiert. Alle drei Saisons entstammen bewusst derselben Reglementära (Ground-Effect-Fahrzeuge ab 2022), um widersprüchliche Muster durch Regelbrüche zu vermeiden. Dennoch können Teamwechsel, Fahrerentwicklungen oder kleinere Regelanpassungen innerhalb der Ära dazu führen, dass gelernte Muster auf zukünftige Saisons nur eingeschränkt übertragbar sind. Eine **temporale Validierung** (Training auf 2022/23, Test auf 2024) ist daher realistischer als ein zufälliger Train-Test-Split und spiegelt den tatsächlichen Anwendungsfall — Vorhersage zukünftiger Rennen — besser wider.

---

## Projektstruktur

```
f1-podium-predictor/
├── cache/                  # FastF1 Session Cache
├── DATA.csv                # Aufbereiteter Rohdatensatz
├── F1Collector.py          # FastF1Collector Klasse
├── model.py                # Modellarchitektur & Training
├── visualizer.py           # Visualisierungen
├── plots/                  # Plots des Visualizers
└── documentation.md        # Diese Dokumentation
```

---

## Abhängigkeiten

```bash
pip install fastf1 pandas tqdm scikit-learn torch
```

---

*Erstellt im Rahmen des Informationstechnik-Unterrichts — Thema: Neuronale Netze & maschinelles Lernen*