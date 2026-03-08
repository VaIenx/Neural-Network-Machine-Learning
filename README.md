# 🏎 F1 LapTime Prediction

> Reifendegradation in der Formel 1 mit einem Neuronalen Netz vorhersagen

---

## Inhaltsverzeichnis

1. [Was ist das Ziel?](#1-was-ist-das-ziel)
2. [Warum Regression und nicht Klassifikation?](#2-warum-regression-und-nicht-klassifikation)
3. [Datenbasis](#3-datenbasis)
4. [Features](#4-features)
5. [Projektstruktur & Ablauf](#5-projektstruktur--ablauf)
6. [Stufenplan](#6-stufenplan)
7. [Bekannte Bugs & Schwachstellen](#7-bekannte-bugs--schwachstellen)
8. [F1 Begriffe erklärt](#8-f1-begriffe-erklärt)
9. [ML Begriffe erklärt](#9-ml-begriffe-erklärt)

---

## 1. Was ist das Ziel?

Das Modell soll lernen, wie lange eine Formel-1-Runde dauert – abhängig vom **Reifenzustand**, dem **Reifentyp** und der **Rennsituation**.

```
Input:  TyreLife + Compound + LapNumber + Team
           ↓
        [Neuronales Netz]
           ↓
Output: LapTime in Sekunden  (z.B. 91.4s)
```

Das klingt simpel, ist aber strategisch hochrelevant: F1-Teams nutzen genau solche Modelle intern, um zu entscheiden **wann ein Fahrer an die Box muss**, bevor der Reifen zu stark abfällt.

---

## 2. Warum Regression und nicht Klassifikation?

Es gibt zwei grundlegende Arten von Vorhersageproblemen:

| Typ | Antwort ist... | Beispiel |
|---|---|---|
| **Klassifikation** | eine Kategorie | Welcher Reifen? → Soft / Medium / Hard |
| **Regression** | eine Zahl | Wie lange dauert die Runde? → 91.4s |

**Dieses Projekt nutzt Regression**, weil:

- Die Rundenzeit ein klares, messbares Signal ist
- Muster wie *"Reifen wird älter → Runde wird langsamer"* direkt abbildbar sind
- Kein Datenleck-Risiko (siehe Abschnitt 7)
- Regression ist für Einsteiger in NNs einfacher zu evaluieren

### Warum Reifentyp-Klassifikation schwieriger wäre

Das Problem: Medium und Hard unterscheiden sich im Fahrverhalten kaum messbar. Das Netz würde nicht scheitern weil es schlecht ist, sondern weil **der Unterschied in den Daten einfach nicht groß genug ist**.

```
Runde 8, Soft,   LapTime 89.2s
Runde 8, Medium, LapTime 89.4s  ← Fast identisch
```

---

## 3. Datenbasis

Daten werden über die **FastF1 Python Library** geladen – eine Open-Source-Bibliothek die offizielle F1-Telemetriedaten bereitstellt.

### Rennen (Saison 2021)

| Grand Prix | Streckentyp | Besonderheit |
|---|---|---|
| Belgien (Spa) | Hochgeschwindigkeit | Lange Geraden, wenig Reifenverschleiß |
| Frankreich (Paul Ricard) | Mittel | Hoher Reifenverschleiß |
| Österreich (Red Bull Ring) | Kurz & schnell | Viele Kurven |
| Silverstone | Highspeed-Kurven | Hohe Lateralkräfte |
| Abu Dhabi (Yas Marina) | Nacht, glatt | Kaum Degradation |

Die Auswahl verschiedener Streckentypen ist bewusst – das Modell soll lernen, **nicht nur auf einer Streckenart** zu funktionieren.

### Teams

- Red Bull Racing
- Mercedes
- Ferrari
- McLaren

Nur diese 4 Teams werden geladen, da sie die meisten Rennen unter sich ausmachen und repräsentative Daten liefern.

### Ungefähre Datenmenge

```
5 Rennen × ~50 Runden × ~8 Fahrer = ~2.000 Datenpunkte
```

Das ist ein realistischer Startpunkt für ein erstes Neuronales Netz.

### Filterkriterien

Der Dataloader filtert aktiv unbrauchbare Daten heraus:

- ✅ Nur Slick-Reifen: Soft, Medium, Hard (keine Regenreifen)
- ✅ Keine Boxenrunden (`pick_wo_box`)
- ✅ Nur Runden ohne Rennunterbrechungen (`TrackStatus == '1'`)
- ✅ Nur Runden mit vorhandenen Telemetriedaten

> **Warum TrackStatus == '1'?**
> Runden unter Safety Car oder Virtual Safety Car sind künstlich langsam und haben nichts mit dem Reifenzustand zu tun. Solche Ausreißer würden das Modell verfälschen.

---

## 4. Features

### Aktuelle Features (Stufe 1)

| Feature | Typ | Relevanz | Beschreibung |
|---|---|---|---|
| `TyreLife` | Zahl | ★★★★★ | Wie viele Runden ist der Reifen schon gefahren |
| `Compound` | Kategorie* | ★★★★★ | Reifentyp: Soft / Medium / Hard |
| `LapNumber` | Zahl | ★★★★ | Aktuelle Runde im Rennen |
| `Team` | Kategorie* | ★★★ | Unterschiedliche Autocharakteristiken |

*Kategorische Features müssen vor dem Training in Zahlen umgewandelt werden → **One-Hot-Encoding**

### Warum LapNumber?

Je früher im Rennen, desto schwerer ist das Auto (voller Tank = ca. 110kg Benzin). Das verlangsamt die Rundenzeit um ca. 0.03–0.05s pro Kilogramm. LapNumber ist ein indirekter Proxy für das Tankgewicht.

### Warum kein einzelner Fahrer?

Fahrerunterschiede (~0.3–0.5s) sind klein im Vergleich zum Reifeneffekt (~2–3s über einen Stint). Der Unterschied zwischen Fahrern ist **akzeptables Rauschen** für ein erstes Modell.

### Spätere Features (Stufe 2 & 3)

| Feature | Quelle | Warum interessant |
|---|---|---|
| `MeanSpeed` | Telemetrie | Besseres Signal als nur MaxSpeed |
| `MeanThrottle` | Telemetrie | Aggressiver Fahrstil → mehr Degradation |
| `TrackTemp` | Wetterdaten | Heißer Track → schnellere Degradation |
| `Rainfall` | Wetterdaten | Regen verändert alles |

---

## 5. Projektstruktur & Ablauf

```
f1-prediction/
│
├── FastF1.py          # Daten laden & als CSV exportieren  ✅ vorhanden
├── DATA.csv           # Exportierte Rohdaten               ✅ wird generiert
├── cache/             # FastF1 Cache (automatisch erstellt) ✅ wird generiert
│
├── train.py           # Neuronales Netz trainieren          🔜 TODO
├── predict.py         # Vorhersagen für neue Eingaben       🔜 TODO
└── model.keras        # Gespeichertes Modell nach Training  🔜 wird generiert
```

### Ablauf

```
1. python FastF1.py
      → Lädt Telemetrie aus FastF1 API
      → Filtert & bereinigt die Daten
      → Exportiert DATA.csv

2. python train.py         (TODO)
      → Liest DATA.csv
      → One-Hot-Encoding für Compound & Team
      → Normalisierung der Zahlenwerte
      → Trainiert das Neuronale Netz
      → Speichert model.keras

3. python predict.py       (TODO)
      → Lädt model.keras
      → Nimmt neue Eingaben entgegen
      → Gibt vorhergesagte LapTime aus
```

### Geplante Modellarchitektur

```
Input Layer    →  ~10 Neuronen (nach One-Hot-Encoding)
Hidden Layer 1 →  64 Neuronen, ReLU Aktivierung
Hidden Layer 2 →  32 Neuronen, ReLU Aktivierung
Output Layer   →   1 Neuron   (LapTime in Sekunden)
```

> **Empfohlenes Framework:** Keras (TensorFlow) – einsteigerfreundlicher als PyTorch, weniger Boilerplate nötig.

---

## 6. Stufenplan

Das Projekt ist bewusst in Stufen aufgeteilt. So sieht man bei jedem Schritt **was ein neues Feature wirklich bringt**.

| Stufe | Features | Ziel |
|---|---|---|
| **Stufe 1** ← Jetzt | TyreLife, Compound, LapNumber | Erstes NN zum Laufen bringen |
| **Stufe 2** | + Team, MeanSpeed | Genauigkeit verbessern |
| **Stufe 3** | + Wetter, Throttle, Brake | Feintuning & Experimente |

### Wie bewertet man ob das Modell gut ist?

Der wichtigste Metrik für Regression ist der **MAE (Mean Absolute Error)**: der durchschnittliche Fehler in Sekunden.

- MAE < 1.0s → sehr gut
- MAE 1.0–2.0s → akzeptabel für Stufe 1
- MAE > 2.0s → Modell lernt kaum etwas

---

## 7. Bekannte Bugs & Schwachstellen

### Bugs

**`UnboundLocalError` bei leeren Daten**
`final_df` wird nur definiert wenn Daten vorhanden sind, aber am Ende immer zurückgegeben.
```python
# Fix: Am Anfang der Methode initialisieren
final_df = pd.DataFrame()
```

**Stiller `except`-Block**
`except: continue` schluckt jeden Fehler lautlos. Man weiß nie warum Runden übersprungen werden.
```python
# Fix:
except Exception as e:
    print(f"Übersprungen: {e}")
    continue
```

**`print(car_data)` in der inneren Schleife**
Gibt bei tausenden Runden tausende Zeilen aus. Vor dem echten Training entfernen.

**CSV-Einlesen fehlt**
Der `else`-Zweig liest die bestehende `DATA.csv` nicht ein – nur ein Platzhalter-Print.

### Konzeptionelle Schwachstellen

**Kein Feature-Scaling**
LapTime (~90s) und TyreLife (~1–40) haben unterschiedliche Größenordnungen. Neuronale Netze brauchen normalisierte Inputs.
```python
# Vor dem Training: StandardScaler oder MinMaxScaler aus sklearn verwenden
```

**Datenleck-Risiko (falls TyreLife als Ziel genutzt wird)**
TyreLife ist eine direkte Ableitung des Reifentyps – das würde Klassifikation trivial machen. Bei Regression kein Problem, da TyreLife hier ein Input-Feature ist.

**Nur 5 Rennen aus einer Saison**
Das Modell kennt nur 2021. Ob es auf 2022 oder andere Reglements generalisiert, ist unbekannt.

---

## 8. F1 Begriffe erklärt

**Compound / Reifentyp**
Pirelli stellt drei Trockenreifen-Varianten bereit: Soft (rot, weich, schnell aber kurzlebig), Medium (gelb, Kompromiss) und Hard (weiß, hart, langsam aber langlebig). Welche Mischungen pro Rennen verfügbar sind, entscheidet Pirelli vorab.

**TyreLife**
Wie viele Runden ein Reifen bereits gefahren wurde seit dem letzten Boxenstopp. Je höher der Wert, desto mehr hat der Reifen degradiert.

**Degradation**
Der Leistungsverlust eines Reifens über Zeit. Ein frischer Soft-Reifen ist sehr schnell, aber nach 15–20 Runden deutlich langsamer als ein frischer Medium. Das Verstehen von Degradation ist der Kern jeder F1-Strategie.

**Stint**
Eine ununterbrochene Fahrtphase zwischen zwei Boxenstopps (oder Start und erstem Stopp). Ein Rennen besteht typischerweise aus 2–3 Stints.

**Boxenstopp / Pit Stop**
Das Wechseln der Reifen in der Boxengasse. Dauert in der Formel 1 ca. 2–3 Sekunden. Die Entscheidung wann gestoppt wird ist strategisch entscheidend.

**Undercut**
Eine Strategie: Fahrer A stoppt eine Runde früher als Fahrer B, bekommt frische Reifen, fährt schnellere Runden und ist nach Fahrer Bs Stopp vorne. Funktioniert nur wenn die frischen Reifen genug Zeit einsparen.

**TrackStatus**
Ein Code der den aktuellen Rennzustand beschreibt:
- `1` = normales Rennen (grüne Flagge)
- `2` = Yellow Flag (Gefahr auf der Strecke)
- `4` = Safety Car
- `6` = Virtual Safety Car (VSC)
- `5` = Red Flag (Rennen unterbrochen)

**LapNumber**
Die aktuelle Rundennummer im Rennen. Wichtig weil das Auto am Anfang durch das Benzingewicht (~110 kg) schwerer und damit langsamer ist.

**Telemetrie**
Echtzeit-Fahrzeugdaten die während der Fahrt übertragen werden: Geschwindigkeit, Drehzahl, Bremsdruck, Gaspedalstellung, Gang usw. FastF1 stellt diese Daten im Nachhinein bereit.

**DRS (Drag Reduction System)**
Ein einfahrbarer Heckflügel der den Luftwiderstand reduziert und auf Geraden ca. 10–15 km/h Mehrgeschwindigkeit bringt. Darf nur in bestimmten Zonen genutzt werden, wenn der Abstand zum Vordermann weniger als 1 Sekunde beträgt.

---

## 9. ML Begriffe erklärt

**Neuronales Netz (NN)**
Ein Modell das grob dem menschlichen Gehirn nachempfunden ist. Es besteht aus Schichten von "Neuronen" die miteinander verbunden sind. Das Netz lernt durch viele Beispiele, welche Verbindungen wichtig sind.

**Regression**
Das Vorhersagen eines kontinuierlichen Zahlenwerts (z.B. LapTime). Gegenteil: Klassifikation.

**Klassifikation**
Das Zuordnen zu einer Kategorie (z.B. Soft / Medium / Hard).

**Feature**
Eine einzelne Eingabevariable für das Modell. In diesem Projekt: TyreLife, Compound, LapNumber, Team.

**One-Hot-Encoding**
Umwandlung von Kategorien in Zahlen. "Soft / Medium / Hard" wird zu drei Spalten:
```
Soft   → [1, 0, 0]
Medium → [0, 1, 0]
Hard   → [0, 0, 1]
```

**Normalisierung / Scaling**
Alle Zahlenwerte auf einen ähnlichen Wertebereich bringen (z.B. 0–1). Neuronale Netze lernen deutlich besser wenn Inputs ähnlich groß sind.

**Overfitting**
Das Modell lernt die Trainingsdaten auswendig statt die eigentlichen Muster. Erkenntlich daran, dass es auf Trainingsdaten sehr gut funktioniert, auf neuen Daten aber schlecht.

**Training / Validation Split**
Die Daten werden aufgeteilt: ~80% zum Trainieren, ~20% zum Testen. So sieht man ob das Modell auch auf Daten funktioniert die es nie gesehen hat.

**Loss Function**
Eine Formel die misst wie falsch das Modell gerade liegt. Das Netz versucht diese Zahl zu minimieren. Bei Regression typischerweise MSE (Mean Squared Error).

**MAE (Mean Absolute Error)**
Durchschnittlicher absoluter Fehler zwischen Vorhersage und echtem Wert. Bei Laptime-Vorhersage: "Im Schnitt liegt das Modell X Sekunden daneben."

**Epoch**
Ein kompletter Durchlauf durch alle Trainingsdaten. Typischerweise trainiert man 50–200 Epochen.

**ReLU (Rectified Linear Unit)**
Eine Aktivierungsfunktion: `f(x) = max(0, x)`. Gibt negative Werte als 0 aus, positive unverändert. Standard für versteckte Schichten in NNs.

**Batch Size**
Wie viele Datenpunkte das Netz gleichzeitig verarbeitet bevor es die Gewichte anpasst. Typisch: 32 oder 64.

---

## Abhängigkeiten

```bash
pip install fastf1 pandas tqdm numpy
pip install tensorflow  # für Keras (Stufe 1+)
pip install scikit-learn  # für Preprocessing (Stufe 1+)
```

## Schnellstart

```bash
# 1. Daten laden
python FastF1.py
# → Fragt ob DATA.csv überschrieben werden soll
# → Lädt Telemetrie (~5-15 Minuten je nach Internet)
# → Erstellt DATA.csv (DEBUG)

# 2. Training (TODO)
python train.py
```

---

*Datenbasis: FastF1 Python Library | Saison 2021 | Teams: Red Bull · Mercedes · Ferrari · McLaren*