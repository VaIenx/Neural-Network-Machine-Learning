# F1 Compound Predictor: Neural Network Classification

**Klassifizierung von Reifenmischungen basierend auf Telemetrie-Daten der FIA Formel 1**

---

## 1. Projektziel (Objective)

Das Ziel dieses Projekts ist die Entwicklung eines neuronalen Netzwerks (Multi-Class Classifier), das in der Lage ist, die verwendete Reifenmischung (**Soft, Medium oder Hard**) eines Fahrzeugs allein anhand der fahrphysikalischen Daten einer abgeschlossenen Runde vorherzusagen.

Das Modell lernt die subtilen Unterschiede in den Grip-Niveaus, die sich in höheren Kurvengeschwindigkeiten, kürzeren Bremswegen und der allgemeinen Rundenzeit widerspiegeln.

## 2. Datengrundlage (Data Engineering)

Die Daten werden über die **FastF1 API** bezogen, welche offizielle Timing- und Telemetriedaten der Formel 1 bereitstellt.

### Genutzte Features (Eingabevariablen):

* **LapTime:** Die absolute Dauer der Runde (normalisiert).
* **MaxSpeed:** Die erreichte Höchstgeschwindigkeit (Indikator für Luftwiderstand/Motorleistung).
* **MinSpeed:** Die geringste Geschwindigkeit in Kurven (Indikator für mechanischen Grip).
* **TyreLife:** Das Alter des Reifens in Runden (berücksichtigt den Performance-Abfall).
* **Stint:** Die Nummer des aktuellen Stints im Rennen.

### zu eliminierende Runden:
* Runden mit SafetyCar
* Runden mit Regen
* Runden mit gelber Flagge
* nicht abgeschlossene Rennen


### Zielvariable (Labels):

* `Compound`: Kategoriale Variable [0: Soft, 1: Medium, 2: Hard].

---

## 3. Methodik & Modellarchitektur

Das Projekt evaluiert verschiedene Modellansätze, um die höchste Präzision zu erreichen:

1. **Baseline Modell:** Ein einfacher Random Forest Classifier zur Bestimmung der Feature Importance.
2. **Neuronales Netzwerk (Deep Learning):**
* **Architektur:** Fully Connected Feedforward Neural Network (Multilayer Perceptron).
* **Input Layer:** 5–7 Neuronen (je nach Feature Selection).
* **Hidden Layers:** 2 Schichten mit ReLU-Aktivierungsfunktion und Dropout-Layern zur Regularisierung.
* **Output Layer:** 3 Neuronen mit **Softmax-Aktivierung** zur Ausgabe der Wahrscheinlichkeiten für jede Reifenklasse.
* **Optimizer:** Adam mit einer adaptiven Lernrate.



---

## 4. Risiken & Herausforderungen (Risk Assessment)

### Bias (Verzerrung)

* **Top-Team Bias:** Top-Teams (z. B. Red Bull) fahren auf einem Hard-Reifen eventuell schnellere Kurvengeschwindigkeiten als Hinterbänkler auf Soft-Reifen. Das Modell könnte fälschlicherweise "Schnelligkeit" mit "Soft-Reifen" gleichsetzen.
* **Lösung:** Normalisierung der Daten pro Fahrer oder Team.

### Overfitting

* Wenn das Modell nur auf Daten einer einzigen Rennstrecke (z. B. Monza) trainiert wird, lernt es die Streckencharakteristik statt der Reifeneigenschaften.
* **Lösung:** Nutzung von Daten aus verschiedenen Saisons und unterschiedlichen Streckentypen (High-Downforce vs. Low-Downforce).

### Datenqualität (Noise)

* Runden unter Gelben Flaggen, Safety-Car-Phasen oder Boxeneinfahrten verfälschen die Telemetrie massiv.
* **Lösung:** Aggressives Data Cleaning (Entfernung aller Runden, die mehr als 7% außerhalb der Median-Rundenzeit liegen).

---

## 5. Ergebnisse & Dokumentation

Die Ergebnisse werden mittels einer **Confusion Matrix** visualisiert, um zu analysieren, bei welchen Reifenmischungen das Modell die größten Schwierigkeiten hat (z. B. Verwechslung von Medium und Hard).

* **Metriken:** Accuracy, F1-Score und Categorical Crossentropy Loss.
* **Visualisierung:** Plot der Trainings- und Validierungskurven zur Überprüfung der Generalisierung.

---

## 6. Installation & Nutzung

```bash
# Benötigte Libraries
pip install fastf1 pandas tensorflow scikit-learn matplotlib

```
