# Rotationsaugmentation in der landmark-basierten Gebärdenerkennung

Code zum Forschungsantrag (M. Klein, HAW Hamburg, M.Sc. Digital Reality):
Einfluss von Roll-Rotationsaugmentation (±5°, ±10°, ±20°) auf ein LSTM für
isolierte ASL-Gebärden. Gepaartes Design mit 45 Seeds je Bedingung
(4 × 45 = 180 Trainingsläufe).

## Daten

Verwendet wird ASL Citizen (Desai et al., 2023). Der Datensatz ist nicht
enthalten und muss über die Originalquelle bezogen werden. Nach der Lizenz
dürfen weder Videos noch daraus extrahierte Landmarks weitergegeben werden.

Vor dem Start die drei Split-Dateien aus dem Archiv
(`ASL_Citizen/splits/{train,val,test}.csv`) nach `data/aslc_splits/`
kopieren. Die Auswahl der 100 Klassen steht in
`data/aslc_class_selection.json` (`all`). Die Schlüssel `existing` und `new`
halten nur fest, in welchem von zwei Extraktionsläufen mit identischer
Konfiguration die Klassen verarbeitet wurden. Die Auswahl wurde ohne Blick
auf den Testsplit oder auf Modellergebnisse getroffen.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

MediaPipe ist auf `0.10.9` festgelegt, weil `mp.solutions.holistic` in
späteren Versionen nicht mehr enthalten ist.

## Ablauf

Alle Befehle werden im Wurzelverzeichnis dieses Pakets ausgeführt.
Parameter, die nicht als Argument übergeben werden, stehen in
`configs/default.yaml`.

**1. Videos der ausgewählten Klassen entpacken**

```bash
python -m src.data.prepare_asl_citizen --zip <pfad>/ASL_Citizen.zip
```

**2. Landmark-Extraktion mit MediaPipe Holistic**

```bash
python -m src.data.extract_landmarks --videos-dir data/videos_aslc --out-dir data/landmarks_aslc
```

Einstellungen: `model_complexity=1`, kürzere Bildseite auf 384 px,
jeder Frame, höchstens 300 Frames. Randframes ohne erkannte Hand werden
abgeschnitten, Sequenzen mit weniger als 8 Frames verworfen, fehlende Werte
innerhalb der Sequenz linear interpoliert. Ergebnis: `(T, 75, 3)` je Video
(33 Pose- und 2 × 21 Hand-Landmarks) sowie `extraction_stats.json`.

**3. Ausschlussregeln und Manifest**

```bash
python -m src.data.filter_asl_citizen --config configs/default.yaml
```

Ausgeschlossen werden Sequenzen mit mehr als 30 % Frames ohne Hand
(Anteil vor der Interpolation) und Klassen mit weniger als 10
Trainingssequenzen. Das Manifest (`data/landmarks_aslc/manifest.json`)
protokolliert die Zahl ausgeschlossener Sequenzen, die Sequenzen, in denen
die linke bzw. rechte Hand in keinem Frame erkannt wurde, sowie Klassen,
Sequenzen und Personen je Teilmenge und deren Überschneidung.

**4. Konfigurationswahl (AP 1, nur Train/Val)**

```bash
python -m src.ap1_search --seeds 3
python -m src.ap1_search --seeds 8 --only B_klein_1layer
python -m src.ap1_search --seeds 8 --only C_klein_subset
python -m src.ap1_search --seeds 8 --only E_klein_bs32
```

Ergebnisse in `results_aslc/ap1_search.jsonl`. Die Zusammenfassung
vergleicht zusätzlich die beiden besten Kandidaten mit einem gepaarten
t-Test über die gemeinsamen Seeds. Wenn alle Läufe im jsonl bereits
vorliegen, wird nur die Zusammenfassung ausgegeben. Die Werte des gewählten
Kandidaten (`C_klein_subset`) sind in `configs/default.yaml` eingetragen und
gelten für alle vier Bedingungen.

**5. Training (180 Läufe)**

```bash
python -m src.run_experiments
```

Einzellauf:

```bash
python -m src.train --condition rot10 --seed 7
```

**6. Statistische Auswertung und Abbildungen**

```bash
python -m analysis.run_analysis --results results_aslc --figures figures_aslc
```

Die Auswertung benötigt nur `results_aslc/` und lässt sich ohne Datensatz
und ohne erneutes Training ausführen.

Abbildung zur Rotation eines Einzelframes (benötigt die Videos):

```bash
python -m analysis.figure_rotation_demo --out figures_aslc/fig_rotation_demo.png
```

## Methodische Festlegungen

**Vorverarbeitung.** Ankerpunkt ist die Schultermitte (Pose 11/12) je Frame,
skaliert wird mit dem Median der Schulterbreite. Danach wird jede Sequenz
linear auf 64 Frames umgerechnet und auf 56 Landmarks reduziert (Schultern
bis Hüfte und beide Hände).

**Rotation.** Pro Trainingssequenz und Epoche wird ein Winkel
θ ~ U(−A, +A) gezogen und auf alle Frames angewendet (x und y um den
Ankerpunkt, z unverändert). Validierungs- und Testdaten werden nicht rotiert.

**Bewegungsmerkmale.** Nach der Rotation werden die Differenzen zum
Vorframe angehängt: `[x, y, z, dx, dy, dz]` je Landmark, also
56 × 6 = 336 Eingabewerte pro Frame.

**Seeds.** 45 Seeds (`src/seeds.py`) mit drei getrennten Zufallsströmen:
Initialisierung und Dropout `seed`, Batch-Reihenfolge `seed + 20000`,
Rotationswinkel `seed + 10000`. Baseline und Rotationsbedingungen
unterscheiden sich bei gleichem Seed nur in der Rotation. cuDNN läuft
deterministisch.

**Evaluationszeitpunkt.** Evaluiert wird nach der letzten Epoche (120).
Der Validierungssplit dient nur AP 1 und wird im Hauptexperiment nicht zur
Modellauswahl verwendet.

**Statistik.** Für jeden Seed wird die Differenz zwischen zwei Bedingungen
gebildet. Einseitige gepaarte t-Tests für H1a–H1c (Holm-korrigiert) und H2
(separat), Cohen's dz, 95 %-Bootstrap-KI (B = 10 000) und die Zahl der Paare
mit positiver Differenz, jeweils für Top-1-Accuracy und Macro-F1. Für die
Top-1-Accuracy wird die mittlere Differenz zusätzlich in korrekt
klassifizierte Testsequenzen umgerechnet. Robustheitsprüfung mit gepaartem
Permutationstest (für H1a–H1c ebenfalls Holm-korrigiert) und
Wilcoxon-Test. TOST (Grenze dz = ±0,50) für beide Zielgrößen bei den
Vergleichen, die in der Top-1-Accuracy nicht signifikant sind.

## Enthaltene Ergebnisse

`results_aslc/` enthält die Ergebnisse der 180 Trainingsläufe und der
Konfigurationswahl. Die Statistikdateien sind mit
`analysis.run_analysis` aus `all_results.csv` erzeugt.

| Datei | Inhalt |
|---|---|
| `results_aslc/<bedingung>/seed_XXX.json` | Test-Accuracy, Macro-F1, Val-Accuracy und Val-Macro-F1 je Epoche, Konfiguration, Umgebung |
| `results_aslc/all_results.csv` | ein Eintrag je Lauf, Grundlage der Auswertung |
| `results_aslc/ap1_search.jsonl` | Ergebnisse der Konfigurationswahl |
| `results_aslc/statistics.md`, `statistics.csv`, `statistics_f1.csv` | Testergebnisse für Top-1-Accuracy und Macro-F1 |

Die Abbildungen werden nach `figures_aslc/` geschrieben:
`qq_differences.png` (Q-Q-Plots der gepaarten Differenzen),
`accuracy_by_condition.png` (Top-1-Accuracy je Bedingung mit
Seed-Paarlinien) und `forest_effects.png/.pdf` (mittlere gepaarte
Differenzen mit 95 %-KI für Top-1-Accuracy und Macro-F1).

## Struktur

```
configs/default.yaml              Daten-, Modell- und Trainingsparameter
data/aslc_class_selection.json    Auswahl der 100 Klassen
src/data/prepare_asl_citizen.py   Videos aus dem Archiv nach Klassen ablegen
src/data/extract_landmarks.py     Landmark-Extraktion mit MediaPipe
src/data/filter_asl_citizen.py    Ausschlussregeln, Manifest
src/data/preprocessing.py         Normalisierung, Resampling, Geschwindigkeiten
src/data/dataset.py               PyTorch-Dataset mit Rotation im Training
src/augmentation.py               Rotation und Winkelziehung
src/seeds.py                      Seeds und Offsets der Zufallsströme
src/model.py                      LSTM
src/ap1_search.py                 Konfigurationswahl auf Train/Val
src/train.py                      ein Trainingslauf mit Evaluation
src/run_experiments.py            alle 180 Läufe
analysis/statistics.py            gepaarte Tests, Effektgrößen, KIs
analysis/plots.py                 Abbildungen
analysis/run_analysis.py          Auswertung starten
analysis/figure_rotation_demo.py  Abbildung zur Rotation eines Frames
results_aslc/                     Ergebnisse der Trainingsläufe und Statistik
```
