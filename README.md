# dlo-project-09

Das Projekt sollte in einzelne Phasen unterteilt werden, um den Fortschritt entsprechend schriftlich festhalten zu können. 

Python Version: 3.8
CudaVersion >= 11.3 

# Phase 1: Daten-Aufbereitung und -Verwaltung

- Eigenes Dataset aufstellen: 200 Bilder
- Dataset Klassen angleichen
- Einlesen der Daten (Bilder auf eine einheitliche Pixelgröße einstellen)
- Preprocessing:
    - Augmentation: Spiegeln (beide Richtungen), Farben ändern (Hintergrund anpassen, Helligkeit anpassen)
    - Nur Schwarze Kanten, Farbkanäle einführen

# Phase 2: Basismodell aufbauen

- Kochbuch:
    - Literatur Rechereche: Was gibt es für bestehende Modelle?
        -> Bericht: Einleitung
    - Baseline Modell erstellen

### Optimierungsalgorithmus - Basismodell:
- SGD mit Momentum und learning rate decay/Adam
- Batch Normalization kann großen Einfluss haben → früh testen
- Regularisierung nötig? → Dropout
- ggf. transfer learning mit vortrainierten Netzen

# Phase 3: Optimierung 1

- Anwendungen in manchen Bereichen profitieren von einer
ersten Phase mit un ̈uberwachtem Lernen oder halb- ̈uberwachtem Lernen


- Rücksprache mit Herr Salmen.

# Phase 4: Optimierung 2

?Augmentation: Skalierung (Zoom, verkleinern), Rotation?



# Phase 5: Fertigstellung des Berichts