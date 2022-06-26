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
- Ergebnis Phase 1: 
	augmented_pictures
	\-scissors
	 -stone
	 -paper
	 -misc
	 
	-> Wird geladen von Dataloader
	Augmentation speichert die Bilder in 4 seperaten Ordnern je nach Label
	Dataloader greift auf diese Verzeichnisse zu


# Phase 2: Basismodell aufbauen

- Hyperparameter
	Test- und Trainsplit							-> 80% zu 20% (9x200 + 2200 von Kaggle (ohne misc)) -> skewed data set	
		+ Cross Validation	(Split und Validation sollten zwischen Basismodell und optimierten Modellen gleich bleiben um Vergleichbarkeit zu erzeugen)
	Anzahl Schichten und Neuronen pro Schichten
	Aktivierungsfunktionen
	Learning Rate (Optimierungsalgorithmus) 		-> SGD
	
	
300x200 Pixel sind sehr große Bilder -> runter skalieren (30x20?) (falls zu klein 60:40, falls zu groß (15:10 (angelehnt an LeNet)))
	
- Netz Architektur:
	Conv:
	Convolution Layer + (Max)Pooling Layer 		} mehrmals (Größenordnung 1-2) 
	Sigmoid
	Flatten (?)
	Fully Connected Layer:
	3 Stück
	Als Aktivierungsfunktionen Sigmoid oder ReLU

- Kochbuch:
    - Literatur Rechereche: Was gibt es für bestehende Modelle? (Siehe Kaggle Dataset: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
        -> Bericht: Einleitung
    - Baseline Modell erstellen
	
Potentiell zu Phase 3 verschieben
- Evaluation von Phase 2 (Bilder für Bericht machen!):
	Recall, Accuracy, Precision und F1 Score (Ergeben Matrix)
	

### Optimierungsalgorithmus - Basismodell:

Aus Phase 2:
	Dropout (Leicht zu implementieren, erhöht performance)
	Learning Rate (Optimierungsalgorithmus)			-> Adam


- SGD mit Momentum und learning rate decay/Adam
- Batch Normalization kann großen Einfluss haben → früh testen
- Batch Size (Hyperparameter)
- Regularisierung nötig? → Dropout
- ggf. transfer learning mit vortrainierten Netzen
- Verhältnis zwischen Recall und Precision als Hyperparameter einstellen
- Overfitting analysieren (Early Stopping) (Durch Dropout sollte overfitting vermieden werden)

# Phase 3: Optimierung 1

- Anwendungen in manchen Bereichen profitieren von einer
ersten Phase mit un ̈uberwachtem Lernen oder halb- ̈uberwachtem Lernen


- Rücksprache mit Herr Salmen.

# Phase 4: Optimierung 2

?Augmentation: Skalierung (Zoom, verkleinern), Rotation?



# Phase 5: Fertigstellung des Berichts