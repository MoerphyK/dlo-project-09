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
    - Nur Schwarze Kanten, 
	(- Farbkanäle einführen)
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
Salmen:
``` 
Besonders wichtig ist mir ein systematisches, wissenschaftliches Vorgehen, das in Ihrem
Abschlussbericht dementsprechend dokumentiert wird. Stellen Sie sich z.B. vor, dass Sie Ihre
Erfahrungen/Erkenntnisse während der Entwicklung für Kollegen oder Auftraggeber festhalten. Alle
relevanten Entscheidungen (etwa für oder gegen bestimmte Module) sollten genauso gut begründet
werden wie die konkrete Wahl von Parametern. Typischerweise wird dafür ein iterativer Prozess nötig
sein mit jeweils zielgerichteten Experimenten.
```
## Bericht Aufbau
```
Note: Zu jedem Schritt, welcher die Leistung verändern kann sollte es eine kleine vorher nachher Anekdote geben.
Mit kleineren Codesnippets und Diagramme.
```
TODO:: Erstellung von Diagrammen zur beseren Veranschaulichung.

Aufbau
1. Abstract
3. Einleitung
- Aufgabenstellung / Ziel
- State of the Art
- Erwartungshaltung / Hypothese (Kaggle Repos)
- Baseline Modell
4. Datenaufbereitung / Dateneinlesung
- Aufbau des Datensets
	- Abweichung vom Kaggle Datensatz
	- Größe
	- Klassenkriterien beschreiben
- Entscheidung Klassenanzahl (nur Optionen beschreiben mit Gründen)
- Aufbereitung
	- Größenverhältnis
	- Erzeugen neuer Bilder
		- GaussianBlur
		- RandomRotation
		- Grayscale
		- RandomVerticalFlip
		- RandomHorizontalFlip
		- ColorJitter
		- Normalize
6. Netzaufbau
- Layeraufbau an AlexNet 32x32 angelehnt
- Dropout
- Early Stopping
	- Speichern des bisher besten Netzes
7. Netztraining
- Layeranzahl anpassen
- Neuronenanzahl anpassen
- Dropout Rate anpassen
- Early Stopping anpassen
8. Auswertungen / Vergleiche
- Testdatensatz Nutzung
- Vergleichstabelle
	- Initiales Netz ohne Augmentation, Dropout oder Early Stopping
	- Nur Augmentation
	- Dropout hinzufügen
	- Early Stopping hinzufügen
	- Hyperparameter Tuning
		- Learnrate
		- Neuronen Anzahl
		- Layer Anzahl
- Vergleich mit der initial aufgestellten Hypothese / Kaggle Repos
9. Fazit
- Auf Auswertung eingehen (Ergebnis des Testdatensatzes)
- Vergleichen wie sich unser großer Datensatz im Vergleich zu den kleinen Kaggle Datensatz / Repos andere verhält.
- Anmerkungen was sich als effektivstes Änderung herausgestellt hat
- ggf. Ausblick