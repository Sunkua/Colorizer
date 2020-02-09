# Colorizer
Bild und Video-Kolorierung mittels CNN

# Problembeschreibung
Dieses Projekt soll mittels Deeplearning-Technologien Graustufen-Videos in Farbvideos konvertieren. Hierbei soll der Fokus nicht auf einer möglichst exakten Reproduktion der realen Farben liegen sondern das Video in glaubhaften Farben darstellen.

# Related Work
Es existieren bereits mehrere Ansätze zur Kolorierung von Bildern und Videos:

## Kolorierung von Videos:

### DeepRemaster: Temporal Source-Reference Attention Networks for Comprehensive Video Enhancement (Satoshi Iizuka, Edgar Simo-Serra): http://iizuka.cs.tsukuba.ac.jp/projects/remastering/data/remastering_siggraphasia2019.pdf
Verwendet Temporal Convolutional Neural Networks zur Kolorierung von Videosequenzen statt RNNs.

## Kolorierung von Bildern:
### Colorful Image Colorization(Richard Zhang, Phillip Isola, Alexei A. Efros): https://arxiv.org/abs/1603.08511
Bild-Kolorierung mittels eines Klassifikationsansatzes



# Datensätze
Für das Training der Modelle wurden Bild- und Videodatensätze verwendet. Die Bilddatensätze wurden zum Test der generellen Architektur verwendet und das Modell anschließend auf den Videodaten trainiert.

## Youtube-Videos
Für das initiale Training wurden zufällig ca. 14.000 Youtube-Videos mit einer Auflösung von mindestens 640px in der Breite heruntergeladen. Eine Liste mit den Youtube-Video-IDs für den Download befindet sich im Dataset-Ordner. Die Videos können beispielsweise mit youtube-dl heruntergeladen werden.

### Preprocessing
Die Videos wurden mittels ffmpeg auf einheitliche 480 * 320 Pixel skaliert und gepadded. Nach ersten Tests wurde die Auflösung aufgrund des hohen Rechenleistungsbedarfs weiter auf 240 * 135 reduziert. 

Für die Verarbeitung im CNN müssen alle Videos in den LAB-Farbraum konvertiert werden. Um einen Zugriff via Index auf die Daten zu ermöglichen wurde ein Pre-Processing-Script geschrieben, dass auf meheren Knoten und mithilfe einer Redisdatenbank die Videos ins HDF5-Datenformat konvertiert. (/preprocess_lab/job.py) Die erstellten HDF5-Dateien müssen anschließend noch mit dem Script (/preprocess_job/reduce.py) in eine Datei überführt werden.

## STL10
Bei STL10 handelt es sich um einen Bild-Datensatz, der 100.000 unklassifizierte Bilder enthält. Da für diesen Datensatz bereits ein Dataset in PyTorch existiert, eignet er sich sehr gut als Baseline.
http://ai.stanford.edu/~acoates/stl10/

## ImageNet
ImageNet ist der größte Bild-Datensatz mit mehr als 1.000.000 Bildern. Für die Verwendung des Datensatz wurde eine Anfrage gestellt. Diese ist jedoch bisher unbeantwortet geblieben. Allerdings gibt es noch Quellen für ältere Versionen des Datensatzes. Daher wurde ein Subset von ca. 250.000 Bildern auch von ImageNet als weitere Baseline verwendet.
http://www.image-net.org/

## Allgemeines Bild-Preprocessing
Alle Bilder werden vom RGB in den LAB-Farbraum konvertiert. Dafür wird das OpenCV-Framework verwendet. Die Bilder werden zunächst von uInt8 nach float32 konvertiert und auf einen Bereich von 0-1 normalisiert. Anschließend werden sie in den LAB-Farbraum konvertiert. Dort liegen der L-Wert zwischen 0 und 100, a und b zwischen -127 und 127. a und b werden auf -1 bis 1 normalisiert und l auf 0 bis 1. Der L-Kanal dient dann als Eingabe für die Modelle.
Es wurden ebenfalls Experimente mit einer Sigmoid-Funktion als Aktivierung des letzten Layers der Modelle durchgeführt. Hier wurden dann die a und b-Werte zwischen 0 und 1 normalisiert.

# Architekturen

## Generelle Informationen

Bei den Architekturen wurde ein genereller Unterschied zwischen Regressions- und Klassifikationsmodellen gemacht. 
Da die ersten Ergebnisse auf Basis von Regressionsverfahren (MSE-Loss, L1-Loss und Huber-Loss zwischen realem Bild und Modell-Vorhersage) sehr schlechte (beinahe ausschließlich Graustufen) Ergebnisse lieferte, wurde im weiteren Projektverlauf noch ein Klassifikationsmodell auf Basis des Colorful Image Colorization Papers implementiert. 

### Regression
Bei der Regression wird das Modell darauf trainiert die zwei Farbkanäle a und b des LAB Farbraums auf Basis der Eingabe des L-Kanals vorherzusagen. Als Fehler wird dabei auf Basis einer Distanzmetrik (In diesem Fall L2-Loss) zwischen Eingabe und Ausgabe bestimmt. 

### Klassifikation
Kolorierung auf Basis eines Regressionsverfahrens funktioniert häufig nicht gut. Das hat sich in den durchgeführten Experimenten und im Paper Colorful Image Colorization gezeigt. Der berechnete Fehler benachteiligt weniger häufig vorkommende Farben und zieht das Bild in einen Sepia-Grauton. Dieses Problem kann umgangen werden, indem statt der zwei Farbkanäle eine Wahrscheinlichkeitsverteilung über Farb-Bins vorhergesagt wird und das Problem als Klassifikationsproblem betrachtet wird.
Statt den Fehler zwischen zwei Farbwerten zu bestimmen, wird der Fehler zwischen zwei Farb-Wahrscheinlichkeiten berechnet. 
Hinzukommt, dass der Fehler auf Basis der am seltensten vorkommenden Farben weiter gewichtet wird. Damit können unterrepräsentierte Farben verstärkt werden. Das Modell sagt damit deutlich gesättigte Bilder vorher. 


## U-Net CNN

## U-Net CNN + Feature-Vektor

## U-Net CNN + LSTM (Stateful und nicht Stateful)

## U-Net CNN + Feature-Vektor + LSTM (Stateful und nicht Stateful)

## Größeres U-Net CNN mit Batchnorm

## Trainingsparameter

### Regressionsmodelle

| Parameter    | Wert   |
|--------------|--------|
|     LR       |  1e-4  |
|  Optimizer   |  ADAM  |
| Batchsize    | 256    |
| Sequenzlänge | Batchsize  |
| Lossfunktion |   MSE  |

### Klassifikationsmodelle
| Parameter    | Wert   |
|--------------|--------|
|     LR       |  1e-4  |
|  Optimizer   |  ADAM  |
| Batchsize    |   256  |
| Sequenzlänge | Batchsize  |
| Lossfunktion |   Multinomial Cross-Entropy  |

Anmerkung zur Loss-Funktion:
Die Loss-Funktion wurde auf Basis des Papers Colorful Image Colorization implementiert. Dabei handelt es sich um eine Variante des Cross-Entropy-Loss, die im Gegensatz zur Pytorch-Implementierung auch Soft-Encodings als Target verarbeiten kann.

# Technische Herausforderungen

## Stateful LSTM
Da Videos meist zu lang sind, um sie vollständig in einem Sample zu verarbeiten, kann das Training und die Inferenz durch Zwischenspeichern der LSTM-States verkürzt werden. Anstelle des Zurücksetzens der States nach jedem Batch, wird für jedes Video der entsprechende Zustand mit in das Modell gegeben. Erst wenn ein Video beendet ist, wird der Zustand genullt.

## Dataloader für Stateful LSTMs
Pytorch unterstützt mit den Basisfunktionen nicht das inkrementelle Laden von Daten aus mehrere Streams. Daher musste für das Training ein Dataloader mit Multiprocessing implementiert werden, der parallel mehrere Videos lädt und über ein Flag das Modell informieren kann, ob ein Video beendet wurde. Diese Implementierung ist leider nicht performant genug um eine Grafikkarte vollständig auszulasten. Aus diesem Grund wurde in den Trainings immer nur ein Video zur Zeit geladen und verarbeitet.


# Ergebnisse und Evaluation
Zur Evaluation der kolorierten Bilder und Videos gibt es keine Metrik, die die Genauigkeit des Modells misst. Verwandte Arbeiten haben hierfür Befragungen durchgeführt (vgl. Zhang et al. Colorful Image Colorization) und auf Basis der Befragungsergebnisse die Güte ihres Modells bewertet. 


## Beispielbilder

# Diskussion
Die Kolorierung funktioniert nur in seltenen Fällen wirklich gut. Häufig sind sind die Farben wenig gesättigt und häufig fehlerhaft. Wenn ein Modell auf ImageNet- oder STL10-Daten trainiert wurde sehen die Ergebnisse auf dem Validierungs-Set häufig besser aus als auf ausgewählten Videos. Vermutlich liegt dies an den deutlichen Unterschieden und Ansätzen der Datensätze. Die Daten aus ImageNet und STL10 sollen immer in den Bildern die Klasse des dargestellten Objekts erkennen lassen wohingegen der Youtube-Datensatz nur aus zufälligen Videos besteht. In diesen Videos kann es vorkommen, dass bestimmte Klassen nicht vorkommen.  

Vermutlich liegt dies am Training auf entweder nur Bildern oder der geringen Varianz in den Daten aus den Youtube-Videos. Da die LSTM-Modelle sich nur auf Videos wirklich evaluieren lassen, wurden auch nur dort Videos als Trainingsmaterial verwendet. Die Videos sind häufig 











