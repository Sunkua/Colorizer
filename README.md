# Colorizer
Bild und Video-Kolorierung mittels CNN

# Problembeschreibung



# Related Work


# Datensätze
Für das Training der Modelle wurden Bild- und Videodatensätze verwendet. Die Bilddatensätze wurden zum Test der generellen Architektur verwendet und das Modell anschließend auf den Videodaten trainiert.

## Youtube-Videos
Für das initiale Training wurden zufällig ca. 14.000 Youtube-Videos mit einer Auflösung von mindestens 640px in der Breite heruntergeladen. 

### Preprocessing
Die Videos wurden mittels ffmpeg auf einheitliche 480 * 320 Pixel skaliert und gepadded. Nach ersten Tests wurde die Auflösung aufgrund des hohen Rechenleistungsbedarfs weiter auf 240 * 135 reduziert. 

Für die Verarbeitung im CNN müssen alle Videos in den LAB-Farbraum konvertiert werden. Um einen Zugriff via Index auf die Daten zu ermöglichen wurde ein Pre-Processing-Script geschrieben, dass auf meheren Knoten und mithilfe einer Redisdatenbank die Videos ins HDF5-Datenformat konvertiert. (/preprocess_lab/job.py) Die erstellten HDF5-Dateien müssen anschließend noch mit dem Script (/preprocess_job/reduce.py) in eine Datei überführt werden.

## STL10
Bei STL10 handelt es sich um einen Bild-Datensatz, der 100000 unklassifizierte Bilder enthält. Da für diesen Datensatz bereits ein Dataset in PyTorch existiert, eignet er sich sehr gut als Baseline.
http://ai.stanford.edu/~acoates/stl10/

## ImageNet
ImageNet ist der größte Bild-Datensatz mit mehr als 1.000.000 Bildern. Für die Verwendung des Datensatz wurde eine Anfrage gestellt. Diese ist jedoch bisher unbeantwortet geblieben. Allerdings gibt es noch Quellen für ältere Versionen des Datensatzes. Daher wurde ein Subset von ca. 250.000 Bildern auch von ImageNet als weitere Baseline verwendet.
http://www.image-net.org/


# Architekturen

## U-Net CNN



## U-Net CNN + Feature-Vektor

## U-Net CNN + LSTM

## U-Net CNN + Feature-Vektor + LSTM

## Größeres U-Net CNN mit Batchnorm













