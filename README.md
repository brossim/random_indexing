# Implementierung und Evaluierung des Random Indexing Wortvektormodells


## Modulprojekt Computerlinguististische Techniken WiSe 2022/2023

Das vorliegende Projekt implementiert die in Sahlgren 2005 vorgestellte Methodik zur Erstellung von randomisierten und dimensionsreduzierten Wortvektoren zur semantischen Repräsention von Wörtern. Zur Evaluierung werden verschiedene Modelle statistisch mit menschlichen Bewertungen zur Relation zwischen Wörtern mittels der Spearman Korrelation verglichen (Intrinsische Evaluierung) und als Features zur Darstellung von Texten verwendet, anhand derer ein Textklassifkation trainiert und auf die kategorisierten Texte des Brown-Korpus angewandt.  

## 1. Struktur & Bestandteile

```
clt_projekt_bross.
+---outputs
|   \---extrinsic_eval
|   \---models
|   \---Output_extrinsic_eval.pdf
+---random_indexing
|   \---__init__.py
|   \---BrownDocVec.py
|   \---CorpusPreprocessor.py
|   \---evaluate_document_vectors.py
|   \---evaluate_ratings.py
|   \---IntrinsicEvaluator.py
|   \---RandomIndexingModel.py
|   \---train_ri_model.py
|   \---train_text_classification.py
+---tests
|   \---__init__.py
|   \---test_brown_docvec.py
|   \---test_corpus_preprocessor.py
|   \---test_random_indexing_model.py
\---__init__.py
\---README.md
\---requirements.txt
```

### 1.1 Ordner 'outputs'

Enthält die von RandomIndexingModel.py generierten Modelle ('models' Unterordner), die für die intrinische und extrinsische Bewertung verwendet wurden, sowie die Training-, Validierungs- und Testsplitdaten der extrinsischen Bewertung (als .csv). Die Outputs des Skripts evaluate_document_vectors, sprich die Precision, Recall und F1 Werte der Modelle über alle Klassen (= Textkategorien aus Brown) und pro jeweiliger Klasse sind in diesem Ordner in der PDF 'Output_extrinsic_eval.pdf' gespeichert. 

### 1.2 Ordner 'random_indexing'

Enthält alle benötigten Klassen und Skripte zur Erstellung und Evaluierung von Random Indexing Modellen.


### 1.3 Ordner 'tests'

Siehe 2.4

### 1.4 requirements.txt

Siehe 2.2

## 2. Installation & Anwendung

### 2.1 Download

Das Projekt kann entweder als lokale Kopie des Repositories geclont oder als komprimierter Ordner (.zip o.ä) über den Download-Button heruntergeladen werden. 

Zum Clonen wird folgender Befehl benutzt: 

```
$ git clone https://gitup.uni-potsdam.de/bross/clt_projekt_bross
```

Es ist zu beachten, dass die ursprüngliche Ordner- und Dateistruktur beibehalten werden muss, um einen fehlerfreien Programmablauf zu gewährleisten.

### 2.2 Installation notwendiger Packages

Alle benötigten packages sind in 'requirements.txt' gelistet und können mithilfe des package installers 'pip' ausgehend vom Projektstammverzeichnis in eine Conda-Umgebung installiert werden:

```
$ pip install -r requirements.txt
```

### 2.3 Kommandozeilenschnittstellen

Das Projekt verfügt über vier Kommandozeilenschnittstellen, die nach Navigieren in den 'random_indexing' über das Terminal ausgeführt werden können:

### 2.3.1 train_ri_model.py

Random Indexing Modelle können über die Kommandozeilenschnittstelle erstellt werden: 

```
$ python train_ri_model.py --corpus "brown" oder "Pfad zu einer Textdatei" -d "Vektordimension als Integer" -n "Anzahl Nicht-Null-Dimensionen als Integer" -l "Größe des linken Fensterkontextes als Integer" -r "Größe des rechten Fensterkontextes als Integer" --name (optional) "Name des Modells"
```

### 2.3.2 evaluate_ratings.py

Mit dieser Schnittstelle können die trainierten Modelle intrinsisch bewertet werden (gegen die Datensätze in 2.6):

```
$ python evaluate_ratings.py -m "Pfad zu einem Modell als .pkl Datei" -r "Pfad zu einem Datensatz als .txt Datei"
```

### 2.3.3 train_text_classification.py

Die für die extrinsische Bewertung notwendigen .csv Daten, sprich Features und Labels des Training-, Validierungs- und Testsplits, können mit dieser Schnittstelle erstellt werden: 

```
$ python train_text_classification.py -p "Pfad zu einem Modell als .pkl Datei" -m "Modus: training/validation/testing; Features und Labels für den jeweiligen Modus"
```

### 2.3.4 evaluate_document_vectors.py

Übergibt die  4 benötigten .csv Dateien für die extrinsische Evaluierung der Modelle:

```
$ python evaluate_document_vectors.py -t "Pfad zum Trainingsset als .csv" -l "Pfad zu den Labels des Trainingssets als .csv" -d "Pfad zu den zu klassifizierenden Dokumenten als .csv" -g "Pfad zu den gold labels der zu klassifizierenden Dokumente als .csv"
```

### 2.5 Python-Version
Für das gesamte Projekt wurde **Python 3.9** verwendet. 

### 2.6 Daten
Die Datensätze zur intrinsischen Bewertung stammen von Simlex999 und WordSim353 und sollten nach dem Download in den Projektordner entpackt werden, beispielsweise durch Anlegen eines Ordners 'data'. Es ist zu beachten, dass die jeweiligen .txt Dateien ('SimLex-999.txt' 'wordsim_relatedness_goldstandard.txt' und 'wordsim_similarity_goldstandard.txt') nicht unbenannt werden dürfen, da die Klasse BrownDocVec anhand des Dateinamens automatisch erkennt, welcher Datensatz eingelesen und verarbeitet werden soll. 

Die Datensätze können unter den folgenden Links heruntergeladen werden: 

[SimLex](https://fh295.github.io/simlex.html)

[WordSim](http://alfonseca.org/eng/research/wordsim353.html)

### 2.7 Unittests 

Drei Unittestsuites befinden sich im 'tests' Ordner.
Zur Ausführung im Terminal wird in den Projektordner navigiert und der Befehl ``` pytest ``` ausgeführt (pytest muss installiert sein), um alle Tests auf einmal auszuführen oder ```pytest TestDateiName.py```, um einzelne Testsdateien auszuführen. 

Alternativ können die Tests auch in einer IDE geöffnet (bspw. PyCharm) und dort mit einer Python tests configuration ausgeführt werden. 
