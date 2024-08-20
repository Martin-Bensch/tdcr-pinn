# Project name
#### Personal information

#### Project description


# Sphinx
- index.rst definiert die generelle Struktur.
- Um automatisch aus den \*.py Dateien die Dokumentation zu erstellen wird für das jeweilige Modul bspw. module.py eine \*.rst Datei "Module" im Root Ordner angelegt mit Inhalt:  
- Nun ```$ sphinx-build -b html ./doc ./doc/build``` ausführen, wobei sourcedir der Root Ordner der Dokumentation i ist und builddir der Buildordner. 
- Im  ```./doc/build``` liegen nun die erstellen html Dateien. Öffne bspw. ```index.html``` in Deinem bevorzugten Browser.  

# Projektablage imes-material
WIKI: https://wiki.projekt.uni-hannover.de/imes-material/start

GIT-Wiki: https://wiki.projekt.uni-hannover.de/imes-material/tutorials/git

2
