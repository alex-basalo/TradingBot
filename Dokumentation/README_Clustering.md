# Trading Bot - Asset Clustering

Dieses Modul ist Teil eines algorithmischen Trading-Bots und gruppiert ein Universum von 100 Finanzwerten (Forex, Krypto, Aktien, Indizes, Rohstoffe) in verschiedene Cluster. 

## Funktionsweise des Codes (`clustering.py`)
Der Code implementiert einen hybriden Clustering-Ansatz für Zeitreihen:
1. **Datenbeschaffung & Feature-Extraktion:** Historische Schlusskurse der letzten 3 Jahre werden über Yahoo Finance bezogen. Daraus werden finanzielle Kennzahlen (Features) wie tägliche Rendite, rollierende Volatilität und Momentum berechnet und normalisiert.
2. **Dynamic Time Warping (DTW):** Um zeitlich verschobene Muster zu erkennen, wird DTW als Distanzmaßstab verwendet.
3. **K-Means Clustering:** Der K-Means-Algorithmus gruppiert die Assets anhand ihrer Features. Die optimale Anzahl der Cluster wird automatisch über den besten *Silhouette Score* ermittelt.
4. **Profil-Erstellung:** Am Ende gibt das Skript die echten Durchschnittswerte (Volatilität, Momentum) für jedes Cluster im Terminal aus, um die mathematischen Entscheidungen des Algorithmus transparent und interpretierbar zu machen.

## Installation und Ausführung

**1. Abhängigkeiten installieren:**
Installiere die benötigten Python-Bibliotheken über die bereitgestellte Requirements-Datei:
```bash
pip install -r clustering_requirements.txt
```

**2. Skript ausführen:**
Starte den Clustering-Prozess:
```Bash
python3 clustering.py
```
**3. Ergebnisse:**
Nach erfolgreichem Durchlauf erstellt das Skript automatisch zwei Dateien im selben Verzeichnis:
1. `bot_cluster_config.csv` - Für die tabellarische Auswertung und Analyse.
2. `bot_cluster_config.json` - Eine maschinenlesbare Dictionary-Datei, die vom Trading-Bot eingelesen wird, um die clusterspezifischen Handelsstrategien (z.B. Support/Resistance) anzuwenden.
