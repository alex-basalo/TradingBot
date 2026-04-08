# Trading Bot - HMM Regime Detection

Dieses Modul ist Teil eines algorithmischen Trading-Bots und unterteilt historische Finanzmarktdaten datengetrieben in verborgene Marktphasen (sogenannte *Regime* wie z.B. Trendphasen, Seitwärtsmärkte oder Phasen hoher/niedriger Volatilität). 

Anstatt Marktphasen durch starre, reaktive Indikatoren zu definieren, nutzt dieser Algorithmus probabilistisches Machine Learning, um die aktuelle Makro-Umgebung des Marktes zu identifizieren.

## Funktionsweise des Codes (`regime_detection.py`)

Der Code implementiert einen unüberwachten Lernansatz (Unsupervised Learning) zur Erkennung von Marktphasen:

1. **Feature-Extraktion & Vermeidung von Multikollinearität:** Aus den rohen H1-Kerzendaten (OHLCV) werden spezifisch zwei finanzielle Dimensionen berechnet: die logarithmischen Renditen (Log-Returns für Richtung/Geschwindigkeit) und die Average True Range (ATR für Unsicherheit/Risiko). *Hinweis: Auf hochkorrelierte Features wie Momentum wurde hier bewusst verzichtet, um mathematische Multikollinearität im probabilistischen Modell zu vermeiden.*
2. **Hidden Markov Models (HMM):** Der Algorithmus lernt die verborgenen Zustände des Marktes ausschließlich anhand der Verteilung der Log-Returns und der Volatilität. Er "klebt" als blinder Filter lediglich mathematische Etiketten (Zustand 0, 1, 2 oder 3) auf die Zeitreihen.
3. **Modell-Optimierung via BIC (Bayesian Information Criterion):** Finanzmärkte sind stark verrauscht. Um Overfitting (das Auswendiglernen von Rauschen) zu verhindern, wird der Suchraum auf 3 bis 4 Regime begrenzt. Das BIC fungiert als mathematischer Schiedsrichter: Es bestraft unnötige Komplexität und wählt vollautomatisch aus, ob ein Asset besser durch 3 oder 4 Regime beschrieben wird.
4. **Zustands-Zuweisung:** Das finale, optimale Modell weist jeder historischen Kerze einen spezifischen HMM-Zustand zu. Das nachfolgende Optimierungs-Modul (Optimizer) kann so empirisch testen, welches dieser Regime empirisch die besten Bedingungen für die Mean-Reversion-Strategie bietet.

## Installation und Ausführung

**1. Abhängigkeiten installieren:**
Installiere die benötigten Python-Bibliotheken über die bereitgestellte Requirements-Datei:
```bash
pip install -r regime_detection_requirements.txt
```
**2. Daten Vorbereiten & Skript ausführen:**
Stelle sicher, dass sich die rohen Asset-Dateien (z.B. `EURUSD_H1.csv`) im Ordner `mt5_h1_daten/` befinden. Starte dann den Erkennungs-Prozess:
```
python3 regime_detection.py
```
**3. Ergebnisse:**
Nach erfolgreichem Durchlauf erstellt das Skript automatisch einen neuen Ordner namens `mt5_h1_daten_regimes/` im selben Verzeichnis.
Dort werden die verarbeiteten `.csv`-Dateien gespeichert. Diese enthalten nun die originalen Kursdaten sowie die neuen Spalten `log_return`, `atr_14` und `hmm_regime`. Diese Dateien dienen als statische Datengrundlage für das nachfolgende Portfolio-Optimierungs-Modul.
