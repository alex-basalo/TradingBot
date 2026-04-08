# Trading Bot - Portfolio & Hyperparameter Optimization

Dieses Modul bildet das "Gehirn" des algorithmischen Trading-Bots. Es nimmt die in Cluster unterteilten Assets sowie die durch das Hidden Markov Model (HMM) erkannten Marktregime und nutzt maschinelles Lernen, um für jedes Cluster die profitabelsten Handelsregeln zu finden.

Das Modul besteht aus einer modularen Architektur von drei zusammenhängenden Skripten, wobei der `master_loop.py` als primärer Ausführungspunkt dient.

## Funktionsweise der Architektur

Die Architektur trennt Indikatoren, Backtesting-Logik und Cluster-Verarbeitung strikt voneinander (Separation of Concerns):

1. **`indicators.py` (Der Werkzeugkasten):** Eine objektorientierte Klasse, die alle akademisch fundierten technischen Indikatoren bereitstellt. Dazu gehören Kategorie A (S/R-Zonen wie Keltner, VWAP, Pivot), Kategorie B (Momentum wie RSI, MACD, MFI) sowie optional Kategorie C (Trend-Filter wie EMA-Cross, ADX, Parabolic SAR).
2. **`optimizer.py` (Die Backtest-Engine):** Beinhaltet die Vektor-Logik für den Backtest (komplett ohne Look-ahead Bias) sowie das Setup für die Bayes'sche Optimierung (`optuna`). Hier ist zudem der Zwang zur statistischen Signifikanz verankert: Parameter-Kombinationen, die zu wenige Trades generieren, werden algorithmisch mit einem Penalty-Score (-1.0) bestraft.
3. **`master_loop.py` (Der Dirigent):** Das einzige Skript, das aktiv ausgeführt werden muss. Es liest die Cluster-Zuteilung (`bot_cluster_config.csv`) ein, lädt alle Asset-Daten gesammelt in den Arbeitsspeicher, wendet den Optimizer auf Cluster-Ebene (Portfolio-Pooling) an und speichert die finalen Parameter.

## Installation und Ausführung

**1. Abhängigkeiten installieren:**
Installiere die benötigten Python-Bibliotheken über die bereitgestellte Requirements-Datei:
```bash
pip install -r optimization_requirements.txt
```
**2. Daten Vorbereiten:**
Stelle sicher, dass folgende Dateien im Verzeichnis vorliegen:
1. Die Konfigurationsdatei aus Phase 1 (`bot_cluster_config.csv`).
2. Die mit Regimen angereicherten `.csv`-Daten aus Phase 2 (im Ordner `mt5_h1_daten_regimes/`).

**3. Skript ausführen:**
Das Modul wird zentral über den Master-Loop gestartet. Im Code kann via Flag (`USE_FRIDAY_EXIT = True/False`) eine Ablationsstudie durchgeführt werden, um den Einfluss von Wochenend-Risiken (Weekend Gaps) zu testen.
```
python3 master_loop.py
```

## Ergebnisse

Nach erfolgreichem Durchlauf erstellt das Skript eine Parameter-Matrix im `.csv`-Format (z.B. `params_WITH_friday_exit.csv` oder `params_WITHOUT_friday_exit.csv`).

Diese Datei enthält für jedes der 12 Cluster die exakt abgestimmten, optimalen Hyperparameter (Indikator-Wahl, Längen, Stop-Loss-Multiplikatoren, Risk-Reward-Ratios und das ideale HMM-Zielregime). Diese `.csv`-Datei dient als finale Regelbasis für den Live-Trading-Bot (Out-of-Sample Test).
