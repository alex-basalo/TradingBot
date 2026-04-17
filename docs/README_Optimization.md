# Trading Bot - Portfolio & Hyperparameter Optimization

Dieses Modul bildet das "Gehirn" des algorithmischen Trading-Bots. Es nimmt die in Cluster unterteilten Assets sowie die durch das Hidden Markov Model (HMM) erkannten Marktregime und nutzt maschinelles Lernen, um für jedes Cluster die profitabelsten Handelsregeln zu finden.

Das Modul besteht aus einer modularen Architektur von drei zusammenhängenden Skripten, wobei der `master_loop.py` als primärer Ausführungspunkt dient.

## Funktionsweise der Architektur

Die Architektur trennt Indikatoren, Backtesting-Logik und Cluster-Verarbeitung strikt voneinander (Separation of Concerns):

1. **`indicators.py` (Der Werkzeugkasten):** Eine objektorientierte Klasse, die alle akademisch fundierten technischen Indikatoren bereitstellt. Dazu gehören Kategorie A (S/R-Zonen wie Keltner, VWAP, Pivot), Kategorie B (Momentum wie RSI, MACD, MFI) sowie optional Kategorie C (Trend-Filter wie EMA-Cross, ADX, Parabolic SAR).
2. **`optimizer.py` (Die Backtest-Engine):** Beinhaltet die Vektor-Logik für den Backtest (komplett ohne Look-ahead Bias) sowie das Setup für die Bayes'sche Optimierung (`optuna`). Hier ist zudem der Zwang zur statistischen Signifikanz verankert: Parameter-Kombinationen, die zu wenige Trades generieren, werden algorithmisch mit einem Penalty-Score (-1.0) bestraft.
3. **`master_loop.py` (Der Dirigent):** Das einzige Skript, das aktiv ausgeführt werden muss. Es liest die Cluster-Zuteilung (`bot_cluster_config.csv`) ein, lädt alle Asset-Daten gesammelt in den Arbeitsspeicher, wendet den Optimizer auf Cluster-Ebene (Portfolio-Pooling) an und speichert die finalen Parameter.

### Erweiterte Skript-Variationen (Robustheitsprüfung & Trends)
Um Overfitting zu vermeiden und tiefere Strategien zu testen, gibt es folgende Variationen:

* **`master_loop_reverse.py`:** Kehrt den Datensplit um. Anstatt auf den ersten 70% der Daten zu trainieren, trainiert dieser Loop auf den *letzten* 70% der Daten. Das Test-Sample (Out-of-Sample) liegt hierbei chronologisch am Anfang (die ersten 30%). Dies dient als eine Art Cross-Validation, um zu prüfen, ob die Strategien unabhängig vom gewählten Zeitraum stabil bleiben.
* **`optimizer_kat_c.py`:** Die erweiterte Version der Backtest-Engine. Dieser Optimizer zwingt den Algorithmus, neben Zonen (Kat A) und Momentum (Kat B) auch einen übergeordneten Trendfilter der **Kategorie C** in die Signallogik aufzunehmen. Der Bayes'sche Suchraum wird dadurch um Indikatoren wie EMA-Cross oder ADX und deren Längen erweitert.

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

**3. Skript konfigurieren (Wechsel zwischen Standard und Kategorie C):**
Standardmäßig nutzen die Master-Loops den `optimizer.py`. Wenn du stattdessen den erweiterten Suchraum mit Trendfiltern nutzen möchtest, musst du eine winzige Anpassung im Import-Bereich des jeweiligen Master-Loops (`master_loop.py` oder `master_loop_reverse.py`) vornehmen.

Öffne die Datei und ändere **Zeile 3**:

    Vorher: `from optimizer import ClusterOptimizer`

    Nachher: `from optimizer_kat_c import ClusterOptimizer`

Ebenso kann direkt in den Optimizer-Dateien via Flag (`USE_FRIDAY_EXIT = True/False`) eine Ablationsstudie durchgeführt werden, um den Einfluss von Wochenend-Risiken (Weekend Gaps) zu testen.

**4. Skript ausführen:**
Starte das gewünschte Setup über das Terminal:
```Bash

# Für das Standard-Training
python3 master_loop.py

# Für das umgekehrte Training (Cross-Validation)
python3 master_loop_reverse.py
```

## Ergebnisse

Nach erfolgreichem Durchlauf erstellt das Skript eine Parameter-Matrix im `.csv`-Format (z.B. `params_WITH_friday_exit.csv` oder `params_WITHOUT_friday_exit.csv`).

Diese Datei enthält für jedes der 12 Cluster die exakt abgestimmten, optimalen Hyperparameter (Indikator-Wahl, Längen, Stop-Loss-Multiplikatoren, Risk-Reward-Ratios und das ideale HMM-Zielregime). Diese `.csv`-Datei dient als finale Regelbasis für den Live-Trading-Bot (Out-of-Sample Test) bzw. das Evaluations-Modul.
