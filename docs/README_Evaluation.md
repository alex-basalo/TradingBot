# Trading Bot - Evaluation & Ensemble

Dieses Modul dient der quantitativen Auswertung von Handelsstrategien. Es wandelt die abstrakten Parameter-Ergebnisse aus dem `master_loop` in greifbare Performance-Metriken, visuelle Equity-Kurven und Logbuch-Einträge um.

## Funktionsweise der Evaluation

Nachdem durch den Optimierungsprozess (`master_loop.py` / `master_loop_reverse.py`) optimierte Parameter-Dateien (z.B. `params_WITH_friday_exit.csv`) erstellt wurden, übernimmt dieses Modul die detaillierte Analyse. 

Es gibt zwei Haupt-Skripte:

1.  **`experiment_evaluation.py`**: Evaluiert ein einzelnes Regelwerk. Es simuliert die Trades über alle Assets eines Clusters hinweg und berechnet den Profit, Drawdown sowie Sharpe- und Sortino-Ratios für die In-Sample- (Train) und Out-of-Sample-Phase (Test).
2.  **`ensemble_evaluator.py`**: Erstellt ein Hybrid-Portfolio. Da Cluster unterschiedlich auf einen vorzeitigen Wochenschluss reagieren, vergleicht dieses Skript pro Cluster die Ergebnisse mit und ohne "Friday Exit". Es wählt automatisch die stabilere Kombination (höhere Train-Sharpe) und fusioniert sie zu einem Gesamt-Portfolio.

### Skript-Variationen
Um alle Optimierungs-Szenarien abzudecken, stehen jeweils vier Versionen zur Verfügung:
* **Standard**: Training auf den ersten 70% der Daten, Test auf den letzten 30%. Berücksichtigt Indikatoren der Kategorien A & B.
* **`_reverse`**: Umgekehrter Split (Test auf den ersten 30%, Training auf den letzten 70%).
* **`_c`**: Berücksichtigt zusätzlich die Kategorie C (Trend-Indikatoren) für die Strategie-Logik.
* **`_c_reverse`**: Die Fusion aus umgekehrtem Datensplit und aktiver Kategorie C.

## Installation und Ausführung

**1. Abhängigkeiten installieren:**
Installiere die benötigten Bibliotheken. (Hinweis: Erstelle eine `evaluation_requirements.txt` mit den Einträgen `pandas`, `numpy` und `matplotlib` im selben Verzeichnis).
```bash
pip install -r evaluation_requirements.txt
```

**2. Skript konfigurieren:**
Vor dem Start müssen im `if __name__ == "__main__":`-Block am Ende der jeweiligen Datei Anpassungen vorgenommen werden:

* **Bei `experiment_evaluation.py`**:
    * `EXPERIMENT`: Name des Durchlaufs (wird für den Plot und das Logbuch verwendet).
    * `PARAMETER_DATEI`: Der Pfad zur CSV aus dem Master-Loop.
* **Bei `ensemble_evaluator.py`**:
    * `DATEI_MIT_EXIT` & `DATEI_OHNE_EXIT`: Hier werden beide korrespondierenden CSVs eingetragen, um die beste Kombination zu finden.

**3. Skript ausführen:**
```bash
# Beispiel für eine Standard-Evaluation
python3 experiment_evaluation.py

# Beispiel für eine Ensemble-Evaluation
python3 ensemble_evaluator.py
```

## Ergebnisse

Nach der Ausführung generiert das Modul folgende Ergebnisse:

1.  **Equity-Graph (`/experiment_plots`)**: Ein hochauflösender Plot, der die Kapitalkurve zeigt. Die Trennung zwischen Train- und Test-Phase ist farblich markiert, inklusive einer Infobox mit allen relevanten Kennzahlen (Sharpe, Profit, Winrate, Max DD).
2.  **Logbuch (`experiment_logbook.csv`)**: Jedes Experiment wird automatisch mit seinen Metriken an diese CSV angehängt, um die Vergleichbarkeit zwischen verschiedenen Optimierungs-Einstellungen zu gewährleisten.
3.  **Monte Carlo Export (`[Name]_OOS_MC_Trades.csv`)**: (Nur bei Ensemble) Erzeugt eine Liste aller R-Multiples der Out-of-Sample Phase. Diese Datei kann direkt für spätere Monte Carlo Simulationen genutzt werden, um die Robustheit der Strategie statistisch zu prüfen.
