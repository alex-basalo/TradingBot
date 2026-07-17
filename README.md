# Strict Walk-Forward-Analyse — Forex-Reversal-Handelssystem

Datengetriebenes Multi-Asset-Handelssystem für den Devisenmarkt (H1), entwickelt
im Rahmen einer Bachelorarbeit. Das System kombiniert eine reine Support-/
Resistance-Reversal-Strategie mit einem Hidden-Markov-Regimefilter, einem
Multi-Seed-Optuna-Ensemble und einem mehrstufigen Risikomanagement (State Machine
plus Circuit Breaker). Die Bewertung erfolgt über eine strikte, look-ahead-freie
Walk-Forward-Analyse mit ereignisgesteuertem Fenster-Shifting.

## Projektstruktur

```
src/                Quellcode des Handelssystems (7 Module, siehe unten)
tools/              Hilfsskripte: converter.py, monte_carlo.py
tests/              Unittests (test_engine.py, test_optimizer.py)
results/            je Durchlauf ein Unterordner mit allen Ausgabedateien
                    (Summary, Trades, FoldLog, AssetSummary, Grafiken)
requirements.txt    gepinnte Abhängigkeiten (Reproduzierbarkeit)
HistData_H1/        H1-CSV-Dateien je Währungspaar (NICHT im Repo, s. u.)
```

### Module in `src/`

| Datei            | Klasse                 | Inhalt                                                       |
|------------------|------------------------|-------------------------------------------------------------|
| `config.py`      | —                      | zentrale Konfiguration (`CONFIG`) + Logger-Setup            |
| `indicators.py`  | `IndicatorEngine`      | Swing-Zonen (Kat. A), RSI-Momentum (Kat. B)                 |
| `regime.py`      | `MarketRegimeDetector` | HMM-Regimeerkennung (Kat. C), look-ahead-frei               |
| `optimizer.py`   | `FoldOptimizer`        | Vektor-Backtest, Fitnesssynthese, Nachbarschaftstest, TPE   |
| `engine.py`      | `WalkForwardEngine`    | Orchestrierung, State Machine, Circuit Breaker, Datenladen  |
| `reporting.py`   | `ReportGenerator`      | Kennzahlen, Tabellen, Grafiken          |
| `main.py`        | —                      | Einstiegspunkt (`python src/main.py`)                       |

### Hilfsskripte in `tools/`

| Datei             | Zweck                                                                          |
|-------------------|--------------------------------------------------------------------------------|
| `converter.py`    | aggregiert HistData-M1-Rohdaten zu H1-CSVs (Datenaufbereitung)                 |
| `monte_carlo.py`  | Bootstrapping-Robustheitsanalyse (Fold-, Block- und Trade-Verfahren)           |


## Installation

```bash
pip install -r requirements.txt
```

Empfohlen: Python 3.11+ in einer virtuellen Umgebung.

## Daten

Die Strategie verarbeitet stündliche OHLC-Daten (H1) je Währungspaar als CSV mit
einer Datums-Spalte `time` und den Spalten `open, high, low, close`. Die Dateien
gehören in den Ordner `HistData_H1/` und sind nach dem Paar benannt
(z. B. `EURUSD_H1.csv`).

Die Rohdaten sind aus Größen- und Lizenzgründen **nicht** im Repository enthalten.
Sie stammen von [HistData.com](https://www.histdata.com) (kostenlose historische
M1-Daten, hier auf H1 aggregiert). Verwendet wurden die 28 Währungspaare der acht
frei konvertierbaren Hauptwährungen (USD, EUR, JPY, GBP, CHF, AUD, NZD, CAD) über
den Zeitraum 2017–2026.

## Datenaufbereitung

Die H1-CSVs werden mit `tools/converter.py` aus den HistData-M1-Rohdaten erzeugt.

1. M1-Rohdaten je Paar von HistData.com herunterladen (Generic ASCII, M1) und in
   den Ordner `HistData_M1/` legen (Unterordner sind erlaubt; das Skript
   durchsucht rekursiv).
2. `python tools/converter.py` ausführen. Das Skript gruppiert die Dateien
   automatisch nach Währungspaar, fügt alle Jahre zusammen, aggregiert auf
   Stundenkerzen und schreibt je Paar eine `{PAAR}_H1.csv`.
3. Die erzeugten H1-Dateien liegen anschließend in `HistData_H1/` und werden von
   `src/main.py` eingelesen.

Eingabe- und Ausgabeordner sind oben in `converter.py` über `INPUT_FOLDER` /
`OUTPUT_FOLDER` einstellbar.

## Ausführung

```bash
cd src
python main.py
```

Die Steuerung erfolgt vollständig über `CONFIG` in `src/config.py`. Relevante
Schalter:

- `EXPERIMENT_NAME` — Prefix aller Ausgabedateien
- `DATA_START` / `DATA_END` — Untersuchungszeitraum (Datumsfilter)
- `BANNED_ASSETS` — Liste zu ignorierender Paare (leer = alle gefundenen)
- `BASE_SEED` / `NUM_ENSEMBLES` — Multi-Seed-Ensemble
- `USE_HMM_FILTER` — Regimefilter (Kat. C) ein-/ausschalten (Ablation)
- `USE_NEIGHBOR_TEST` — Nachbarschaftstest der Fitness ein-/ausschalten (Ablation)
- `USE_FOLD_BRAKE` — globale Fold-Notbremse ein-/ausschalten (Ablation)
- `USE_RANDOM_ENTRY` / `ENTRY_SEED` — Kontrolltest: ersetzt die Bestätigungslogik
  (Momentum-Hook, Wick-Rejection) durch Zufallssignale gleicher Signalrate;
  der übrige Risiko-Apparat bleibt unverändert. Für eine belastbare Aussage
  mehrere Läufe mit verschiedenem `ENTRY_SEED`.

Jeder Lauf erzeugt `{NAME}_Summary.json`, `{NAME}_Trades.csv`,
`{NAME}_FoldLog.csv`, `{NAME}_AssetSummary.csv`, `{NAME}_Equity.png`,
`{NAME}_Underwater.png` und `{NAME}_FoldPnL.png` sowie eine angehängte Zeile in
`runs_comparison.csv`. Die Ergebnisdateien werden je Durchlauf unter
`results/{EXPERIMENT_NAME}/` abgelegt.

## Analyse

`tools/monte_carlo.py` führt eine Bootstrapping-Robustheitsanalyse auf den
Ergebnissen eines Laufs durch (Endkapital-, Drawdown- und Verlustverteilung über
10.000 Pfade). Über die Variable `METHOD` im Skript stehen drei Verfahren zur
Verfügung: `fold` (Ziehung ganzer Walk-Forward-Fenster, für die Ergebnis­unsicherheit),
`block` (Ziehung zusammenhängender Trade-Blöcke, für das Drawdown-Risiko) und
`iid` (Ziehung einzelner Trades, nur als Referenz). Vor dem Start `TRADE_FILE`
(und für `fold` zusätzlich `FOLD_FILE`) im Skript auf den gewünschten Lauf setzen,
dann `python tools/monte_carlo.py`.

## Tests

Die Unittests in `tests/` sichern die Funktionen ab, deren Korrektheit
unmittelbar in die berichteten Ergebnisse eingeht:

- `test_engine.py` — Portfolio-Limit, Kennzahlenberechnung (PnL, Profit-Faktor,
  Drawdown) und die risikoadjustierten Maße samt t-Statistik.
- `test_optimizer.py` — Nachweis der Look-ahead-Freiheit der iterativen
  Regimevorhersage sowie die konservative Intrabar-Auflösung (Stop-Loss vor
  Take-Profit).

## Reproduzierbarkeit

Die exakten Bibliotheksversionen sind in `requirements.txt` gepinnt und werden
zusätzlich in jeder `_Summary.json` mitprotokolliert; sofern das Projekt als
Git-Repository vorliegt, wird zudem der Commit-Hash des Laufs festgehalten. Bei
identischer Datenbasis, identischer `CONFIG` und identischen Versionen ist ein
Lauf deterministisch reproduzierbar (feste Seeds je Optuna-Studie und für die
Portfolio-Zuteilung).
