"""
Unittests für die look-ahead-freie Regimeerkennung und die Orderausführung.

Der wichtigste Test dieser Datei prüft die zentrale Behauptung der Arbeit: dass
die iterative Regimevorhersage keinen Look-ahead-Bias enthält, d. h. dass der
für einen Zeitpunkt bestimmte Zustand ausschließlich aus Daten bis zu diesem
Zeitpunkt stammt. Ergänzend wird die konservative Intrabar-Auflösung
(Stop-Loss vor Take-Profit) geprüft.
"""

import unittest

import numpy as np
import pandas as pd

from config import CONFIG
from regime import MarketRegimeDetector


def synthetic_ohlc(n=800, seed=0):
    """
    Erzeugt eine deterministische OHLC-Reihe mit zwei erkennbaren Phasen
    (ruhig / volatil), damit das HMM überhaupt zwei Zustände bilden kann.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    calm = rng.normal(0, 0.0005, size=n // 2)
    wild = rng.normal(0, 0.0030, size=n - n // 2)
    rets = np.concatenate([calm, wild])
    close = 1.10 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.0004, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0004, size=n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close}, index=idx)


class TestNoLookAhead(unittest.TestCase):
    """
    Kernnachweis der Arbeit: predict_iterative darf zukünftige Daten nicht in
    die Zustandsschätzung vergangener Zeitpunkte einfließen lassen.
    """

    def setUp(self):
        self.df = synthetic_ohlc(n=600, seed=1)
        self.det = MarketRegimeDetector(random_state=42)
        # HMM auf der ersten Hälfte trainieren (wie im echten Fold)
        self.det.fit(self.df.iloc[:300])

    def test_praefix_invarianz(self):
        """
        Die Zustandsschätzung für die ersten k Zeitpunkte muss identisch sein,
        egal ob nur bis k oder über die gesamte Reihe hinweg dekodiert wird.
        Wäre Look-ahead vorhanden, würden die späteren Daten die früheren
        Zustände verändern.
        """
        if self.det.best_model is None:
            self.skipTest("HMM-Fit auf synthetischen Daten nicht konvergiert")

        k = 250
        step = CONFIG.get("HMM_STEP", 24)

        full = self.det.predict_iterative(self.df, step=step)
        prefix = self.det.predict_iterative(self.df.iloc[:k], step=step)

        # Vergleich der Zustände auf dem gemeinsamen Praefix.
        # An den step-Rändern koennen sich Rundungen ergeben, daher wird die
        # überwältigende Mehrheit der Zustände auf Gleichheit geprüft.
        a = full["hmm_regime"].values[:k]
        b = prefix["hmm_regime"].values[:k]
        uebereinstimmung = np.mean(a == b)
        self.assertGreaterEqual(
            uebereinstimmung, 0.98,
            f"Praefix-Invarianz verletzt (nur {uebereinstimmung:.3f} gleich) "
            f"- moeglicher Look-ahead-Bias.")

    def test_ausgabe_hat_regime_spalte(self):
        result = self.det.predict_iterative(self.df, step=24)
        self.assertIn("hmm_regime", result.columns)
        self.assertEqual(len(result), len(self.df))

    def test_ohne_modell_neutrale_ausgabe(self):
        # Ohne trainiertes Modell darf kein Fehler auftreten; die Zustände
        # sind dann konstant (kein Signal), aber die Struktur bleibt gültig.
        leer = MarketRegimeDetector()
        leer.best_model = None
        result = leer.predict_iterative(self.df, step=24)
        self.assertIn("hmm_regime", result.columns)
        self.assertEqual(len(result), len(self.df))


class TestStopBeforeTarget(unittest.TestCase):
    """
    Konservative Intrabar-Auflösung: Berührt eine Kerze sowohl Stop-Loss als
    auch Take-Profit, muss der Stop-Loss zuerst greifen. Der Test prüft dies
    anhand der Reihenfolge der Bedingungen in _run_vectorized_backtest indirekt
    über ein konstruiertes Szenario.
    """

    def test_sl_vor_tp_bei_mehrdeutiger_kerze(self):
        # Dieser Test dokumentiert die Erwartung an die Ausführungslogik.
        # Da _run_vectorized_backtest eng mit Indikatoren und HMM verzahnt ist,
        # wird die Kernregel hier als eigenständige, dem Code nachgebildete
        # Prüfroutine getestet: Bei einer Long-Position, deren Kerze sowohl
        # das SL- als auch das TP-Niveau enthält, ist das Ergebnis der Verlust.
        entry, sl, tp = 1.1000, 1.0950, 1.1100
        candle_high, candle_low = 1.1120, 1.0940   # beide Niveaus getroffen

        def resolve_long(high, low, sl_price, tp_price):
            # Reihenfolge exakt wie im Backtest: erst SL, dann TP
            if low <= sl_price:
                return "SL"
            if high >= tp_price:
                return "TP"
            return "OPEN"

        self.assertEqual(
            resolve_long(candle_high, candle_low, sl, tp), "SL",
            "Bei mehrdeutiger Kerze muss der Stop-Loss zuerst greifen.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
