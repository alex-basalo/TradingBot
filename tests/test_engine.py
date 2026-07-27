"""
Unittests für die Kernlogik der Orchestrierung und der Kennzahlenberechnung.

Geprüft werden jene Funktionen, deren Korrektheit unmittelbar in die berichteten
Ergebnisse eingeht: die Einhaltung des Portfolio-Limits, die Kennzahlenberechnung
(PnL, Profit-Faktor, Drawdown) sowie die risikoadjustierten Masse samt
t-Statistik.
"""

import unittest

import numpy as np
import pandas as pd

from config import CONFIG
from engine import WalkForwardEngine
from reporting import ReportGenerator


def make_trade(entry_t, exit_t, r_mult, risk, fee, notional, asset="EURUSD"):
    """
    Erzeugt ein Trade-Tupel in der vom System verwendeten Struktur.
    Die ersten sechs Felder entsprechen der von calc_kpis und
    apply_portfolio_limit erwarteten Reihenfolge; das letzte Feld ist der
    Asset-Name (Position -1), wie ihn die Fold-Notbremse ausliest.
    """
    return (entry_t, exit_t, r_mult, risk, fee, notional, asset)


class TestPortfolioLimit(unittest.TestCase):
    """apply_portfolio_limit sichert das Limit gleichzeitig offener Positionen."""

    def test_leere_liste(self):
        self.assertEqual(WalkForwardEngine.apply_portfolio_limit([], 10), [])

    def test_limit_wird_nie_ueberschritten(self):
        # 20 Trades, die sich ALLE zeitlich vollständig überlappen
        # (Entry 0, Exit 100). Bei max_concurrent=3 dürfen höchstens 3
        # akzeptiert werden.
        notional = 1000.0
        trades = [make_trade(0.0, 100.0, 1.0, 60.0, 1.0, notional,
                             asset=f"P{i}") for i in range(20)]
        accepted = WalkForwardEngine.apply_portfolio_limit(trades, 3)
        self.assertLessEqual(len(accepted), 3)

    def test_sequentielle_trades_alle_akzeptiert(self):
        # Nicht überlappende Trades dürfen trotz kleinem Limit alle
        # akzeptiert werden, da nie mehr als einer gleichzeitig offen ist.
        notional = 1000.0
        trades = [make_trade(float(i), float(i) + 0.5, 1.0, 60.0, 1.0,
                             notional, asset=f"P{i}") for i in range(10)]
        accepted = WalkForwardEngine.apply_portfolio_limit(trades, 2)
        self.assertEqual(len(accepted), 10)

    def test_margin_grenze_wird_eingehalten(self):
        # Notional so groß, dass die Margin zweier gleichzeitiger Trades
        # das Startkapital übersteigt -> nur einer darf durchkommen.
        huge = CONFIG["START_CAPITAL"] * CONFIG["MAX_LEVERAGE"]
        trades = [
            make_trade(0.0, 100.0, 1.0, 60.0, 1.0, huge, asset="A"),
            make_trade(0.0, 100.0, 1.0, 60.0, 1.0, huge, asset="B"),
        ]
        accepted = WalkForwardEngine.apply_portfolio_limit(trades, 10)
        self.assertEqual(len(accepted), 1)


class TestCalcKpis(unittest.TestCase):
    """calc_kpis berechnet die fundamentalen Portfolio-Kennzahlen korrekt."""

    def test_leere_trades(self):
        kpi = WalkForwardEngine.calc_kpis([])
        self.assertEqual(kpi["profit"], 0.0)
        self.assertEqual(kpi["win_rate"], 0.0)
        self.assertEqual(kpi["equity"], [CONFIG["START_CAPITAL"]])

    def test_profit_summe(self):
        # Zwei Gewinner (+2R und +1R bei Risiko 60, Gebühr 1) und ein
        # Verlierer (-1R). PnL = (2*60-1) + (1*60-1) + (-1*60-1) = 119+59-61 = 117
        trades = [
            make_trade(0.0, 1.0, 2.0, 60.0, 1.0, 1000.0),
            make_trade(1.0, 2.0, 1.0, 60.0, 1.0, 1000.0),
            make_trade(2.0, 3.0, -1.0, 60.0, 1.0, 1000.0),
        ]
        kpi = WalkForwardEngine.calc_kpis(trades)
        self.assertAlmostEqual(kpi["profit"], 117.0, places=6)

    def test_win_rate(self):
        # 2 von 4 Trades positiv -> 50 %
        trades = [
            make_trade(0.0, 1.0, 1.0, 60.0, 1.0, 1000.0),
            make_trade(1.0, 2.0, 1.0, 60.0, 1.0, 1000.0),
            make_trade(2.0, 3.0, -1.0, 60.0, 1.0, 1000.0),
            make_trade(3.0, 4.0, -1.0, 60.0, 1.0, 1000.0),
        ]
        kpi = WalkForwardEngine.calc_kpis(trades)
        self.assertAlmostEqual(kpi["win_rate"], 50.0, places=6)

    def test_max_drawdown(self):
        # PnL-Folge: +100, -300, +50. Equity ausgehend vom Startkapital:
        # Peak nach +100, dann Rücksetzer um 300 -> Max-DD = 300.
        # Gebühr 0, damit r_mult*risk exakt die PnL ergibt.
        trades = [
            make_trade(0.0, 1.0, 100.0, 1.0, 0.0, 1000.0),
            make_trade(1.0, 2.0, -300.0, 1.0, 0.0, 1000.0),
            make_trade(2.0, 3.0, 50.0, 1.0, 0.0, 1000.0),
        ]
        kpi = WalkForwardEngine.calc_kpis(trades)
        self.assertAlmostEqual(kpi["max_dd"], 300.0, places=6)


class TestRiskMetrics(unittest.TestCase):
    """
    ReportGenerator._risk_metrics liefert Sharpe, Sortino und t-Statistik.
    Geprüft wird gegen eine synthetische Renditereihe mit bekanntem Vorzeichen.
    """

    def _tage(self, n, start="2021-01-04"):
        # Geschäftstage ab einem Montag
        return pd.bdate_range(start, periods=n)

    def test_leere_trades(self):
        rm = ReportGenerator._risk_metrics([], CONFIG["START_CAPITAL"])
        self.assertIsNone(rm["sharpe"])
        self.assertIsNone(rm["t_stat"])

    def test_t_statistik_konventionsunabhaengig(self):
        # Die t-Statistik hängt nicht vom Annualisierungsfaktor ab; sie muss
        # sich aus mean/std * sqrt(n) der Tagesrenditen reproduzieren lassen.
        tage = self._tage(30)
        rng = np.random.RandomState(0)
        pnls = rng.normal(5.0, 20.0, size=len(tage))
        trades = [make_trade(t, t, float(p), 1.0, 0.0, 1000.0)
                  for t, p in zip(tage, pnls)]
        rm = ReportGenerator._risk_metrics(trades, CONFIG["START_CAPITAL"])

        rets = pnls / CONFIG["START_CAPITAL"]
        expected_t = rets.mean() / rets.std(ddof=1) * np.sqrt(len(rets))
        self.assertAlmostEqual(rm["t_stat"], expected_t, places=4)


class TestPnlHelper(unittest.TestCase):
    """Die PnL-Definition ist über alle Module hinweg identisch."""

    def test_pnl_formel(self):
        # PnL = r_mult * risk - fee
        t = make_trade(0.0, 1.0, 2.0, 60.0, 1.5, 1000.0)
        self.assertAlmostEqual(ReportGenerator._pnl(t), 2.0 * 60.0 - 1.5,
                               places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
