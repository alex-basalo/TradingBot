"""
Einstiegspunkt der Strict Walk-Forward-Analyse.

STRICT WALK FORWARD ANALYSIS (0% Look-Ahead Bias)
Version: PURE S/R REVERSAL | HMM Regime Filter + Multi-Seed Ensemble + Circuit Breaker

Aufruf:  python main.py
"""

from engine import WalkForwardEngine


if __name__ == "__main__":
    WalkForwardEngine.run_strict_wfa()
