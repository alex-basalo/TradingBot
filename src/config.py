"""
Globale System-Konfiguration und Logger-Setup.

Dieses Modul kapselt ausschließlich die Konfiguration. Es importiert keine
projekteigenen Module und ist daher die unterste Schicht: alle anderen Module
importieren CONFIG von hier.
"""

import logging
import warnings

import optuna

# =================================================================================
# GLOBALE SYSTEM-KONFIGURATION
# =================================================================================
CONFIG = {
    "EXPERIMENT_NAME": "FF1_Main_2020-2026_28A_HistData",
    "RAW_DATA_DIR": "HistData_H1",

    "DATA_START": "2020-03-31",  
    "DATA_END":   "2026-03-31",

    # Assets, die für den Lauf ignoriert/gebannt werden sollen.
    "BANNED_ASSETS": [],

    # Ablations-Schalter (FF4)
    "USE_HMM_FILTER": True,      
    "USE_NEIGHBOR_TEST": True,   
    "USE_FOLD_BRAKE": True, 

    # Random-Entry-Baseline (Kontrolltest)
    "USE_RANDOM_ENTRY": False,
    "ENTRY_SEED": 1,   

    "MAX_OPEN_TRADES": 10,
    "MAX_LEVERAGE": 30.0,

    # Walk-Forward Fenster
    "TRAIN_MONTHS": 15,
    "INNER_VAL_MONTHS": 3,
    "TEST_MONTHS": 2,

    "BASE_SEED": 42,
    "NUM_ENSEMBLES": 10,

    # Robustheits-Schwellenwerte
    "MIN_TRAIN_SHARPE": 0.05,
    "MAX_OOS_LOSS_PER_ASSET": -200.0,

    # Hidden Markov Model Parameter
    "HMM_STATES": 2,
    "HMM_ITERATIONS": 500,

    # Optuna Ensemble Parameter
    "TRIALS_PER_FOLD": 250,
    "MIN_TRADES_PER_ASSET": 1,
    "MIN_ACTIVE_ASSETS_PERCENT": 0.1,

    # Suchräume für Optuna
    "OPT_SWING_LEN": (130, 240),
    "OPT_ZONE_TOL_SWING": (0.1, 0.8),
    "OPT_RSI_LEN": (14, 28),
    "OPT_RSI_OVERSOLD": (20, 35),
    "OPT_RSI_OVERBOUGHT": (65, 80),
    "OPT_RRR_SR": (1.0, 1.5),
    "OPT_SL_MULT_SR": (1.5, 3.0),

    # Kapital & Risikomanagement
    "START_CAPITAL": 10000.0,
    "RISK_PER_TRADE": 60.0,
    "FEE_RATE": 0.000035,
    "SL_SLIPPAGE": 0.1,
    "SPREAD_TRIGGER": 0.0002,
}

# Logger und Warnungen stummschalten
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
