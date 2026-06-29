"""
Technische Indikatoren fuer das Reversal-System.

Kapselt die Zonenberechnung (Kategorie A) und das RSI-Momentum (Kategorie B).
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


class IndicatorEngine:
    """
    Stellt die bereinigten technischen Indikatoren fuer das Reversal-System bereit.
    Kapselt die Zonenberechnung (Kategorie A) und das Momentum (Kategorie B).
    """

    @staticmethod
    def add_swing_zones(df: pd.DataFrame, length: int) -> pd.DataFrame:
        """
        Kategorie A: Berechnet statische Struktur-Zonen (Support/Resistance).
        Findet lokale Hochs/Tiefs und zieht diese als flache horizontale Linien nach rechts.
        """
        df_ind = df.copy()
        window = (length * 2) + 1

        # Rollendes Maximum/Minimum der Periode finden
        df_ind['roll_max'] = df_ind['high'].rolling(window=window, center=False).max()
        df_ind['roll_min'] = df_ind['low'].rolling(window=window, center=False).min()

        # Bestaetigung erst 'length' Kerzen spaeter, um Look-Ahead-Bias zu vermeiden
        is_swing_high = df_ind['high'].shift(length) == df_ind['roll_max']
        is_swing_low = df_ind['low'].shift(length) == df_ind['roll_min']

        df_ind['swing_high_val'] = np.where(is_swing_high, df_ind['high'].shift(length), np.nan)
        df_ind['swing_low_val'] = np.where(is_swing_low, df_ind['low'].shift(length), np.nan)

        df_ind['zone_upper'] = df_ind['swing_high_val'].ffill()
        df_ind['zone_lower'] = df_ind['swing_low_val'].ffill()

        df_ind.drop(columns=['roll_max', 'roll_min', 'swing_high_val', 'swing_low_val'], inplace=True)
        return df_ind

    @staticmethod
    def add_rsi_momentum(df: pd.DataFrame, length: int) -> pd.DataFrame:
        """
        Kategorie B: Berechnet den Relative Strength Index (RSI) als Reversal-Bestaetigung.
        """
        df_ind = df.copy()
        df_ind['momentum_main'] = ta.rsi(df_ind['close'], length=length)
        return df_ind
