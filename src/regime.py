"""
Unueberwachte Regime-Erkennung mittels Hidden-Markov-Modell (Kategorie C).
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from hmmlearn.hmm import GaussianHMM

from config import CONFIG


class MarketRegimeDetector:
    """
    Kapselt die unueberwachte Regime-Erkennung mittels eines Hidden-Markov-Modells (HMM).
    Dient als richtungsneutraler Marktphasen-Filter (Kategorie C).
    """
    def __init__(self, random_state: int = 42):
        self.n_states = CONFIG["HMM_STATES"]
        self.iterations = CONFIG["HMM_ITERATIONS"]
        self.random_state = random_state
        self.best_model = None
        self.state_map = {}

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Berechnet geometrische Merkmale fuer das HMM (Choppiness, SMA-Distanz, ATR). """
        df_feat = df.copy()
        df_feat['chop'] = ta.chop(df_feat['high'], df_feat['low'], df_feat['close'], length=14)

        sma50 = df_feat['close'].rolling(50).mean()
        df_feat['dist_sma'] = abs(df_feat['close'] - sma50) / sma50
        df_feat['atr_14'] = ta.atr(df_feat['high'], df_feat['low'], df_feat['close'], length=14)

        return df_feat.ffill().fillna(0)

    def fit(self, df_train: pd.DataFrame):
        """ Trainiert das Modell ueber den Baum-Welch-Algorithmus auf In-Sample-Daten. """
        df_features = self._calculate_features(df_train)
        X = df_features[['chop', 'dist_sma']].values

        model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=self.iterations, random_state=self.random_state)
        try:
            model.fit(X)
            self.best_model = model
            means = model.means_[:, 0]
            sorted_states = np.argsort(means)[::-1]
            self.state_map = {old_state: new_state for new_state, old_state in enumerate(sorted_states)}
        except Exception:
            self.best_model = None

    def predict_iterative(self, df: pd.DataFrame, step: int = 24) -> pd.DataFrame:
        """ Generiert rollierend Viterbi-Zustandsschaetzungen zur Vermeidung von Vorausschau-Bias. """
        df_features = self._calculate_features(df)
        X = df_features[['chop', 'dist_sma']].values
        n = len(df)
        aligned_states = np.zeros(n)

        if self.best_model is not None:
            last_state = 0
            for i in range(0, n, step):
                current_X = X[:i+1]
                if len(current_X) > 10:
                    raw_states = self.best_model.predict(current_X)
                    last_state = self.state_map.get(raw_states[-1], 0)
                chunk_end = min(i + step, n)
                aligned_states[i:chunk_end] = last_state

        df_result = df.copy()
        df_result['hmm_regime'] = aligned_states
        df_result['atr_14'] = df_features['atr_14']
        return df_result
