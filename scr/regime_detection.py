import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
import os
import glob

# Warnungen von hmmlearn unterdrücken (oft bei Konvergenz-Tests üblich)
warnings.filterwarnings("ignore")

class MarketRegimeDetector:
    """
    Klasse zur dynamischen Erkennung von Marktphasen (Regimen) mittels Hidden Markov Models (HMM).
    
    Diese Klasse berechnet zunächst die notwendigen Features (Log-Renditen und Volatilität) 
    aus rohen OHLCV-Daten. Anschließend trainiert sie probabilistische Modelle, um verborgene 
    Marktzustände zu identifizieren. Sie evaluiert verschiedene Zustandsanzahlen (z.B. 3 oder 4) 
    und wählt das optimale Modell anhand des Bayesian Information Criterion (BIC) aus.
    """
    
    def __init__(self, min_states: int = 3, max_states: int = 4, random_state: int = 42):
        """
        Initialisiert den HMM-Detector.
        
        Args:
            min_states (int): Minimale Anzahl der zu testenden Hidden States (Regime).
            max_states (int): Maximale Anzahl der zu testenden Hidden States.
            random_state (int): Seed für reproduzierbare Ergebnisse.
        """
        self.min_states = min_states
        self.max_states = max_states
        self.random_state = random_state
        self.best_model = None
        self.optimal_states = None
        
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet die benötigten Features (Log-Returns und ATR) für das HMM aus den Rohdaten.
        
        Args:
            df (pd.DataFrame): DataFrame mit rohen H1-Kerzendaten (benötigt 'high', 'low', 'close').
                               
        Returns:
            pd.DataFrame: Ein bereinigter DataFrame inkl. 'log_return' und 'atr_14'.
        """
        df_feat = df.copy()
        
        # 1. Logarithmische Renditen (Log-Returns) berechnen
        df_feat['log_return'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
        
        # 2. Average True Range (ATR) berechnen (Standard-Periode: 14)
        df_feat['prev_close'] = df_feat['close'].shift(1)
        df_feat['tr1'] = df_feat['high'] - df_feat['low']
        df_feat['tr2'] = abs(df_feat['high'] - df_feat['prev_close'])
        df_feat['tr3'] = abs(df_feat['low'] - df_feat['prev_close'])
        
        df_feat['true_range'] = df_feat[['tr1', 'tr2', 'tr3']].max(axis=1)
        df_feat['atr_14'] = df_feat['true_range'].rolling(window=14).mean()
        
        # Hilfsspalten entfernen und NaNs droppen
        df_feat.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        df_feat.dropna(inplace=True)
        
        return df_feat

    def _calculate_bic(self, model: GaussianHMM, X: np.ndarray) -> float:
        """
        Berechnet das Bayesian Information Criterion (BIC) für ein trainiertes HMM.
        """
        log_likelihood = model.score(X)
        n_features = X.shape[1]
        n_states = model.n_components
        
        # Formel für Parameter im Gaussian HMM mit diagonaler Kovarianz
        n_params = n_states**2 + 2 * n_states * n_features - 1
        n_samples = X.shape[0]
        
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        return bic

    def fit(self, df: pd.DataFrame) -> None:
        """
        Trainiert HMMs mit verschiedenen Zustandsanzahlen und speichert das beste Modell.
        """
        df_features = self._calculate_features(df)
        X = df_features[['log_return', 'atr_14']].values
        best_bic = np.inf
        
        for n_states in range(self.min_states, self.max_states + 1):
            model = GaussianHMM(
                n_components=n_states, 
                covariance_type="diag", 
                n_iter=1000, 
                random_state=self.random_state
            )
            
            try:
                model.fit(X)
                bic = self._calculate_bic(model, X)
                
                if bic < best_bic:
                    best_bic = bic
                    self.best_model = model
                    self.optimal_states = n_states
            except Exception:
                pass # Falls ein Modell nicht konvergiert, wird es leise übersprungen

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sagt das Marktregime für jede Kerze voraus.
        """
        if self.best_model is None:
            raise ValueError("Das Modell muss zuerst mit fit() trainiert werden!")
            
        df_features = self._calculate_features(df)
        X = df_features[['log_return', 'atr_14']].values
        
        # Verborgene Zustände vorhersagen
        hidden_states = self.best_model.predict(X)
        
        # Die Vorhersage als neue Spalte anhängen
        df_features['hmm_regime'] = hidden_states
        
        return df_features

# --- Batch-Verarbeitung für alle Assets ---
if __name__ == "__main__":
    input_folder = "mt5_h1_daten"
    output_folder = "mt5_h1_daten_regimes"
    
    # Ausgabeordner erstellen, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Alle CSV-Dateien im Input-Ordner finden
    csv_files = glob.glob(f"{input_folder}/*.csv")
    
    if not csv_files:
        print(f"Keine CSV-Dateien im Ordner '{input_folder}' gefunden.")
    else:
        print(f"Starte Regime-Erkennung für {len(csv_files)} Assets...\n")
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            print(f"Verarbeite: {filename}")
            
            # Daten laden
            df = pd.read_csv(file_path, index_col='time', parse_dates=True)
            
            # Detektor initialisieren (testet 3 vs 4 Zustände)
            detector = MarketRegimeDetector(min_states=3, max_states=4)
            
            # Modell trainieren
            detector.fit(df)
            print(f" -> Optimales Modell gewählt: {detector.optimal_states} Zustände")
            
            # Regimes vorhersagen (gibt DataFrame mit neuen Features + 'hmm_regime' zurück)
            df_with_regimes = detector.predict(df)
            
            # In neuen Ordner speichern
            output_path = os.path.join(output_folder, filename)
            df_with_regimes.to_csv(output_path)
            
        print(f"\nFertig! Alle Dateien wurden inkl. 'hmm_regime' in '{output_folder}' gespeichert.")
