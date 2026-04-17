import pandas as pd
import numpy as np
import optuna
import os
import warnings
from indicators import IndicatorBaukasten

optuna.logging.set_verbosity(optuna.logging.INFO)
warnings.filterwarnings("ignore")

# --- GLOBALE EINSTELLUNGEN ---
USE_FRIDAY_EXIT = True # Schließt alle offenen Trades am Freitag um 21:00 Uhr
MIN_TRADES_PER_ASSET = 30 # Wie viele Trades MUSS jedes Asset im Schnitt mindestens machen?

class ClusterOptimizer:
    """
    Modul 4: Der finale Bayes'sche Optimizer.
    Nutzt Portfolio-Pooling für das ganze Cluster und integriert den Friday-Exit.
    INKLUSIVE KATEGORIE C (Trendfilter).
    """
    
    def __init__(self, data_dict: dict, cluster_id: int):
        self.data_dict = data_dict
        self.cluster_id = cluster_id
        self.baukasten = IndicatorBaukasten()
        
    def _run_vectorized_backtest(self, df: pd.DataFrame, params: dict) -> list:
        """
        Führt den Backtest durch und gibt eine Liste ALLER Trade-Renditen zurück, 
        anstatt sie sofort auszuwerten.
        """
        # 1. Zone hinzufügen (Kat A)
        df_sim = self.baukasten.add_category_a_zones(
            df, indicator_name=params['cat_a_name'], length=params['cat_a_length'],
            std=params.get('cat_a_std', 2.0), scalar=params.get('cat_a_scalar', 1.5)
        )
        # 2. Momentum hinzufügen (Kat B)
        df_sim = self.baukasten.add_category_b_momentum(
            df_sim, indicator_name=params['cat_b_name'], length=params['cat_b_length']
        )
        # 3. Trendfilter hinzufügen (Kat C)
        df_sim = self.baukasten.add_category_c_trend(
            df_sim, indicator_name=params['cat_c_name'], length=params['cat_c_length']
        )
        
        df_sim.dropna(inplace=True)
        if len(df_sim) < 100:
            return []

        closes = df_sim['close'].values
        highs = df_sim['high'].values
        lows = df_sim['low'].values
        opens = df_sim['open'].values
        
        zone_lower = df_sim['zone_lower'].values
        zone_upper = df_sim['zone_upper'].values
        momentum = df_sim['momentum_main'].values
        trend = df_sim['trend_main'].values  
        
        atrs = df_sim['atr_14'].values
        regimes = df_sim['hmm_regime'].values
        
        # Friday Exit Bedingung: Ist es Freitag (Tag 4) und 21 Uhr oder später?
        is_friday_close = np.array((df_sim.index.dayofweek == 4) & (df_sim.index.hour >= 21))
        
        target_regime = params['target_regime']
        sl_multiplier = params['sl_multiplier']
        rrr = params['rrr']
        
        # Oszillator vs. Nulllinie (MACD)
        if params['cat_b_name'] == 'macd':
            mom_oversold_cond = momentum < 0
            mom_overbought_cond = momentum > 0
        else:
            mom_oversold_cond = momentum < params.get('mom_oversold', 30)
            mom_overbought_cond = momentum > params.get('mom_overbought', 70)

        # Signale kombinieren (Regime + Zone + Momentum + Trend)
        long_signals = (regimes == target_regime) & (closes < zone_lower) & mom_oversold_cond & (trend == 1)
        short_signals = (regimes == target_regime) & (closes > zone_upper) & mom_overbought_cond & (trend == -1)
        
        long_entries = np.roll(long_signals, 1)
        short_entries = np.roll(short_signals, 1)
        long_entries[0] = False
        short_entries[0] = False
        
        trade_returns = []
        in_trade = False
        trade_type = 0
        entry_price = sl_price = tp_price = 0.0
        
        # Vektorisierte Auswertung
        for i in range(1, len(closes)):
            if in_trade:
                # --- FRIDAY EXIT LOGIK ---
                if USE_FRIDAY_EXIT and is_friday_close[i]:
                    if trade_type == 1:
                        trade_returns.append((closes[i] - entry_price) / entry_price)
                    else:
                        trade_returns.append((entry_price - closes[i]) / entry_price)
                    in_trade = False
                    continue 
                    
                # --- NORMALE SL/TP LOGIK ---
                if trade_type == 1: # Long
                    if lows[i] <= sl_price:
                        trade_returns.append((sl_price - entry_price) / entry_price)
                        in_trade = False
                    elif highs[i] >= tp_price:
                        trade_returns.append((tp_price - entry_price) / entry_price)
                        in_trade = False
                elif trade_type == -1: # Short
                    if highs[i] >= sl_price:
                        trade_returns.append((entry_price - sl_price) / entry_price)
                        in_trade = False
                    elif lows[i] <= tp_price:
                        trade_returns.append((entry_price - tp_price) / entry_price)
                        in_trade = False
            else:
                # --- ENTRY LOGIK ---
                if USE_FRIDAY_EXIT and is_friday_close[i]:
                    continue
                    
                if long_entries[i]:
                    in_trade = True
                    trade_type = 1
                    entry_price = opens[i]
                    sl_dist = atrs[i-1] * sl_multiplier
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + (sl_dist * rrr)
                elif short_entries[i]:
                    in_trade = True
                    trade_type = -1
                    entry_price = opens[i]
                    sl_dist = atrs[i-1] * sl_multiplier
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - (sl_dist * rrr)

        return trade_returns

    def objective(self, trial: optuna.Trial) -> float:
        target_regime = trial.suggest_int('target_regime', 0, 3)
        cat_a_name = trial.suggest_categorical('cat_a_name', ['bollinger', 'keltner', 'vwap', 'ichimoku', 'pivot'])
        cat_b_name = trial.suggest_categorical('cat_b_name', ['rsi', 'macd', 'stoch', 'mfi', 'adaptive_rsi'])
        
        # Kategorie C in den Suchraum aufnehmen
        cat_c_name = trial.suggest_categorical('cat_c_name', ['ema_cross', 'adx', 'psar']) 
        
        cat_a_length = trial.suggest_int('cat_a_length', 10, 50)
        cat_b_length = trial.suggest_int('cat_b_length', 5, 25)
        # Länge für Trendindikatoren 
        cat_c_length = trial.suggest_int('cat_c_length', 20, 100) 
        
        mom_oversold = trial.suggest_int('mom_oversold', 20, 40)
        mom_overbought = trial.suggest_int('mom_overbought', 60, 80)
        
        sl_multiplier = trial.suggest_float('sl_multiplier', 1.0, 3.5, step=0.1)
        rrr = trial.suggest_float('rrr', 1.0, 3.0, step=0.5)

        params = {
            'target_regime': target_regime,
            'cat_a_name': cat_a_name, 'cat_a_length': cat_a_length,
            'cat_b_name': cat_b_name, 'cat_b_length': cat_b_length,
            'cat_c_name': cat_c_name, 'cat_c_length': cat_c_length, 
            'mom_oversold': mom_oversold, 'mom_overbought': mom_overbought,
            'sl_multiplier': sl_multiplier, 'rrr': rrr
        }
        
        # --- CLUSTER POOLING ---
        all_cluster_returns = []
        for asset_name, df in self.data_dict.items():
            asset_returns = self._run_vectorized_backtest(df, params)
            all_cluster_returns.extend(asset_returns)
            
        min_required_trades = len(self.data_dict) * MIN_TRADES_PER_ASSET
        if len(all_cluster_returns) < min_required_trades:
            return -1.0
            
        ret_arr = np.array(all_cluster_returns)
        mean_ret = np.mean(ret_arr)
        std_ret = np.std(ret_arr)
        
        if std_ret == 0:
            return 0.0
            
        return mean_ret / std_ret

    def run_optimization(self, n_trials: int = 100):
        print(f"\nStarte Optimierung für Cluster {self.cluster_id} (mit Friday-Exit: {USE_FRIDAY_EXIT})...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("\n--- Optimierung Abgeschlossen ---")
        print(f"Beste Sharpe Ratio: {study.best_value:.4f}")
        print("Beste gefundene Regeln:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        return study.best_value, study.best_params

# --- Testlauf ---
if __name__ == "__main__":
    test_file = "mt5_h1_daten_regimes/EURUSD_H1.csv"
    if os.path.exists(test_file):
        df_eurusd = pd.read_csv(test_file, index_col='time', parse_dates=True)
        df_train = df_eurusd.iloc[:int(len(df_eurusd)*0.7)]
        
        cluster_data = {'EURUSD': df_train}
        optimizer = ClusterOptimizer(data_dict=cluster_data, cluster_id=1)
        
        optimizer.run_optimization(n_trials=10)
