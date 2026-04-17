import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from indicators import IndicatorBaukasten

class EnsembleEvaluator:
    """
    Erstellt und evaluiert ein Hybrid-Portfolio. 
    Entscheidet pro Cluster datengetrieben (basierend auf der Train_Sharpe), 
    ob ein Friday-Exit vorteilhaft ist oder nicht.
    """

    def __init__(self, experiment_name: str, file_with_exit: str, file_without_exit: str, risk_per_trade: float = 100.0, cost_per_trade: float = 3.0):
        self.experiment_name = experiment_name
        self.file_with_exit = file_with_exit
        self.file_without_exit = file_without_exit
        self.risk_per_trade = risk_per_trade
        self.cost_per_trade = cost_per_trade

        self.data_dir = "mt5_h1_daten_regimes"
        self.config_file = "bot_cluster_config.csv"
        self.plots_dir = "experiment_plots"
        self.log_file = "experiment_logbook.csv"

        self.baukasten = IndicatorBaukasten()

        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def _get_filename(self, yahoo_ticker: str, available_files: list) -> str | None:
        clean_ticker = yahoo_ticker.replace('=X', '').replace('^', '').replace('-', '')
        mapping = {
            'DJI': 'US30', 'GSPC': 'US500', 'NDX': 'NAS100', 'RUT': 'US2000',
            'N225': 'JPN225', 'FCHI': 'FRA40', 'IBEX': 'SPA35', 'HSI': 'HK50',
            'STOXX50E': 'EUSTX50', 'FTSE': 'UK100', 'GDAXI': 'GER40', 'AXJO': 'AUS200',
            'CL=F': 'XTIUSD', 'BZ=F': 'XBRUSD', 'NG=F': 'XNGUSD', 'HG=F': 'XCUUSD',
            'GC=F': 'XAUUSD', 'SI=F': 'XAGUSD', 'PL=F': 'XPTUSD', 'PA=F': 'XPDUSD',
            'ZC=F': 'Corn', 'ZW=F': 'Wheat', 'SB=F': 'Sugar', 'KC=F': 'Coffee',
            'ATOMUSD': 'ATMUSD',
            'NEARUSD': 'NERUSD'
        }
        search_name = mapping.get(clean_ticker, clean_ticker)
        for f in available_files:
            if f.startswith(search_name + "_H1"):
                return f
        return None

    def _calculate_metrics(self, trades: list) -> tuple:
        if not trades or len(trades) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        returns_arr = np.array([(self.risk_per_trade * t['r_mult']) - self.cost_per_trade for t in trades])
        total_profit = np.sum(returns_arr)

        wins = returns_arr[returns_arr > 0]
        win_rate = (len(wins) / len(trades)) * 100 if len(trades) > 0 else 0

        cumulative = np.cumsum(returns_arr)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)
        sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

        downside_returns = returns_arr[returns_arr < 0]
        downside_dev = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 1e-6
        sortino = (mean_ret / downside_dev)

        return total_profit, win_rate, max_dd, sharpe, sortino

    def _extract_trades(self, df: pd.DataFrame, params: dict, use_friday_exit: bool) -> list:
        # Kategorie A, B und C Indikatoren hinzufügen
        df_sim = self.baukasten.add_category_a_zones(
            df, indicator_name=params['cat_a_name'], length=params['cat_a_length'],
            std=params.get('cat_a_std', 2.0), scalar=params.get('cat_a_scalar', 1.5)
        )
        df_sim = self.baukasten.add_category_b_momentum(
            df_sim, indicator_name=params['cat_b_name'], length=params['cat_b_length']
        )
        df_sim = self.baukasten.add_category_c_trend(
            df_sim, indicator_name=params['cat_c_name'], length=params['cat_c_length']
        )

        df_sim.dropna(inplace=True)
        if len(df_sim) < 100:
            return []

        # Saubere Extrahierung aller Numpy-Arrays
        closes, highs, lows, opens = df_sim['close'].values, df_sim['high'].values, df_sim['low'].values, df_sim['open'].values
        zone_lower, zone_upper = df_sim['zone_lower'].values, df_sim['zone_upper'].values
        
        momentum = df_sim['momentum_main'].values
        trend = df_sim['trend_main'].values
        atrs = df_sim['atr_14'].values
        regimes = df_sim['hmm_regime'].values
        timestamps = df_sim.index

        is_friday_close = np.array((df_sim.index.dayofweek == 4) & (df_sim.index.hour >= 21))
        target_regime, sl_multiplier, rrr = params['target_regime'], params['sl_multiplier'], params['rrr']

        if params['cat_b_name'] == 'macd':
            mom_oversold_cond, mom_overbought_cond = momentum < 0, momentum > 0
        else:
            mom_oversold_cond = momentum < params.get('mom_oversold', 30)
            mom_overbought_cond = momentum > params.get('mom_overbought', 70)

        # Trend-Filter in die Signale integriert
        long_signals = (regimes == target_regime) & (closes < zone_lower) & mom_oversold_cond & (trend == 1)
        short_signals = (regimes == target_regime) & (closes > zone_upper) & mom_overbought_cond & (trend == -1)

        long_entries, short_entries = np.roll(long_signals, 1), np.roll(short_signals, 1)
        long_entries[0], short_entries[0] = False, False

        trades = []
        in_trade, trade_type = False, 0
        entry_price = sl_price = tp_price = sl_dist = 0.0

        for i in range(1, len(closes)):
            if in_trade:
                if use_friday_exit and is_friday_close[i]:
                    r_mult = (closes[i] - entry_price) / sl_dist if trade_type == 1 else (entry_price - closes[i]) / sl_dist
                    trades.append({'time': timestamps[i], 'r_mult': r_mult})
                    in_trade = False
                    continue
                if trade_type == 1:
                    if lows[i] <= sl_price:
                        trades.append({'time': timestamps[i], 'r_mult': -1.0})
                        in_trade = False
                    elif highs[i] >= tp_price:
                        trades.append({'time': timestamps[i], 'r_mult': rrr})
                        in_trade = False
                elif trade_type == -1:
                    if highs[i] >= sl_price:
                        trades.append({'time': timestamps[i], 'r_mult': -1.0})
                        in_trade = False
                    elif lows[i] <= tp_price:
                        trades.append({'time': timestamps[i], 'r_mult': rrr})
                        in_trade = False
            else:
                if use_friday_exit and is_friday_close[i]:
                    continue
                if long_entries[i]:
                    in_trade, trade_type, entry_price = True, 1, opens[i]
                    sl_dist = atrs[i-1] * sl_multiplier
                    sl_price, tp_price = entry_price - sl_dist, entry_price + (sl_dist * rrr)
                elif short_entries[i]:
                    in_trade, trade_type, entry_price = True, -1, opens[i]
                    sl_dist = atrs[i-1] * sl_multiplier
                    sl_price, tp_price = entry_price + sl_dist, entry_price - (sl_dist * rrr)

        return trades

    def build_best_ensemble(self) -> pd.DataFrame:
        """
        Vergleicht die Train_Sharpe Metriken beider CSV-Dateien und 
        erstellt ein DataFrame mit den jeweils besten Parametern pro Cluster.
        """
        df_with = pd.read_csv(self.file_with_exit)
        df_without = pd.read_csv(self.file_without_exit)
        
        best_params_list = []
        
        for cid in range(12):
            row_w = df_with[df_with['Cluster_ID'] == cid].iloc[0].copy()
            row_wo = df_without[df_without['Cluster_ID'] == cid].iloc[0].copy()
            
            # Wähle das Setup mit der höheren In-Sample Sharpe Ratio
            if row_w['Train_Sharpe'] >= row_wo['Train_Sharpe']:
                row_w['used_exit'] = True
                best_params_list.append(row_w)
            else:
                row_wo['used_exit'] = False
                best_params_list.append(row_wo)
                
        return pd.DataFrame(best_params_list)

    def run(self):
        if not os.path.exists(self.file_with_exit) or not os.path.exists(self.file_without_exit):
            print(f"Fehler: Eine der CSV-Dateien wurde nicht gefunden!")
            return

        ensemble_df = self.build_best_ensemble()
        cluster_config = pd.read_csv(self.config_file)
        available_files = os.listdir(self.data_dir)

        all_trades = []

        print(f"\nStarte REVERSE ENSEMBLE Auswertung (Hybrid-Modell): '{self.experiment_name}'")
        print("=" * 75)
        print(f"{'Cluster':<10} | {'Friday Exit':<12} | {'Train Sharpe':<15} | {'Trades'}")
        print("-" * 75)

        for index, row in ensemble_df.iterrows():
            cid = int(row['Cluster_ID'])
            use_exit = bool(row['used_exit'])
            train_sharpe = float(row['Train_Sharpe'])
            
            # Entferne Hilfsspalten für das Dictionary
            params = row.drop(['Cluster_ID', 'Train_Sharpe', 'used_exit']).to_dict()
            assets = cluster_config[cluster_config['Cluster_ID'] == cid]['Asset'].tolist()

            cluster_trades = []
            for asset in assets:
                filename = self._get_filename(asset, available_files)
                if filename:
                    filepath = os.path.join(self.data_dir, filename)
                    df = pd.read_csv(filepath, index_col='time', parse_dates=True)
                    cluster_trades.extend(self._extract_trades(df, params, use_exit))

            if cluster_trades:
                all_trades.extend(cluster_trades)
                exit_str = "AN" if use_exit else "AUS"
                print(f"Cluster {cid:02d} | {exit_str:<12} | {train_sharpe:<15.3f} | {len(cluster_trades)} aggregiert")

        if not all_trades:
            print("Keine Trades gefunden. Abbruch.")
            return

        print("=" * 75)

        # ---------------------------------------------------------
        # REVERSE SPLIT LOGIK (30% OOS Test / 70% In-Sample Train)
        # ---------------------------------------------------------
        all_trades.sort(key=lambda x: x['time'])
        first_date, last_date = all_trades[0]['time'], all_trades[-1]['time']
        
        # Split liegt bei 30% der Zeitleiste
        split_date = first_date + (last_date - first_date) * 0.3

        # OOS sind die ERSTEN 30%, Train sind die LETZTEN 70%
        oos_trades = [t for t in all_trades if t['time'] < split_date]
        train_trades = [t for t in all_trades if t['time'] >= split_date]
        
        split_idx = len(oos_trades)
        # ---------------------------------------------------------

        # Metriken berechnen
        p_train, w_train, dd_train, sh_train, so_train = self._calculate_metrics(train_trades)
        p_oos, w_oos, dd_oos, sh_oos, so_oos = self._calculate_metrics(oos_trades)
        p_total, w_total, dd_total, sh_total, so_total = self._calculate_metrics(all_trades)

        # Equity Curve
        start_capital = 10000.0
        capital = start_capital
        equity_curve = [capital]

        for trade in all_trades:
            capital += (self.risk_per_trade * trade['r_mult']) - self.cost_per_trade
            equity_curve.append(capital)

        # Plot erstellen und speichern
        self._create_and_save_plot(equity_curve, split_idx, p_train, w_train, dd_train, sh_train, so_train,
                                   p_oos, w_oos, dd_oos, sh_oos, so_oos, p_total, w_total, dd_total, sh_total, so_total)

        # Im Logbuch dokumentieren
        self._log_results(p_total, p_train, p_oos, sh_total, sh_train, sh_oos,
                          so_total, so_train, so_oos, dd_total, dd_oos, w_total, len(all_trades))

        # MONTE CARLO EXPORT (NUR OUT-OF-SAMPLE) ---
        # Speichert nur die R-Multiples der OOS-Phase 
        export_df = pd.DataFrame([t['r_mult'] for t in oos_trades], columns=['R_Mult'])
        export_df.to_csv(f"{self.experiment_name}_OOS_MC_Trades.csv", index=False)
        print(f"=> OOS-Trades für Monte Carlo exportiert als '{self.experiment_name}_OOS_MC_Trades.csv'")

    def _create_and_save_plot(self, equity_curve, split_idx, p_train, w_train, dd_train, sh_train, so_train,
                              p_oos, w_oos, dd_oos, sh_oos, so_oos, p_total, w_total, dd_total, sh_total, so_total):
        plt.figure(figsize=(14, 9))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f} €".replace(',', '.')))

        # ---------------------------------------------------------
        # FARBEN UND LABELS FÜR REVERSE-PLOT
        # ---------------------------------------------------------
        # Der linke Teil (bis split_idx) ist OOS (Grün)
        plt.plot(range(split_idx), equity_curve[:split_idx], color='green', linewidth=1.5, label='Out-of-Sample (Test, Erste 30%)')
        
        if split_idx > 0 and split_idx < len(equity_curve):
            # Der rechte Teil (ab split_idx) ist In-Sample/Train (Blau)
            plt.plot(range(split_idx-1, len(equity_curve)), equity_curve[split_idx-1:], color='blue', linewidth=1.5, label='In-Sample (Train, Letzte 70%)')
            plt.axvline(x=split_idx, color='red', linestyle='--', linewidth=2, label='Start Train (In-Sample)')
        # ---------------------------------------------------------

        plt.title(f"Experiment: {self.experiment_name} | {int(self.risk_per_trade)}€ Risk | {int(self.cost_per_trade)}€ Fee", fontsize=14, fontweight='bold')
        plt.xlabel('Anzahl der Trades', fontsize=12)
        plt.ylabel('Kontostand (€)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')

        stats_text = (
            f"--- IN-SAMPLE (Train) ---\n"
            f"Profit: {p_train:+,.0f} €\n"
            f"Win Rate: {w_train:.1f} %\n"
            f"Max DD: {dd_train:,.0f} €\n"
            f"Sharpe/Sortino: {sh_train:.2f} / {so_train:.2f}\n\n"
            f"--- OUT-OF-SAMPLE (Test) ---\n"
            f"Profit: {p_oos:+,.0f} €\n"
            f"Win Rate: {w_oos:.1f} %\n"
            f"Max DD: {dd_oos:,.0f} €\n"
            f"Sharpe/Sortino: {sh_oos:.2f} / {so_oos:.2f}\n\n"
            f"--- OVERALL (Gesamtzeitraum) ---\n"
            f"Profit: {p_total:+,.0f} €\n"
            f"Win Rate: {w_total:.1f} %\n"
            f"Max DD: {dd_total:,.0f} €\n"
            f"Sharpe/Sortino: {sh_total:.2f} / {so_total:.2f}\n"
        )
        plt.figtext(0.13, 0.45, stats_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"{self.experiment_name}_Equity.png"), dpi=300)
        plt.close()

    def _log_results(self, p_total, p_train, p_oos, sh_total, sh_train, sh_oos,
                     so_total, so_train, so_oos, dd_total, dd_oos, w_total, total_trades):
        log_data = {
            'Experiment_Name': [self.experiment_name],
            'Total_Trades': [total_trades],
            'Total_Profit_EUR': [round(p_total, 2)],
            'Train_Profit_EUR': [round(p_train, 2)],
            'OOS_Profit_EUR': [round(p_oos, 2)],
            'Total_Sharpe': [round(sh_total, 3)],
            'Train_Sharpe': [round(sh_train, 3)],
            'OOS_Sharpe': [round(sh_oos, 3)],
            'Total_Sortino': [round(so_total, 3)],
            'Train_Sortino': [round(so_train, 3)],
            'OOS_Sortino': [round(so_oos, 3)],
            'Total_Max_DD_EUR': [round(dd_total, 2)],
            'OOS_Max_DD_EUR': [round(dd_oos, 2)],
            'Total_Win_Rate_Pct': [round(w_total, 2)]
        }

        df_log = pd.DataFrame(log_data)

        if os.path.exists(self.log_file):
            df_log.to_csv(self.log_file, mode='a', header=False, index=False)
        else:
            df_log.to_csv(self.log_file, index=False)

        print(f"\n=> Auswertung erfolgreich!")
        print(f"=> OVERALL Sharpe: {sh_total:.3f} | OVERALL Profit: {p_total:+,.0f} €")
        print(f"=> OOS Sharpe: {sh_oos:.3f} | OOS Profit: {p_oos:+,.0f} €")
        print(f"=> Ergebnisse angehängt in '{self.log_file}'.")


# =================================================================================
# AUSFÜHRUNG: HIER DEN NAMEN UND DIE DATEIEN FÜR EIN NEUES EXPERIMENT ANPASSEN!
# =================================================================================
if __name__ == "__main__":
    
    # Der Name dieses Experiments im Logbuch
    EXPERIMENT = "Reverse_OOS_Ensemble_Hybrid_Kat_C_500_Trials"
    
    # Trage hier die beiden CSV-Dateien (mit Kategorie C) ein, die das Skript vergleichen soll
    DATEI_MIT_EXIT = "params_WITH_friday_exit_kat_c_reverse_(500).csv"
    DATEI_OHNE_EXIT = "params_WITHOUT_friday_exit_kat_c_reverse_(500).csv"

    evaluator = EnsembleEvaluator(
        experiment_name=EXPERIMENT,
        file_with_exit=DATEI_MIT_EXIT,
        file_without_exit=DATEI_OHNE_EXIT,
        risk_per_trade=100.0,  
        cost_per_trade=3.0     
    )
    
    evaluator.run()
