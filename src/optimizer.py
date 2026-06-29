"""
Optimierungs-Engine: Vektor-Backtest, mehrschichtige Fitnessfunktion mit
Nachbarschaftstest und TPE-Optimierung fuer ein isoliertes Walk-Forward-Fenster.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import optuna

from config import CONFIG
from indicators import IndicatorEngine


class FoldOptimizer:
    """
    Optimierungs-Engine zur Durchfuehrung der Vektor-Backtests und der Parameterfindung
    fuer ein isoliertes Walk-Forward-Fenster (Fold) mittels Optuna (TPE).
    """
    def __init__(self, data_dict: dict, val_start=None):
        self.data_dict = data_dict
        self.val_start = val_start

    def _run_vectorized_backtest(self, df: pd.DataFrame, params: dict, valid_start=None, asset_name: str = "") -> list:
        """ Fuehrt die Vektor-Simulation der Preisgeometrie, Momentum-Hooks und Orderausfuehrung durch. """
        # Waehrungsanpassung fuer korrekte Notional-Werte (JPY Fix)
        is_jpy = "JPY" in asset_name.upper()
        quote_mult = 100.0 if is_jpy else 1.0

        df_sim = IndicatorEngine.add_swing_zones(df, length=params['cat_a_length'])
        df_sim = IndicatorEngine.add_rsi_momentum(df_sim, length=params['cat_b_length'])

        # Vola-Explosionsschutz (Crash Protector)
        df_sim['atr_10'] = ta.atr(df_sim['high'], df_sim['low'], df_sim['close'], length=10)
        df_sim['atr_100'] = ta.atr(df_sim['high'], df_sim['low'], df_sim['close'], length=100)
        df_sim['vola_ratio'] = df_sim['atr_10'] / df_sim['atr_100'].replace(0, 1e-6)
        df_sim['atr_pct'] = (df_sim['atr_14'] / df_sim['close']) * 100

        df_sim.dropna(inplace=True)
        if len(df_sim) < 100: return []

        closes, highs, lows, opens = df_sim['close'].values, df_sim['high'].values, df_sim['low'].values, df_sim['open'].values
        zone_lower, zone_upper = df_sim['zone_lower'].values, df_sim['zone_upper'].values
        momentum = df_sim['momentum_main'].values
        vola_ratio = df_sim['vola_ratio'].values
        atr_pct = df_sim['atr_pct'].values
        atrs, regimes = df_sim['atr_14'].values, df_sim['hmm_regime'].values
        times = df_sim.index

        crash_protect = (vola_ratio < 1.5) & (atr_pct < 0.4)
        t_regime = params.get('target_regime', 0)
        sl_mult = params['sl_multiplier']
        rrr = params['rrr']

        # Dynamische Zonentoleranz fuer Entries
        zone_tol = params.get('zone_tol', 0.3)
        eff_zone_upper = zone_upper - (atrs * zone_tol)
        eff_zone_lower = zone_lower + (atrs * zone_tol)

        # 3-Kerzen Momentum Gedaechtnis (Hook)
        oversold_level = params.get('mom_oversold', 30)
        overbought_level = params.get('mom_overbought', 70)
        mom_buy_raw = ((momentum > oversold_level) & (np.roll(momentum, 1) <= oversold_level))
        mom_sell_raw = ((momentum < overbought_level) & (np.roll(momentum, 1) >= overbought_level))
        mom_buy_memory = pd.Series(mom_buy_raw).rolling(3).max().fillna(0).values == 1
        mom_sell_memory = pd.Series(mom_sell_raw).rolling(3).max().fillna(0).values == 1

        # 2-Kerzen Wick Rejection (Abpraller)
        candle_range = np.where((highs - lows) == 0, 1e-6, highs - lows)
        bullish_rejection_raw = ((closes - lows) / candle_range > 0.5) & (lows <= eff_zone_lower)
        bearish_rejection_raw = ((highs - closes) / candle_range > 0.5) & (highs >= eff_zone_upper)
        bullish_rejection_memory = pd.Series(bullish_rejection_raw).rolling(2).max().fillna(0).values == 1
        bearish_rejection_memory = pd.Series(bearish_rejection_raw).rolling(2).max().fillna(0).values == 1

        # Finale Synthese des S/R Signals
        long_sig = (regimes == t_regime) & mom_buy_memory & bullish_rejection_memory & crash_protect
        short_sig = (regimes == t_regime) & mom_sell_memory & bearish_rejection_memory & crash_protect

        long_ent, short_ent = np.roll(long_sig, 1), np.roll(short_sig, 1)
        long_ent[0], short_ent[0] = False, False

        trade_returns = []
        in_trade, trade_type, entry_price, sl_price, tp_price, sl_dist = False, 0, 0.0, 0.0, 0.0, 0.0
        current_entry_time = None
        actual_risk, actual_fee, current_notional = 0.0, 0.0, 0.0
        cooldown_until_index = 0

        for i in range(1, len(closes)):
            is_entry_candle = False

            if not in_trade:
                if valid_start is not None and times[i] < valid_start: continue
                if i < cooldown_until_index: continue

                if long_ent[i] or short_ent[i]:
                    candidate_type = 1 if long_ent[i] else -1
                    candidate_entry = opens[i]

                    min_sl_dist = candidate_entry * 0.001
                    temp_sl_dist = max(atrs[i-1] * sl_mult, min_sl_dist)

                    # Proximity Filter: Unterdrueckt spaete Entries
                    max_allowed_distance = temp_sl_dist * 0.6
                    if candidate_type == 1 and (candidate_entry - zone_lower[i-1]) > max_allowed_distance: continue
                    if candidate_type == -1 and (zone_upper[i-1] - candidate_entry) > max_allowed_distance: continue

                    trade_type = candidate_type
                    in_trade, entry_price = True, candidate_entry
                    current_entry_time = times[i]
                    sl_dist = temp_sl_dist

                    ideal_pos_size = (CONFIG["RISK_PER_TRADE"] * quote_mult) / sl_dist
                    max_pos_size = (CONFIG["START_CAPITAL"] * CONFIG["MAX_LEVERAGE"]) / (entry_price / quote_mult)
                    actual_pos_size = min(ideal_pos_size, max_pos_size)

                    actual_risk = (actual_pos_size * sl_dist) / quote_mult
                    current_notional = (actual_pos_size * entry_price) / quote_mult
                    actual_fee = current_notional * CONFIG["FEE_RATE"] * 2

                    # Harter Gebuehren-Filter
                    if actual_fee > (CONFIG["RISK_PER_TRADE"] * 0.10):
                        in_trade = False
                        continue

                    if trade_type == 1:
                        sl_price, tp_price = entry_price - sl_dist, entry_price + (sl_dist * rrr)
                    else:
                        sl_price, tp_price = entry_price + sl_dist, entry_price - (sl_dist * rrr)

                    current_meta = (trade_type, entry_price, sl_price, tp_price, zone_lower[i], zone_upper[i], momentum[i], atrs[i], asset_name)
                    is_entry_candle = True

            if in_trade:
                # Order Execution Logic (SL/TP)
                if trade_type == 1:
                    if not is_entry_candle and opens[i] <= sl_price:
                        gap_r = (opens[i] - entry_price) / sl_dist
                        trade_returns.append((current_entry_time, times[i], gap_r - CONFIG["SL_SLIPPAGE"], actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False
                    elif not is_entry_candle and opens[i] >= tp_price:
                        trade_returns.append((current_entry_time, times[i], rrr, actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False
                    elif lows[i] <= sl_price:
                        trade_returns.append((current_entry_time, times[i], -(1.0 + CONFIG["SL_SLIPPAGE"]), actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False
                        cooldown_until_index = i + 4
                    elif highs[i] >= (tp_price + (entry_price * CONFIG["SPREAD_TRIGGER"])):
                        trade_returns.append((current_entry_time, times[i], rrr, actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False

                elif trade_type == -1:
                    if not is_entry_candle and opens[i] >= sl_price:
                        gap_r = (entry_price - opens[i]) / sl_dist
                        trade_returns.append((current_entry_time, times[i], gap_r - CONFIG["SL_SLIPPAGE"], actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False
                    elif not is_entry_candle and opens[i] <= tp_price:
                        trade_returns.append((current_entry_time, times[i], rrr, actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False
                    elif highs[i] >= sl_price:
                        trade_returns.append((current_entry_time, times[i], -(1.0 + CONFIG["SL_SLIPPAGE"]), actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False
                        cooldown_until_index = i + 4
                    elif lows[i] <= (tp_price - (entry_price * CONFIG["SPREAD_TRIGGER"])):
                        trade_returns.append((current_entry_time, times[i], rrr, actual_risk, actual_fee, current_notional) + current_meta)
                        in_trade = False

        if in_trade:
            s_dist = sl_dist if sl_dist > 1e-6 else 1e-6
            r_val = (closes[-1] - entry_price) / s_dist if trade_type == 1 else (entry_price - closes[-1]) / s_dist
            trade_returns.append((current_entry_time, times[-1], r_val, actual_risk, actual_fee, current_notional) + current_meta)

        return trade_returns

    def objective(self, trial: optuna.Trial) -> float:
        """ Bewertet die statistische Robustheit einer Parameterkombination fuer Optuna. """
        cat_a_length = trial.suggest_int('swing_len', *CONFIG["OPT_SWING_LEN"], step=4)
        zone_tol = trial.suggest_float('zone_tol_swing', *CONFIG["OPT_ZONE_TOL_SWING"], step=0.1)
        cat_b_length = trial.suggest_int('rsi_len', *CONFIG["OPT_RSI_LEN"], step=2)
        mom_oversold = trial.suggest_int('rsi_oversold', *CONFIG["OPT_RSI_OVERSOLD"])
        mom_overbought = trial.suggest_int('rsi_overbought', *CONFIG["OPT_RSI_OVERBOUGHT"])
        cat_c_length = trial.suggest_int('hmm_len', 45, 60, step=2)
        t_regime = trial.suggest_int('target_regime', 0, 1)
        sl_multiplier = trial.suggest_float('sl_multiplier', *CONFIG["OPT_SL_MULT_SR"], step=0.1)
        rrr = trial.suggest_float('rrr', *CONFIG["OPT_RRR_SR"], step=0.1)

        base_params = {
            'cat_a_length': cat_a_length, 'zone_tol': zone_tol,
            'cat_b_length': cat_b_length, 'mom_oversold': mom_oversold, 'mom_overbought': mom_overbought,
            'cat_c_length': cat_c_length, 'target_regime': t_regime,
            'sl_multiplier': sl_multiplier, 'rrr': rrr
        }

        def evaluate_parameter_set(test_params, trial_to_update=None):
            # Lazy import bricht den Zirkelbezug Optimizer <-> Engine auf
            # (Modul ist zur Laufzeit bereits geladen; kein Re-Import-Overhead).
            from engine import WalkForwardEngine

            all_rets = []
            active_assets_in_val = 0

            for asset, df in self.data_dict.items():
                asset_trades = self._run_vectorized_backtest(df, test_params, asset_name=asset)
                all_rets.extend(asset_trades)

                if self.val_start is not None:
                    val_trades_for_asset = [t for t in asset_trades if t[1] >= self.val_start]
                    if len(val_trades_for_asset) >= CONFIG.get("MIN_TRADES_PER_ASSET", 2):
                        active_assets_in_val += 1

            if self.val_start is None: return -1.0

            min_required_assets = max(1, int(len(self.data_dict) * CONFIG.get("MIN_ACTIVE_ASSETS_PERCENT", 0.3)))
            if active_assets_in_val < min_required_assets: return -1.0

            filtered_trades = WalkForwardEngine.apply_portfolio_limit(all_rets, CONFIG["MAX_OPEN_TRADES"])
            train_trades = [t for t in filtered_trades if t[1] < self.val_start]
            val_trades = [t for t in filtered_trades if t[1] >= self.val_start]

            if len(train_trades) < 20 or len(val_trades) < 5: return -1.0

            def get_sharpe(trades_list, cap, is_val=False):
                if len(trades_list) < 5: return 0.0
                r_arr = np.array([r[2] for r in trades_list])
                pnl_arr = (r_arr * np.array([r[3] for r in trades_list])) - np.array([r[4] for r in trades_list])
                mean_pnl, std_pnl = np.mean(pnl_arr), np.std(pnl_arr)

                safe_std = max(std_pnl, CONFIG["RISK_PER_TRADE"] * 0.20)
                raw_sharpe = (mean_pnl / safe_std) * np.sqrt(min(len(pnl_arr), cap))

                trade_penalty = len(trades_list) / 10.0 if len(trades_list) < 10 else 1.0
                final_sharpe = raw_sharpe * trade_penalty

                if final_sharpe > 1.2:
                    return 1.2 if is_val else -0.1
                return final_sharpe

            train_sharpe = get_sharpe(train_trades, cap=200, is_val=False)
            val_sharpe = get_sharpe(val_trades, cap=75, is_val=True)

            val_pnls = np.array([(t[2] * t[3]) - t[4] for t in val_trades])
            if np.sum(val_pnls) <= 0: return -1.0

            n_trades = len(val_pnls)
            equity_curve = np.cumsum(val_pnls) + CONFIG["START_CAPITAL"]

            # 1. Ulcer Index
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = (peaks - equity_curve) / peaks * 100
            ui = np.sqrt(np.mean(drawdowns**2))
            ui_multiplier = max(0.5, np.exp(-max(0, ui - 1.5) / 20.0))

            # 2. K-Ratio
            x = np.arange(1, n_trades + 1)
            slope, intercept = np.polyfit(x, equity_curve, 1)
            residuals = equity_curve - (slope * x + intercept)
            std_error = np.std(residuals) if np.std(residuals) > 0 else 1e-6
            k_ratio = (slope * n_trades) / (std_error * np.sqrt(n_trades))
            k_multiplier = max(0.5, min(1.2, k_ratio))

            # 3. Gini-Koeffizient
            winning_trades = np.sort(val_pnls[val_pnls > 0])
            if len(winning_trades) > 2:
                n_wins = len(winning_trades)
                idx = np.arange(1, n_wins + 1)
                gini = (2 * np.sum(idx * winning_trades) / (n_wins * np.sum(winning_trades))) - (n_wins + 1) / n_wins
                gini_multiplier = max(0.5, 1.0 - (gini * 0.8))
            else:
                gini_multiplier = 0.5

            # 4. Skewness
            mean_val_pnl = np.mean(val_pnls)
            std_val_pnl = np.std(val_pnls) if np.std(val_pnls) > 0 else 1e-6
            skewness = np.mean(((val_pnls - mean_val_pnl) / std_val_pnl)**3)
            skew_multiplier = 1.0 if skewness > 0 else max(0.5, np.exp(skewness / 4.0))

            # 5. Asset-Breadth
            profitable_assets = sum(1 for a in set(t[-1] for t in val_trades) if sum(
                (tr[2] * tr[3]) - tr[4] for tr in val_trades if tr[-1] == a) > 0)
            traded_assets = len(set(t[-1] for t in val_trades))
            asset_win_rate = (profitable_assets / traded_assets) if traded_assets > 0 else 0.0
            asset_multiplier = 1.0 if asset_win_rate >= 0.5 else 0.7

            # Finale Synthese
            raw_base = min(train_sharpe, val_sharpe)
            if raw_base <= 0.0: return float(raw_base)

            _all_mult = [ui_multiplier, k_multiplier, gini_multiplier, skew_multiplier, asset_multiplier]
            robust_score = raw_base * float(np.prod(_all_mult) ** (1.0 / len(_all_mult)))

            if n_trades < 15: robust_score *= max(0.5, (n_trades / 15.0))

            if trial_to_update is not None:
                trial_to_update.set_user_attr("train_sharpe", train_sharpe)
                trial_to_update.set_user_attr("val_sharpe", val_sharpe)
                trial_to_update.set_user_attr("asset_win_rate", asset_win_rate)
                trial_to_update.set_user_attr("ulcer_index", float(ui))
                trial_to_update.set_user_attr("k_ratio", float(k_ratio))
                trial_to_update.set_user_attr("gini_coeff", float(gini) if len(winning_trades) > 2 else 1.0)

            return float(robust_score)

        # Neighborhood Test zur Validierung von Parametertaelern
        score_base = evaluate_parameter_set(base_params, trial_to_update=trial)
        if score_base < 0.1: return float(score_base)

        fast_params = base_params.copy()
        fast_params['cat_a_length'] = max(10, int(base_params['cat_a_length'] * 0.90))
        fast_params['cat_b_length'] = max(5,  int(base_params['cat_b_length'] * 0.90))
        score_fast = evaluate_parameter_set(fast_params)

        slow_params = base_params.copy()
        slow_params['cat_a_length'] = int(base_params['cat_a_length'] * 1.10)
        slow_params['cat_b_length'] = int(base_params['cat_b_length'] * 1.10)
        score_slow = evaluate_parameter_set(slow_params)

        return float(min(score_base, score_fast, score_slow))

    def run_optimization(self, n_trials: int, seed_val: int = 42):
        """ Startet den Bayes'schen TPE-Optimierer. """
        sampler = optuna.samplers.TPESampler(seed=seed_val)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials, n_jobs=1)
        return study
