"""
Orchestrierung der sequenziellen Walk-Forward-Analyse.

Steuert Datafeed, Multi-Seed-Ensembles, das State-Machine-Risikomanagement und
den globalen Circuit Breaker; stoesst am Ende die Report-Erzeugung an.
"""

import os
import glob
import random

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config import CONFIG
from regime import MarketRegimeDetector
from optimizer import FoldOptimizer
from reporting import ReportGenerator


class WalkForwardEngine:
    """
    Master-Klasse zur Orchestrierung der sequenziellen Walk-Forward-Analyse.
    Ueberwacht den Datafeed, steuert die Multi-Seed Ensembles und wendet das
    State-Machine Risikomanagement sowie den globalen Circuit Breaker an.
    """

    @staticmethod
    def apply_portfolio_limit(trades_list, max_concurrent):
        """ Sichert die Margin-Limitierungen ueber das gesamte Portfolio. """
        if not trades_list: return []
        trades_with_tiebreaker = [(t[0], random.random(), t) for t in trades_list]
        trades_with_tiebreaker.sort(key=lambda x: (x[0], x[1]))
        trades_sorted = [x[2] for x in trades_with_tiebreaker]

        accepted_trades, active_trades = [], []
        for trade in trades_sorted:
            entry_t, exit_t, notional_value = trade[0], trade[1], trade[5]
            active_trades = [t for t in active_trades if t[0] > entry_t]
            current_margin_used = sum([t[1] for t in active_trades]) / CONFIG["MAX_LEVERAGE"]
            required_margin = notional_value / CONFIG["MAX_LEVERAGE"]

            if len(active_trades) < max_concurrent and (current_margin_used + required_margin) <= CONFIG["START_CAPITAL"]:
                accepted_trades.append(trade)
                active_trades.append((exit_t, notional_value))
        return accepted_trades

    @staticmethod
    def calc_kpis(filtered_trades: list) -> dict:
        """ Berechnet fundamentale Portfolio-Kennzahlen. """
        if not filtered_trades:
            return {'profit': 0.0, 'win_rate': 0.0, 'max_dd': 0.0, 'sharpe': 0.0, 'equity': [CONFIG["START_CAPITAL"]]}

        returns_usd = np.array([(r[2] * r[3]) - r[4] for r in filtered_trades])
        profit = float(np.sum(returns_usd))
        win_rate = (len(returns_usd[returns_usd > 0]) / len(returns_usd)) * 100 if len(returns_usd) > 0 else 0.0

        capital, peak, max_dd = CONFIG["START_CAPITAL"], CONFIG["START_CAPITAL"], 0.0
        equity = [capital]
        for pnl in returns_usd:
            capital += pnl
            equity.append(capital)
            if capital > peak: peak = capital
            if (peak - capital) > max_dd: max_dd = peak - capital

        std_pnl = np.std(returns_usd)
        safe_std = max(std_pnl, CONFIG["RISK_PER_TRADE"] * 0.20)
        raw_sharpe = (np.mean(returns_usd) / safe_std) if len(returns_usd) > 0 else 0.0

        trade_penalty = len(filtered_trades) / 10.0 if len(filtered_trades) < 10 else 1.0
        return {'profit': profit, 'win_rate': win_rate, 'max_dd': max_dd, 'sharpe': raw_sharpe * trade_penalty, 'equity': equity}

    @staticmethod
    def load_all_data(data_dir: str) -> dict:
        files = glob.glob(f"{data_dir}/*.csv")
        data_dict = {}
        banned = set(CONFIG.get("BANNED_ASSETS", []))
        d_start = pd.to_datetime(CONFIG["DATA_START"]) if CONFIG.get("DATA_START") else None
        d_end   = pd.to_datetime(CONFIG["DATA_END"])   if CONFIG.get("DATA_END")   else None
        for path in files:
            asset_name = os.path.basename(path).replace(".csv", "").replace("_H1", "")
            if asset_name in banned:
                continue
            df = pd.read_csv(path, index_col="time", parse_dates=True)
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if d_start is not None:
                df = df[df.index >= d_start]
            if d_end is not None:
                df = df[df.index <= d_end]
            if not df.empty:
                data_dict[asset_name] = df
        return data_dict

    @staticmethod
    def run_strict_wfa():
        all_data = WalkForwardEngine.load_all_data(CONFIG["RAW_DATA_DIR"])
        if not all_data: return

        global_start, global_end = list(all_data.values())[0].index.min(), list(all_data.values())[0].index.max()

        print("="*85)
        print(f"STARTE DYNAMISCHE EVENT-DRIVEN WFA: {CONFIG['EXPERIMENT_NAME']}")
        print("="*85)
        if CONFIG.get("BANNED_ASSETS"):
            print(f"Gebannte Assets (ignoriert): {CONFIG['BANNED_ASSETS']}")

        all_oos_trades, fold_logs, all_asset_fold_records = [], [], []
        curr_start = global_start
        fold_idx = 0
        shift_events = []
        universe_size = len(all_data)
        data_span = (global_start, global_end)

        while True:
            seed_death_times = []

            t_start = curr_start
            t_end = curr_start + relativedelta(months=CONFIG["TRAIN_MONTHS"])
            test_start = t_end
            test_end = test_start + relativedelta(months=CONFIG["TEST_MONTHS"])

            if test_start >= global_end: break
            test_end_actual = min(test_end, global_end)

            print(f"\n--- FOLD {fold_idx + 1} [{t_start.date()} -> {test_end_actual.date()}] ---")

            warmup_start = test_start - relativedelta(days=70)
            inner_val_months = CONFIG.get("INNER_VAL_MONTHS", 6)
            inner_split_date = t_start + relativedelta(months=CONFIG["TRAIN_MONTHS"] - inner_val_months)

            dict_train_raw, dict_test_raw, dict_inner_train_raw = {}, {}, {}
            valid_assets = []

            for asset, df in all_data.items():
                df_train = df[(df.index >= t_start) & (df.index < t_end)].copy()
                if not df_train.empty and len(df_train) > 500:
                    dict_train_raw[asset] = df_train
                    dict_test_raw[asset] = df[(df.index >= warmup_start) & (df.index <= test_end)].copy()
                    dict_inner_train_raw[asset] = df_train[df_train.index < inner_split_date].copy()
                    valid_assets.append(asset)

            dict_train_hmm, dict_test_hmm = {}, {}
            print(f"  -> Trainiere Aligned HMM blind auf ersten {CONFIG['TRAIN_MONTHS'] - inner_val_months} Monaten...")
            detector = MarketRegimeDetector()
            for asset in dict_inner_train_raw.keys():
                detector.fit(dict_inner_train_raw[asset])
                dict_train_hmm[asset] = detector.predict_iterative(dict_train_raw[asset], step=24)

                if not dict_test_raw[asset].empty:
                    df_continuous = all_data[asset][(all_data[asset].index >= t_start) & (all_data[asset].index <= test_end)].copy()
                    hmm_result = detector.predict_iterative(df_continuous, step=24)
                    dict_test_hmm[asset] = hmm_result.loc[dict_test_raw[asset].index]

            print(f"  >>> Optimiere Forex-Universum ({len(valid_assets)} Assets)...")
            opt_engine = FoldOptimizer(dict_train_hmm, val_start=inner_split_date)

            num_ensembles = CONFIG["NUM_ENSEMBLES"]
            top_diverse_trials = []
            fold_accepted_scores = []

            print(f"    🌟 STARTE MULTI-SEED SUCHE ({num_ensembles} unabhängige Studien à {CONFIG['TRIALS_PER_FOLD']} Trials)")
            for i in range(num_ensembles):
                current_seed = CONFIG["BASE_SEED"] + i
                random.seed(current_seed)
                np.random.seed(current_seed)

                study = opt_engine.run_optimization(n_trials=CONFIG["TRIALS_PER_FOLD"], seed_val=current_seed)
                best_trial = study.best_trial

                if best_trial.value >= 0.07:
                    top_diverse_trials.append(best_trial)
                    fold_accepted_scores.append(best_trial.value)
                    print(f"      -> Seed {current_seed}: Erfolgreich! (Score: {best_trial.value:.2f})")
                else:
                    print(f"      -> Seed {current_seed}: Verworfen")

            if not top_diverse_trials:
                print(f"    [!] KEINE ROBUSTEN MODELLE IN ALLEN SEEDS GEFUNDEN. Fold bleibt in Cash!")
                fold_logs.append({"Fold": fold_idx + 1, "Net_Profit_USD": 0.0,
                                  "Circuit_Breakers_Hit": 0})
                shift_events.append({"fold": fold_idx + 1, "cut_short": False,
                                     "reason": None, "cash_fold": True})
                curr_start += relativedelta(months=CONFIG["TEST_MONTHS"])
                fold_idx += 1
                continue

            print(f"\n    🌟 OOS-TEST: Starte Ensemble mit {len(top_diverse_trials)} Modellen!")
            fold_stats = {'total_live_profit': 0.0, 'total_live_trades': 0, 'total_probe_trades': 0, 'total_paper_trades': 0, 'cb_hits': 0, 'best_seed_profit': -999999.0}
            raw_fold_trades = []

            for rank, best_trial in enumerate(top_diverse_trials):
                raw_p = best_trial.params
                _train_sharpe_check = best_trial.user_attrs.get('train_sharpe', -99.0)
                if _train_sharpe_check < CONFIG.get("MIN_TRAIN_SHARPE", 0.0):
                    continue

                best_params = {
                    'cat_a_length': raw_p.get('swing_len', 0),
                    'zone_tol': raw_p.get('zone_tol_swing', 0.3),
                    'cat_b_length': raw_p.get('rsi_len', 0),
                    'mom_oversold': raw_p.get('rsi_oversold', 30),
                    'mom_overbought': raw_p.get('rsi_overbought', 70),
                    'cat_c_length': raw_p.get('hmm_len', 50),
                    'target_regime': raw_p.get('target_regime', 0),
                    'sl_multiplier': raw_p['sl_multiplier'],
                    'rrr': raw_p['rrr']
                }

                model_oos_trades = []
                for a in valid_assets:
                    if a in dict_test_hmm and not dict_test_hmm[a].empty:
                        raw_asset_trades = opt_engine._run_vectorized_backtest(dict_test_hmm[a], best_params, valid_start=test_start, asset_name=a)
                        model_oos_trades.extend(raw_asset_trades)

                model_oos_trades.sort(key=lambda x: x[0])

                # DYNAMISCHES RISIKO-MANAGEMENT (State Machine)
                live_model_trades = []
                state = "PROBE"
                buffer = 0
                real_cum_pnl = peak_pnl = 0.0
                max_trailing_dd = -3.5 * CONFIG["RISK_PER_TRADE"]
                seed_stats = {'live': 0, 'probe': 0, 'paper': 0}
                cb_hit = False

                for t in model_oos_trades:
                    if state == "DEAD" or cb_hit: break

                    r_mult, base_risk, fee = t[2], t[3], t[4]
                    is_win = r_mult > 0

                    if state == "PROBE":
                        base_risk_factor = 0.25
                        seed_stats['probe'] += 1
                    elif state == "LIVE":
                        base_risk_factor = 1.0
                        seed_stats['live'] += 1
                    else:
                        base_risk_factor = 0.0
                        seed_stats['paper'] += 1

                    mm_multiplier = 1.0
                    if state == "LIVE":
                        if real_cum_pnl >= (CONFIG["RISK_PER_TRADE"] * 2.0):
                            mm_multiplier = 1.5
                        elif real_cum_pnl < 0.0:
                            mm_multiplier = 0.5

                    final_risk_factor = base_risk_factor * mm_multiplier
                    actual_risk_used = base_risk * final_risk_factor
                    actual_fee = fee * (final_risk_factor if final_risk_factor > 0 else 0)
                    real_trade_pnl = (r_mult * actual_risk_used) - actual_fee

                    if final_risk_factor > 0:
                        modified_trade = (t[0], t[1], r_mult, actual_risk_used, actual_fee, t[5]*final_risk_factor) + t[6:]
                        live_model_trades.append(modified_trade)

                        real_cum_pnl += real_trade_pnl
                        if real_cum_pnl > peak_pnl: peak_pnl = real_cum_pnl
                        if (real_cum_pnl - peak_pnl) <= max_trailing_dd:
                            state = "DEAD"
                            cb_hit = True
                            fold_stats['cb_hits'] += 1
                            seed_death_times.append(t[1])
                            continue

                    # State Machine Logik
                    if state == "PROBE":
                        if is_win: state, buffer = "LIVE", 1
                        else: state = "PAPER"
                    elif state == "LIVE":
                        if is_win: buffer = min(buffer + 1, 2)
                        else:
                            buffer -= 1
                            if buffer < 0: state, buffer = "PAPER", 0
                    elif state == "PAPER":
                        if is_win: state = "PROBE"

                filtered_model_trades = WalkForwardEngine.apply_portfolio_limit(live_model_trades, CONFIG["MAX_OPEN_TRADES"])
                raw_fold_trades.extend(filtered_model_trades)

                m_kpi = WalkForwardEngine.calc_kpis(filtered_model_trades)
                status = "🟢" if m_kpi['profit'] > 0 else "🔴" if m_kpi['profit'] < 0 else "⚪"
                cb_flag = " 💀 [CB HIT]" if cb_hit else ""

                fold_stats['total_live_profit'] += m_kpi['profit']
                fold_stats['total_live_trades'] += seed_stats['live']
                fold_stats['total_probe_trades'] += seed_stats['probe']
                fold_stats['total_paper_trades'] += seed_stats['paper']
                if m_kpi['profit'] > fold_stats['best_seed_profit']: fold_stats['best_seed_profit'] = m_kpi['profit']

                print(f"      -> {status} Seed {CONFIG['BASE_SEED'] + rank}: {m_kpi['profit']:+,.2f} $ | Trades (Live/Probe/Paper): {seed_stats['live']}/{seed_stats['probe']}/{seed_stats['paper']} {cb_flag}")

            # DYNAMISCHE EVENT-DRIVEN FOLD-NOTBREMSE (CIRCUIT BREAKER)
            raw_fold_trades.sort(key=lambda x: x[1])
            filtered_fold_trades = []
            global_fold_pnl = 0.0
            fold_cb_hit = False
            premature_end_time = None
            stop_reason = None

            fold_median_score = float(np.median(fold_accepted_scores)) if fold_accepted_scores else 0.0
            if fold_median_score >= 0.50: MAX_FOLD_DRAWDOWN = -1240.0
            elif fold_median_score >= 0.25: MAX_FOLD_DRAWDOWN = -1080.0
            else: MAX_FOLD_DRAWDOWN = -880.0

            print(f"  -> Dynamische Fold-Bremse: {MAX_FOLD_DRAWDOWN:.0f}$")
            max_allowed_dead_seeds = max(1, int(len(top_diverse_trials) * 0.70))
            asset_running_pnl = {}

            for t in raw_fold_trades:
                if fold_cb_hit: break
                asset = t[-1]
                if asset_running_pnl.get(asset, 0.0) <= CONFIG.get("MAX_OOS_LOSS_PER_ASSET", -9999.0): continue

                actual_trade_pnl = (t[2] * t[3]) - t[4]
                global_fold_pnl += actual_trade_pnl
                filtered_fold_trades.append(t)
                asset_running_pnl[asset] = asset_running_pnl.get(asset, 0.0) + actual_trade_pnl

                dead_seeds_now = sum(1 for dt in seed_death_times if dt <= t[1])

                if global_fold_pnl <= MAX_FOLD_DRAWDOWN:
                    print(f"    🚨 GLOBAL FOLD STOP HIT! PnL fiel auf {global_fold_pnl:.2f} $.")
                    fold_cb_hit = True
                    premature_end_time = t[1]
                    stop_reason = "fold_brake"
                elif dead_seeds_now >= max_allowed_dead_seeds:
                    print(f"    🚨 SEED CRASH STOP HIT! {dead_seeds_now} von {len(top_diverse_trials)} Seeds sind tot.")
                    fold_cb_hit = True
                    premature_end_time = t[1]
                    stop_reason = "dead_seed"

            all_oos_trades.extend(filtered_fold_trades)
            kpis = WalkForwardEngine.calc_kpis(filtered_fold_trades)
            print(f"  => Fold {fold_idx+1} Abgeschlossen! OOS Profit: {kpis['profit']:+,.2f} $")

            fold_logs.append({"Fold": fold_idx + 1,
                              "Net_Profit_USD": kpis['profit'],
                              "Circuit_Breakers_Hit": fold_stats['cb_hits']})
            for a, pnl in asset_running_pnl.items():
                all_asset_fold_records.append({"Fold": fold_idx + 1, "Asset": a, "PnL_USD": pnl})
            shift_events.append({"fold": fold_idx + 1,
                                 "cut_short": premature_end_time is not None,
                                 "reason": stop_reason, "cash_fold": False})

            # DYNAMISCHES FENSTER-SHIFTING
            if premature_end_time is not None:
                new_start = premature_end_time - relativedelta(months=CONFIG["TRAIN_MONTHS"])
                if new_start > curr_start:
                    print(f"  => 🔄 FOLD VORZEITIG BEENDET am {premature_end_time}. Optimiere sofort neu!")
                    curr_start = new_start
                else:
                    # Endlosschleifen-Schutz: Re-Anchoring wuerde NICHT vorruecken
                    # -> regulaerer Schritt erzwingen, damit die WFA terminiert.
                    print(f"  => [!] Dynamic Shift ohne Fortschritt am {premature_end_time}. Regulaerer Schritt erzwungen.")
                    curr_start += relativedelta(months=CONFIG["TEST_MONTHS"])
            else:
                curr_start += relativedelta(months=CONFIG["TEST_MONTHS"])

            fold_idx += 1

        ReportGenerator.generate(
            name=CONFIG["EXPERIMENT_NAME"],
            fold_logs=fold_logs,
            asset_fold_records=all_asset_fold_records,
            all_oos_trades=all_oos_trades,
            config_dict=CONFIG,
            shift_events=shift_events,
            universe_size=universe_size,
            data_span=data_span,
        )
