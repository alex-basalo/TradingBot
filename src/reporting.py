"""
Vollstaendige Ergebnis-Dokumentation eines Walk-Forward-Laufs.

Reines Post-Processing auf den fertigen Listen (Fold-Logs, Trades) – veraendert
weder Trades noch Zufallssequenzen und kann das Ergebnis nicht beeinflussen.
"""

import os
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Erzeugte Artefakte je Lauf (Prefix = EXPERIMENT_NAME):
        {name}_Summary.json      - alle Kennzahlen + Metadaten
        {name}_Trades.csv        - vollstaendige Trade-Liste (Audit + Monte-Carlo)
        {name}_FoldLog.csv       - PnL, CB-Hits, Cut-Short je Fold
        {name}_AssetSummary.csv  - aggregierter PnL je Waehrungspaar
        {name}_Equity.png        - Equity-Kurve (OOS)
        {name}_Underwater.png    - Drawdown-Verlauf
        {name}_FoldPnL.png       - Balkendiagramm der Fold-Ergebnisse
        runs_comparison.csv      - eine Zeile je Lauf (wird angehaengt)
    """

    TRADING_DAYS = 252

    # ---------------------------------------------------------------- Helfer
    @staticmethod
    def _pnl(t):
        """PnL eines Trade-Tupels: R-Vielfaches * Risiko - Gebuehr."""
        return (t[2] * t[3]) - t[4]

    @staticmethod
    def _trade_equity(trades, start_capital):
        """Realisierte Equity-Kurve je Trade (sortiert nach Ausstiegszeit)."""
        ordered = sorted(trades, key=lambda t: t[1])
        pnls = np.array([ReportGenerator._pnl(t) for t in ordered], dtype=float)
        equity = start_capital + np.cumsum(pnls)
        return ordered, pnls, equity

    @staticmethod
    def _daily_equity(trades, start_capital):
        """Taegliche Equity-Kurve (realisierter PnL je Ausstiegstag)."""
        if not trades:
            return pd.Series(dtype=float)
        rows = [(pd.to_datetime(t[1]), ReportGenerator._pnl(t)) for t in trades]
        df = pd.DataFrame(rows, columns=["exit", "pnl"]).set_index("exit").sort_index()
        daily = df["pnl"].resample("D").sum()
        return start_capital + daily.cumsum()

    @staticmethod
    def _drawdown_stats(equity_daily):
        """Max-Drawdown (absolut, Prozent), Underwater-Dauer und Erholungszeit."""
        out = {"max_dd_abs": 0.0, "max_dd_pct": 0.0,
               "max_dd_duration_days": 0, "recovery_days": None}
        if equity_daily is None or len(equity_daily) < 2:
            return out, None
        peak = equity_daily.cummax()
        dd = equity_daily - peak
        out["max_dd_abs"] = float(-dd.min())
        out["max_dd_pct"] = float((-(dd / peak)).max() * 100)

        trough_date = dd.idxmin()
        pre = equity_daily.loc[:trough_date]
        peak_date = pre.idxmax()
        peak_val = float(pre.max())
        post = equity_daily.loc[trough_date:]
        recovered = post[post >= peak_val]
        if len(recovered) > 0:
            rec_date = recovered.index[0]
            out["recovery_days"] = int((rec_date - trough_date).days)
            out["max_dd_duration_days"] = int((rec_date - peak_date).days)
        else:
            out["max_dd_duration_days"] = int((equity_daily.index[-1] - peak_date).days)
        return out, dd

    @staticmethod
    def _safe_version(pkg):
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            return "n/a"

    @staticmethod
    def _git_commit():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "n/a"

    # ----------------------------------------------------------- Hauptmethode
    @staticmethod
    def generate(name, fold_logs, asset_fold_records, all_oos_trades,
                 config_dict, shift_events=None, universe_size=None,
                 data_span=(None, None)):
        """Schreibt Summary, Tabellen, Vergleichszeile und Plots fuer einen Lauf."""
        shift_events = shift_events or []
        start_capital = float(config_dict.get("START_CAPITAL", 10000.0))

        # ------------------------------------------------------------- Trades
        n_trades = len(all_oos_trades)
        if n_trades > 0:
            _, pnls, _ = ReportGenerator._trade_equity(all_oos_trades, start_capital)
            net_profit = float(pnls.sum())
            gross_profit = float(pnls[pnls > 0].sum())
            gross_loss = float(-pnls[pnls < 0].sum())
            win_rate = float((pnls > 0).mean() * 100)
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

            first_entry = pd.to_datetime(min(t[0] for t in all_oos_trades))
            last_exit = pd.to_datetime(max(t[1] for t in all_oos_trades))
            years = max((last_exit - first_entry).days / 365.25, 1e-9)
            final_equity = start_capital + net_profit
            total_return_pct = (final_equity / start_capital - 1.0) * 100
            cagr_pct = (((final_equity / start_capital) ** (1.0 / years) - 1.0) * 100
                        if final_equity > 0 else -100.0)

            eq_daily = ReportGenerator._daily_equity(all_oos_trades, start_capital)
            rets = eq_daily.pct_change().dropna()
            if len(rets) > 1 and rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * np.sqrt(ReportGenerator.TRADING_DAYS))
                downside = rets[rets < 0]
                sortino = (float(rets.mean() / downside.std() * np.sqrt(ReportGenerator.TRADING_DAYS))
                           if len(downside) > 1 and downside.std() > 0 else None)
            else:
                sharpe = sortino = None
            dd_stats, dd_series = ReportGenerator._drawdown_stats(eq_daily)
        else:
            net_profit = total_return_pct = cagr_pct = 0.0
            win_rate = 0.0
            profit_factor = sharpe = sortino = None
            dd_stats = {"max_dd_abs": 0.0, "max_dd_pct": 0.0,
                        "max_dd_duration_days": 0, "recovery_days": None}
            dd_series = None
            first_entry = last_exit = None
            eq_daily = pd.Series(dtype=float)

        pf_out = None if profit_factor is None else round(profit_factor, 3)

        # -------------------------------------------------------------- Folds
        fold_pnls = [float(f.get("Net_Profit_USD", 0.0)) for f in fold_logs]
        n_folds = len(fold_pnls)
        if n_folds > 0:
            arr = np.array(fold_pnls, dtype=float)
            fold_stats = {
                "n_folds": n_folds,
                "profitable_folds": int((arr > 0).sum()),
                "profitable_folds_pct": float((arr > 0).mean() * 100),
                "best_fold": float(arr.max()),
                "worst_fold": float(arr.min()),
                "median_fold": float(np.median(arr)),
                "mean_fold": float(arr.mean()),
                "std_fold": float(arr.std()),
            }
        else:
            fold_stats = {"n_folds": 0, "profitable_folds": 0, "profitable_folds_pct": 0.0,
                          "best_fold": 0.0, "worst_fold": 0.0, "median_fold": 0.0,
                          "mean_fold": 0.0, "std_fold": 0.0}

        # ----------------------------------------------------- Schutzmechanismen
        seed_cb_hits = int(sum(f.get("Circuit_Breakers_Hit", 0) for f in fold_logs))
        folds_cut = [e for e in shift_events if e.get("cut_short")]
        fold_brake_hits = sum(1 for e in folds_cut if e.get("reason") == "fold_brake")
        dead_seed_hits = sum(1 for e in folds_cut if e.get("reason") == "dead_seed")
        cash_folds = sum(1 for e in shift_events if e.get("cash_fold"))
        cap = config_dict.get("MAX_OOS_LOSS_PER_ASSET", -200.0)
        asset_cap_hits = sum(1 for r in (asset_fold_records or [])
                             if r.get("PnL_USD", 0.0) <= cap)

        safeguards = {
            "seed_trailing_cb_hits": seed_cb_hits,
            "fold_brake_hits": fold_brake_hits,
            "dead_seed_stop_hits": dead_seed_hits,
            "dynamic_shifts_total": len(folds_cut),
            "cash_folds": cash_folds,
            "per_asset_cap_hits": asset_cap_hits,
        }

        # ------------------------------------------------------------ Metadaten
        base_seed = config_dict.get("BASE_SEED", config_dict.get("ENSEMBLE_BASE_SEED"))
        n_seeds = config_dict.get("NUM_ENSEMBLES", config_dict.get("NUM_SEEDS"))
        if base_seed is not None and n_seeds:
            seed_band = f"{base_seed}..{base_seed + n_seeds - 1}"
        else:
            seed_band = str(base_seed)

        meta = {
            "experiment_name": name,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": ReportGenerator._git_commit(),
            "data_dir": config_dict.get("RAW_DATA_DIR"),
            "data_start_filter": config_dict.get("DATA_START"),
            "data_end_filter": config_dict.get("DATA_END"),
            "banned_assets": config_dict.get("BANNED_ASSETS", []),
            "data_span_actual": [str(data_span[0]), str(data_span[1])],
            "universe_size": universe_size,
            "assets_traded": int(len({t[14] for t in all_oos_trades})) if n_trades else 0,
            "seed_band": seed_band,
            "versions": {p: ReportGenerator._safe_version(p) for p in
                         ["optuna", "hmmlearn", "pandas", "numpy",
                          "pandas_ta", "scikit-learn"]},
        }

        summary = {
            "meta": meta,
            "performance": {
                "net_profit_usd": round(net_profit, 2),
                "total_return_pct": round(total_return_pct, 2),
                "cagr_pct": round(cagr_pct, 2),
                "sharpe_daily_annualized": round(sharpe, 3) if sharpe is not None else None,
                "sortino_daily_annualized": round(sortino, 3) if sortino is not None else None,
                "profit_factor": pf_out,
                "win_rate_pct": round(win_rate, 2),
                "n_trades": n_trades,
                "first_entry": str(first_entry) if first_entry is not None else None,
                "last_exit": str(last_exit) if last_exit is not None else None,
            },
            "drawdown": {k: (round(v, 2) if isinstance(v, float) else v)
                         for k, v in dd_stats.items()},
            "folds": fold_stats,
            "safeguards": safeguards,
            "config": config_dict,
        }

        with open(f"{name}_Summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

        # --------------------------------------------------- Vergleichszeile (CSV)
        flat = {
            "experiment": name, "data_dir": meta["data_dir"],
            "span": f"{meta['data_start_filter']}..{meta['data_end_filter']}",
            "universe": universe_size, "assets_traded": meta["assets_traded"],
            "seed_band": seed_band, "net_profit": round(net_profit, 2),
            "cagr_pct": round(cagr_pct, 2),
            "sharpe": round(sharpe, 3) if sharpe is not None else None,
            "sortino": round(sortino, 3) if sortino is not None else None,
            "profit_factor": pf_out, "win_rate": round(win_rate, 2),
            "n_trades": n_trades, "max_dd_abs": round(dd_stats["max_dd_abs"], 2),
            "max_dd_pct": round(dd_stats["max_dd_pct"], 2),
            "max_dd_days": dd_stats["max_dd_duration_days"],
            "n_folds": fold_stats["n_folds"],
            "profitable_folds_pct": round(fold_stats["profitable_folds_pct"], 1),
            "best_fold": round(fold_stats["best_fold"], 2),
            "worst_fold": round(fold_stats["worst_fold"], 2),
            "median_fold": round(fold_stats["median_fold"], 2),
            "seed_cb_hits": seed_cb_hits, "fold_brake_hits": fold_brake_hits,
            "dead_seed_hits": dead_seed_hits, "dynamic_shifts": len(folds_cut),
            "cash_folds": cash_folds, "asset_cap_hits": asset_cap_hits,
        }
        comp_path = "runs_comparison.csv"
        row = pd.DataFrame([flat])
        if os.path.exists(comp_path):
            row.to_csv(comp_path, mode="a", header=False, index=False)
        else:
            row.to_csv(comp_path, index=False)

        # ----------------------------------------------------- Trade-Liste (CSV)
        if n_trades > 0:
            rows = list(all_oos_trades)
            if all(len(r) == 15 for r in rows):
                cols = ["entry_time", "exit_time", "r_mult", "risk_usd", "fee_usd",
                        "notional", "trade_type", "entry_price", "sl_price", "tp_price",
                        "zone_lower", "zone_upper", "momentum", "atr", "asset"]
                tdf = pd.DataFrame(rows, columns=cols)
                tdf["pnl_usd"] = tdf["r_mult"] * tdf["risk_usd"] - tdf["fee_usd"]
                tdf.sort_values("exit_time").to_csv(f"{name}_Trades.csv", index=False)
            else:
                pd.DataFrame(rows).to_csv(f"{name}_Trades.csv", index=False)

        # ------------------------------------------------------- Fold-Log (CSV)
        by_fold = {e.get("fold"): e for e in shift_events}
        fl_rows = []
        for k, f in enumerate(fold_logs, start=1):
            fnum = f.get("Fold", k)
            ev = by_fold.get(fnum, {})
            fl_rows.append({"Fold": fnum,
                            "Net_Profit_USD": round(float(f.get("Net_Profit_USD", 0.0)), 2),
                            "Circuit_Breakers_Hit": f.get("Circuit_Breakers_Hit", 0),
                            "Cut_Short": ev.get("cut_short", False),
                            "Stop_Reason": ev.get("reason")})
        if fl_rows:
            pd.DataFrame(fl_rows).to_csv(f"{name}_FoldLog.csv", index=False)

        # --------------------------------------------------- Asset-Summary (CSV)
        if asset_fold_records:
            adf = pd.DataFrame(asset_fold_records)
            g = adf.groupby("Asset")["PnL_USD"]
            summ = pd.DataFrame({
                "total_pnl": g.sum().round(2),
                "folds_active": g.count(),
                "folds_capped": g.apply(lambda s: int((s <= cap).sum())),
            }).reset_index().sort_values("total_pnl", ascending=False)
            summ.to_csv(f"{name}_AssetSummary.csv", index=False)

        # ----------------------------------------------- Equity-Kurve (Hauptgrafik)
        if n_trades > 0 and len(eq_daily) > 1:
            plt.figure(figsize=(14, 5))
            plt.plot(eq_daily.index, eq_daily.values, color="#1f6f54", linewidth=1.4)
            plt.axhline(start_capital, color="gray", linestyle="--", linewidth=0.8)
            plt.title(f"Equity-Kurve (OOS) - {name}", fontsize=12, fontweight="bold")
            plt.ylabel("Kapital ($)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{name}_Equity.png", dpi=200)
            plt.close()

        # ------------------------------------------------------- Underwater-Plot
        if dd_series is not None and len(dd_series) > 1:
            plt.figure(figsize=(14, 4))
            plt.fill_between(dd_series.index, dd_series.values, 0.0,
                             color="#c0392b", alpha=0.6)
            plt.title(f"Underwater (Drawdown) - {name}", fontsize=12, fontweight="bold")
            plt.ylabel("Drawdown ($)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{name}_Underwater.png", dpi=200)
            plt.close()

        # -------------------------------------------------------- Fold-PnL-Balken
        if n_folds > 0:
            plt.figure(figsize=(14, 4))
            colors = ["#27ae60" if p > 0 else "#c0392b" for p in fold_pnls]
            plt.bar(range(1, n_folds + 1), fold_pnls, color=colors)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(f"OOS-Ergebnis je Fold - {name}", fontsize=12, fontweight="bold")
            plt.xlabel("Fold")
            plt.ylabel("Netto-PnL ($)")
            plt.grid(True, axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{name}_FoldPnL.png", dpi=200)
            plt.close()

        # ----------------------------------------------------------- Konsolen-Echo
        print("\n" + "=" * 85)
        print(f"REPORT: {name}")
        print("=" * 85)
        print(f"  Net Profit : {net_profit:+,.2f} $   | CAGR: {cagr_pct:+.2f} %   "
              f"| Sharpe(d,ann): {summary['performance']['sharpe_daily_annualized']}")
        print(f"  Max DD     : {dd_stats['max_dd_abs']:,.2f} $ "
              f"({dd_stats['max_dd_pct']:.1f} %, {dd_stats['max_dd_duration_days']} Tage)")
        print(f"  Folds      : {fold_stats['n_folds']} "
              f"({fold_stats['profitable_folds_pct']:.0f} % profitabel) "
              f"| Median-Fold: {fold_stats['median_fold']:+,.0f} $")
        print(f"  Schutz     : Seed-CB {seed_cb_hits} | Fold-Bremse {fold_brake_hits} "
              f"| Dead-Seed {dead_seed_hits} | Dyn-Shifts {len(folds_cut)} "
              f"| Cash-Folds {cash_folds} | Asset-Cap {asset_cap_hits}")
        print("=" * 85)

        return summary
