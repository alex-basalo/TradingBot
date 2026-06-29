"""
Monte-Carlo-Robustheitsanalyse auf Basis der realisierten OOS-Trades eines Laufs.

Bootstrapping (Ziehen mit Zuruecklegen) der tatsaechlich realisierten Trade-PnLs
aus der {NAME}_Trades.csv. Dadurch bleibt das reale, von der State Machine
skalierte Risiko je Trade erhalten; lediglich die Reihenfolge der Trades wird
zufaellig neu gewuerfelt. Geschaetzt werden Endkapital-, Drawdown- und
Ruin-Verteilung ueber viele Pfade.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- SETUP ---
TRADE_FILE = "FF1_Main_2020-2026_28A_HistData_Trades.csv"   # Trades-CSV des Laufs
START_CAPITAL = 10000.0
SIMULATIONS = 10000


def _load_pnls(path: str) -> np.ndarray:
    """Liest die realisierten Trade-PnLs robust aus der Trades-CSV."""
    df = pd.read_csv(path)
    if "pnl_usd" in df.columns:
        return df["pnl_usd"].to_numpy(dtype=float)
    # Fallback: aus den Rohspalten rekonstruieren (r_mult * risk_usd - fee_usd)
    needed = {"r_mult", "risk_usd", "fee_usd"}
    if needed.issubset(df.columns):
        return (df["r_mult"] * df["risk_usd"] - df["fee_usd"]).to_numpy(dtype=float)
    raise KeyError(
        "Weder 'pnl_usd' noch ('r_mult','risk_usd','fee_usd') in der CSV gefunden."
    )


def run_real_monte_carlo():
    if not os.path.exists(TRADE_FILE):
        print(f"Fehler: '{TRADE_FILE}' nicht gefunden.")
        return

    experiment_name = os.path.basename(TRADE_FILE).replace("_Trades.csv", "")
    real_pnls = _load_pnls(TRADE_FILE)
    n_trades = len(real_pnls)
    if n_trades == 0:
        print("Keine Trades in der Datei.")
        return

    final_balances, max_drawdowns, all_paths = [], [], []
    best_path, worst_path = [], []
    max_balance, min_balance = -float("inf"), float("inf")

    print(f"Simuliere {SIMULATIONS} Pfade aus {n_trades} realisierten Trades...")

    for i in range(SIMULATIONS):
        # Bootstrapping: Ziehen mit Zuruecklegen der realisierten PnLs
        shuffled = random.choices(real_pnls.tolist(), k=n_trades)

        balance, peak, mdd = START_CAPITAL, START_CAPITAL, 0.0
        path = [balance]
        for pnl in shuffled:
            balance += pnl
            path.append(balance)
            if balance > peak:
                peak = balance
            dd = peak - balance
            if dd > mdd:
                mdd = dd

        final_balances.append(balance)
        max_drawdowns.append(mdd)

        if i < 150:                       # nur eine Stichprobe fuer den Hintergrund
            all_paths.append(path)
        if balance > max_balance:
            max_balance, best_path = balance, path.copy()
        if balance < min_balance:
            min_balance, worst_path = balance, path.copy()

    # --- AUSWERTUNG ---
    prob_ruin = len([b for b in final_balances if b <= 0]) / SIMULATIONS * 100
    avg_dd = float(np.mean(max_drawdowns))
    perc_95_dd = float(np.percentile(max_drawdowns, 95))
    avg_final = float(np.mean(final_balances))

    print("\n=== MONTE CARLO STATISTIK ===")
    print(f"Ruin-Wahrscheinlichkeit:      {prob_ruin:.4f} %")
    print(f"Durchschn. Endkapital:        {avg_final:,.0f} $")
    print(f"95%-Konfidenz Max-Drawdown:   {perc_95_dd:,.0f} $")

    # --- GRAFIK ---
    plt.figure(figsize=(14, 8))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f} $"))

    for p in all_paths:
        plt.plot(p, color="gray", alpha=0.15, linewidth=1)

    avg_path = np.mean(all_paths, axis=0)
    plt.plot(avg_path, color="blue", linewidth=2.5, label="Ø Erwartungswert")
    plt.plot(best_path, color="green", linewidth=2, label="Best Case")
    plt.plot(worst_path, color="red", linewidth=2, label="Worst Case")
    plt.axhline(START_CAPITAL, color="black", linestyle="--", linewidth=1.5)

    plt.title(f"Monte-Carlo-Simulation: {experiment_name} | {SIMULATIONS} Pfade",
              fontsize=14, fontweight="bold")
    plt.xlabel("Anzahl der Trades", fontsize=12)
    plt.ylabel("Kontostand ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    stats_text = (
        f"--- MONTE CARLO STATISTIK ---\n"
        f"Simulationen: {SIMULATIONS:,.0f}\n"
        f"Trades je Pfad: {n_trades:,.0f}\n"
        f"Totalverlust-Risiko: {prob_ruin:.4f} %\n\n"
        f"Ø Max Drawdown: {avg_dd:,.0f} $\n"
        f"95%-Konfidenz-DD: {perc_95_dd:,.0f} $\n\n"
        f"Worst Case Endstand: {min_balance:,.0f} $\n"
        f"Ø Erwarteter Endstand: {avg_final:,.0f} $\n"
        f"Best Case Endstand: {max_balance:,.0f} $"
    )
    plt.figtext(0.12, 0.45, stats_text,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"), fontsize=11)

    plt.tight_layout()
    out = f"Monte_Carlo_{experiment_name}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"=> Grafik gespeichert als '{out}'")


if __name__ == "__main__":
    run_real_monte_carlo()
