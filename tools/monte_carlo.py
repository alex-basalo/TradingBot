"""
Monte-Carlo-Robustheitsanalyse mit drei Resampling-Verfahren.

METHODE (per METHOD wählbar):
  "iid"   - Ziehen einzelner Trades mit Zuruecklegen. Zerstört die zeitliche
            Struktur; unterschaetzt den Drawdown systematisch. Nur als
            Referenz/Vergleich, nicht als Risikomaß geeignet.
  "block" - Ziehen zusammenhängender Blöcke von Trades. Erhält die lokale
            Autokorrelation und damit die Verlustclusterung.
  "fold"  - Ziehen ganzer Folds mit Zurücklegen. Erhält die komplette interne
            Struktur jedes Walk-Forward-Fensters und bildet die Unsicherheit
            über die Fold-Auswahl ab. 

Erfordert für METHOD="fold" zusätzlich die {NAME}_FoldLog.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- SETUP ---
TRADE_FILE = "FF1_2020-2026_28A_HistData_Trades.csv"
FOLD_FILE = "FF1_2020-2026_28A_HistData_FoldLog.csv"   # nur für METHOD="fold"
METHOD = "block"          # "fold" | "block" | "iid"
BLOCK_SIZE = 50          # nur für METHOD="block"
START_CAPITAL = 10000.0
SIMULATIONS = 10000
SEED = 42


def load_pnls(path):
    df = pd.read_csv(path)
    if "pnl_usd" in df.columns:
        pnl = df["pnl_usd"].to_numpy(dtype=float)
    else:
        pnl = (df["r_mult"] * df["risk_usd"] - df["fee_usd"]).to_numpy(dtype=float)
    return pnl


def load_fold_pnls(path):
    df = pd.read_csv(path)
    return df["Net_Profit_USD"].to_numpy(dtype=float)


def path_stats(pnls):
    """Endkapital und maximaler Drawdown eines Pfades."""
    equity = START_CAPITAL + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return equity[-1], float(dd.max()), equity


def simulate(rng, source, method):
    if method == "iid":
        idx = rng.randint(0, len(source), size=len(source))
        return source[idx]
    if method == "block":
        n = len(source)
        n_blocks = int(np.ceil(n / BLOCK_SIZE))
        starts = rng.randint(0, max(1, n - BLOCK_SIZE), size=n_blocks)
        seq = np.concatenate([source[s:s + BLOCK_SIZE] for s in starts])
        return seq[:n]
    if method == "fold":
        idx = rng.randint(0, len(source), size=len(source))
        return source[idx]
    raise ValueError(f"unbekannte METHOD: {method}")


def main():
    if not os.path.exists(TRADE_FILE):
        print(f"Fehler: '{TRADE_FILE}' nicht gefunden.")
        return
    name = os.path.basename(TRADE_FILE).replace("_Trades.csv", "")

    if METHOD == "fold":
        if not os.path.exists(FOLD_FILE):
            print(f"Fehler: '{FOLD_FILE}' nicht gefunden (für METHOD='fold' nötig).")
            return
        source = load_fold_pnls(FOLD_FILE)
        unit = "Folds"
    else:
        source = load_pnls(TRADE_FILE)
        unit = "Trades"

    n_units = len(source)
    real_total = float(source.sum())
    rng = np.random.RandomState(SEED)

    finals, dds, paths = [], [], []
    print(f"Simuliere {SIMULATIONS} Pfade | Methode: {METHOD} | {n_units} {unit}...")
    for i in range(SIMULATIONS):
        seq = simulate(rng, source, METHOD)
        fin, dd, eq = path_stats(seq)
        finals.append(fin)
        dds.append(dd)
        if i < 150:
            paths.append(eq)

    finals, dds = np.array(finals), np.array(dds)

    # --- Kennzahlen ---
    p_loss = float((finals < START_CAPITAL).mean() * 100)   # Anteil Pfade mit Verlust
    p_ruin = float((finals <= 0).mean() * 100)
    q05, q50, q95 = np.percentile(finals, [5, 50, 95])
    dd_mean, dd_95 = float(dds.mean()), float(np.percentile(dds, 95))

    print("\n=== MONTE-CARLO-STATISTIK ===")
    print(f"  Methode                : {METHOD} ({n_units} {unit} je Pfad)")
    print(f"  Realer Endstand        : {START_CAPITAL + real_total:,.0f} $")
    print(f"  Median Endstand        : {q50:,.0f} $")
    print(f"  5%-Quantil  Endstand   : {q05:,.0f} $")
    print(f"  95%-Quantil Endstand   : {q95:,.0f} $")
    print(f"  Anteil Pfade mit Verlust: {p_loss:.2f} %")
    print(f"  Ruinwahrscheinlichkeit : {p_ruin:.4f} %")
    print(f"  Ø Max-Drawdown         : {dd_mean:,.0f} $")
    print(f"  95%-Quantil Max-DD     : {dd_95:,.0f} $")

    # --- Grafik ---
    plt.figure(figsize=(14, 7))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f} $"))
    for p in paths:
        plt.plot(p, color="gray", alpha=0.12, linewidth=0.9)
    med_len = min(len(p) for p in paths)
    med_path = np.median(np.array([p[:med_len] for p in paths]), axis=0)
    plt.plot(med_path, color="blue", linewidth=2.2, label="Median-Pfad")
    plt.axhline(START_CAPITAL, color="black", linestyle="--", linewidth=1.4,
                label="Startkapital")
    plt.axhline(START_CAPITAL + real_total, color="green", linestyle=":",
                linewidth=1.8, label="realer Endstand")

    plt.title(f"Monte-Carlo-Simulation ({METHOD}-Bootstrap): {name} | {SIMULATIONS} Pfade",
              fontsize=13, fontweight="bold")
    plt.xlabel(unit)
    plt.ylabel("Kontostand ($)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")

    stats = (f"Methode: {METHOD}-Bootstrap\n"
             f"Simulationen: {SIMULATIONS:,}\n\n"
             f"Median Endstand: {q50:,.0f} $\n"
             f"5%-Quantil:  {q05:,.0f} $\n"
             f"95%-Quantil: {q95:,.0f} $\n\n"
             f"Pfade mit Verlust: {p_loss:.1f} %\n"
             f"Ø Max-Drawdown: {dd_mean:,.0f} $\n"
             f"95%-Quantil Max-DD: {dd_95:,.0f} $")
    plt.figtext(0.14, 0.55, stats,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="black"), fontsize=10)

    plt.tight_layout()
    out = f"Monte_Carlo_{METHOD}_{name}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"\n=> Grafik gespeichert: {out}")


if __name__ == "__main__":
    main()
