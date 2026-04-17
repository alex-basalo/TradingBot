import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import random

# --- SETUP ---
# Trage hier den exakten Namen der CSV-Datei ein
TRADE_FILE = "Ensemble_Hybrid_Modell_OOS_MC_Trades.csv" 
START_CAPITAL = 10000.0
RISK_PER_TRADE = 100.0
FEE_PER_TRADE = 3.0    
SIMULATIONS = 10000    

def run_real_monte_carlo():
    if not os.path.exists(TRADE_FILE):
        print(f"Fehler: '{TRADE_FILE}' nicht gefunden.")
        return

    # Name dynamisch aus der Datei auslesen
    experiment_name = TRADE_FILE.replace('_MC_Trades.csv', '')
    
    df_trades = pd.read_csv(TRADE_FILE)
    real_r_multiples = df_trades['R_Mult'].tolist()
    
    final_balances = []
    max_drawdowns = []
    all_paths = [] 
    
    # Tracking für Best und Worst Case
    best_path = []
    worst_path = []
    max_balance = -float('inf')
    min_balance = float('inf')

    print(f"Simuliere {SIMULATIONS} Pfade...")

    for i in range(SIMULATIONS):
        # Ziehen mit Zurücklegen (Bootstrapping) für realistische Zufallsverteilung
        shuffled = random.choices(real_r_multiples, k=len(real_r_multiples))
        
        balance = START_CAPITAL
        path = [balance]
        peak = balance
        mdd = 0
        
        for r in shuffled:
            pnl = (r * RISK_PER_TRADE) - FEE_PER_TRADE
            balance += pnl
            path.append(balance)
            
            if balance > peak: 
                peak = balance
            dd = peak - balance
            if dd > mdd: 
                mdd = dd
        
        final_balances.append(balance)
        max_drawdowns.append(mdd)
        
        # Nur eine Stichprobe von 150 Pfaden für den grauen Hintergrund speichern (spart RAM)
        if i < 150: 
            all_paths.append(path)
            
        # Best und Worst Case über ALLE 10.000 Simulationen tracken
        if balance > max_balance:
            max_balance = balance
            best_path = path.copy()
        if balance < min_balance:
            min_balance = balance
            worst_path = path.copy()

    # --- AUSWERTUNG ---
    prob_ruin = len([b for b in final_balances if b <= 0]) / SIMULATIONS * 100
    avg_dd = np.mean(max_drawdowns)
    perc_95_dd = np.percentile(max_drawdowns, 95) 
    avg_final = np.mean(final_balances)

    print("\n=== MONTE CARLO STATISTIK ===")
    print(f"Ruin-Wahrscheinlichkeit: {prob_ruin:.4f}%")
    print(f"95% Value-at-Risk (Max DD): {perc_95_dd:.2f}€")

    # --- GRAFIK ---
    plt.figure(figsize=(14, 8))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f} €".replace(',', '.')))
    
    # 1. Hintergrund-Pfade (grau transparent)
    for p in all_paths:
        plt.plot(p, color='gray', alpha=0.15, linewidth=1)
        
    # 2. Durchschnitt (blau)
    avg_path = np.mean(all_paths, axis=0)
    plt.plot(avg_path, color='blue', linewidth=2.5, label="Ø Erwartungswert")
    
    # 3. Best Case (grün) und Worst Case (rot)
    plt.plot(best_path, color='green', linewidth=2, label="Best Case")
    plt.plot(worst_path, color='red', linewidth=2, label="Worst Case")

    plt.axhline(START_CAPITAL, color='black', linestyle='--', linewidth=1.5)
    
    plt.title(f"Monte Carlo Simulation: {experiment_name} | {SIMULATIONS} Pfade", fontsize=14, fontweight='bold')
    plt.xlabel("Anzahl der Trades", fontsize=12)
    plt.ylabel("Kontostand (€)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # --- TEXTBOX ---
    stats_text = (
        f"--- MONTE CARLO STATISTIK ---\n"
        f"Simulationen: {SIMULATIONS:,.0f}\n"
        f"Totalverlust-Risiko: {prob_ruin:.4f} %\n\n"
        f"Ø Max Drawdown: {avg_dd:,.0f} €\n"
        f"95% Konfidenz-DD: {perc_95_dd:,.0f} €\n\n"
        f"Worst Case Endstand: {min_balance:,.0f} €\n"
        f"Ø Erwarteter Endstand: {avg_final:,.0f} €\n"
        f"Best Case Endstand: {max_balance:,.0f} €"
    )
    # Positionierung der Box
    plt.figtext(0.12, 0.45, stats_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'), fontsize=11)

    plt.tight_layout()
    plt.savefig(f"Monte_Carlo_{experiment_name}.png", dpi=300)
    print(f"=> Grafik gespeichert als 'Monte_Carlo_{experiment_name}.png'")
    plt.close()

if __name__ == "__main__":
    run_real_monte_carlo()
