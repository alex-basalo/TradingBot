import pandas as pd
import os
from optimizer import ClusterOptimizer

# --- EINSTELLUNGEN ---
N_TRIALS_PER_CLUSTER = 500  # Für die echte Thesis auf 500 oder 1000 stellen
DATA_DIR = "mt5_h1_daten_regimes"
CONFIG_FILE = "bot_cluster_config.csv"
OUTPUT_FILE = "final_cluster_parameters.csv"

def get_filename_for_asset(yahoo_ticker: str, available_files: list) -> str:
    """
    Übersetzt die Yahoo-Ticker aus der CSV in die echten MT5-Dateinamen.
    """
    # Sonderzeichen entfernen
    clean_ticker = yahoo_ticker.replace('=X', '').replace('^', '').replace('-', '')
    
    # Mapping für Indizes und Rohstoffe 
    mapping = {
        'DJI': 'US30', 'GSPC': 'US500', 'NDX': 'NAS100', 'RUT': 'US2000',
        'N225': 'JPN225', 'FCHI': 'FRA40', 'IBEX': 'SPA35', 'HSI': 'HK50',
        'STOXX50E': 'EUSTX50', 'FTSE': 'UK100', 'GDAXI': 'GER40', 'AXJO': 'AUS200',
        'CL=F': 'XTIUSD', 'BZ=F': 'XBRUSD', 'NG=F': 'XNGUSD', 'HG=F': 'XCUUSD',
        'GC=F': 'XAUUSD', 'SI=F': 'XAGUSD', 'PL=F': 'XPTUSD', 'PA=F': 'XPDUSD',
        'ZC=F': 'Corn', 'ZW=F': 'Wheat', 'SB=F': 'Sugar', 'KC=F': 'Coffee'
    }
    
    search_name = mapping.get(clean_ticker, clean_ticker)
    
    for file in available_files:
        if file.startswith(search_name + "_H1"):
            return file
    return None

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"Fehler: '{CONFIG_FILE}' nicht gefunden!")
        return
        
    cluster_config = pd.read_csv(CONFIG_FILE)
    cluster_ids = sorted(cluster_config['Cluster_ID'].unique())
    available_files = os.listdir(DATA_DIR)
    
    final_results = []

    print(f"Starte Master-Optimierung für {len(cluster_ids)} Cluster...")
    print(f"Trials pro Cluster: {N_TRIALS_PER_CLUSTER}\n")

    for cid in cluster_ids:
        print(f"{'='*50}")
        print(f"LADE DATEN FÜR CLUSTER {cid}")
        print(f"{'='*50}")
        
        assets_in_cluster = cluster_config[cluster_config['Cluster_ID'] == cid]['Asset'].tolist()
        cluster_data = {}
        
        # Lade alle Assets für dieses Cluster in den RAM
        for asset in assets_in_cluster:
            filename = get_filename_for_asset(asset, available_files)
            if filename:
                filepath = os.path.join(DATA_DIR, filename)
                df = pd.read_csv(filepath, index_col='time', parse_dates=True)
                
                # Train-Split (erste 70% der Daten nutzen)
                df_train = df.iloc[:int(len(df)*0.7)]
                cluster_data[asset] = df_train
            else:
                print(f"  -> Warnung: Keine Datei für {asset} gefunden. Wird übersprungen.")
                
        if not cluster_data:
            print(f"Überspringe Cluster {cid}: Keine nutzbaren Daten gefunden.")
            continue
            
        print(f"Cluster {cid} erfolgreich geladen ({len(cluster_data)} Assets). Starte KI...")
        
        # Optimizer initialisieren und ausführen
        optimizer = ClusterOptimizer(data_dict=cluster_data, cluster_id=cid)
        
        try:
            best_sharpe, best_params = optimizer.run_optimization(n_trials=N_TRIALS_PER_CLUSTER)
            
            # Ergebnisse für die CSV speichern
            result_row = {'Cluster_ID': cid, 'Train_Sharpe': round(best_sharpe, 4)}
            result_row.update(best_params)
            final_results.append(result_row)
            
        except Exception as e:
            print(f"Fehler bei Cluster {cid}: {e}")

    # Ergebnisse in eine übersichtliche CSV-Datei exportieren
    if final_results:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n\n{'='*50}")
        print(f"MASTER-DURCHLAUF ABGESCHLOSSEN!")
        print(f"Alle Parameter wurden erfolgreich in '{OUTPUT_FILE}' gespeichert.")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
