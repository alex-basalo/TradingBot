import pandas as pd
import os
from collections import defaultdict
from pathlib import Path

def process_all_pairs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Suche nach Excel- UND CSV/TXT-Dateien in allen Unterordnern
    p = Path(input_folder)
    gefunden_dateien = (
        list(p.rglob("*.xlsx")) + list(p.rglob("*.XLSX")) +
        list(p.rglob("*.csv")) + list(p.rglob("*.CSV")) +
        list(p.rglob("*.txt")) + list(p.rglob("*.TXT"))
    )
    
    all_files = [str(datei) for datei in gefunden_dateien]
    
    if not all_files:
        print(f"Keine passenden Dateien im Ordner {input_folder} gefunden!")
        return

    print(f"{len(all_files)} Dateien gefunden (Excel & CSV). Starte Analyse...")

    # 2. Dateien automatisch nach Währungspaar gruppieren
    files_by_pair = defaultdict(list)
    for file in all_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        # Sucht nach dem 6-stelligen Währungspaar im Dateinamen (z.B. EURUSD)
        pair = next((part for part in parts if len(part) == 6 and part.isalpha()), "UNKNOWN")
        files_by_pair[pair].append(file)

    # Spaltenstruktur für HistData
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    aggregation_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    # 3. Jedes Paar einzeln abarbeiten
    for pair, files in files_by_pair.items():
        if pair == "UNKNOWN":
            continue
            
        print(f"\nBaue {pair} aus {len(files)} Dateien zusammen...")
        df_list = []
        
        for file in files:
            try:
                # Automatische Erkennung des Dateiformats anhand der Endung
                if file.lower().endswith('.xlsx'):
                    df = pd.read_excel(file, header=None, names=column_names, engine='openpyxl')
                elif file.lower().endswith('.csv') or file.lower().endswith('.txt'):
                    # HistData "Generic ASCII" nutzt meistens das Semikolon als Trenner
                    df = pd.read_csv(file, header=None, names=column_names, sep=';', engine='python')
                else:
                    continue # Überspringt Dateien, die weder Excel noch CSV sind
                
                df_list.append(df)
                print(f"  -> {os.path.basename(file)} eingelesen.")
            except Exception as e:
                print(f"Fehler beim Lesen von {file}: {e}")
        
        if not df_list:
            continue
            
        # 4. Alle Jahre des Währungspaares zusammenfügen (egal ob Excel oder CSV)
        combined_df = pd.concat(df_list)
        
        # 5. Datum umwandeln und als Index setzen
        # 'errors="coerce"' wandelt Schrott-Zeilen (wie Copyright-Texte) automatisch in NaT (Not a Time) um
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], format='%Y%m%d %H%M%S', errors='coerce')
        
        # Alle Zeilen löschen, die kein gültiges Datum haben (wirft die Textdateien in den Müll)
        combined_df.dropna(subset=['datetime'], inplace=True)
        
        # Jetzt sicher den Index setzen
        combined_df.set_index('datetime', inplace=True)
        
        # 6. Auf H1 (1 Stunde) resampeln und leere Phasen (Wochenenden) löschen
        df_h1 = combined_df.resample('1h').agg(aggregation_dict)
        df_h1.dropna(inplace=True)
        
        # 7. Speichern als CSV (exakt passend für wft_claude_optimization.py)
        # Dynamischer Dateiname, da der Zeitraum jetzt variieren kann
        output_file = os.path.join(output_folder, f"{pair}_H1_merged.csv")
        
        # Den Index (die erste Spalte) in 'time' umbenennen, damit die Kopfzeile stimmt
        df_h1.index.name = 'time'
        
        # Header=True (Kopfzeile an), sep=',' (Komma statt Semikolon), neues Datumsformat
        df_h1.to_csv(output_file, index=True, header=True, date_format='%Y-%m-%d %H:%M:%S', sep=',')

if __name__ == "__main__":
    # Pfade definieren (Tilde '~' wird automatisch aufgelöst)
    ordner_mit_allen_m1_daten = os.path.expanduser("~/Projekte/TradingBot/Daten/HistData_M1_(3)")
    ordner_fuer_fertige_h1_daten = os.path.expanduser("~/Projekte/TradingBot/Daten/HistData_H1_(3)")
    
    process_all_pairs(ordner_mit_allen_m1_daten, ordner_fuer_fertige_h1_daten)
