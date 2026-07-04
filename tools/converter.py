"""
Datenaufbereitung: HistData-M1-Rohdaten -> H1-CSVs.

Durchsucht den Eingabeordner (inkl. Unterordner) nach HistData-M1-Dateien
(Excel/CSV/TXT), gruppiert sie automatisch nach Währungspaar, fügt alle
Jahre zusammen, aggregiert auf Stundenkerzen (OHLC + Volumen) und schreibt je
Paar eine Datei `{PAAR}_H1.csv` in den Ausgabeordner.

Die Ausgabe ist exakt auf das Einleseformat von `engine.load_all_data`
abgestimmt: Index-Spalte `time`, Spalten `open, high, low, close, volume`,
Komma-getrennt.

Aufruf:  python tools/converter.py
Pfade unten unter INPUT_FOLDER / OUTPUT_FOLDER anpassen.
"""

import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- Pfade (relativ zum Projekt-Wurzelverzeichnis; bei Bedarf anpassen) ---
INPUT_FOLDER = "HistData_M1"    # enthält die M1-Rohdaten (auch in Unterordnern)
OUTPUT_FOLDER = "HistData_H1"   # Zielordner; identisch zu CONFIG["RAW_DATA_DIR"]


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

    # Spaltenstruktur fuer HistData
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
                    continue  # Ueberspringt Dateien, die weder Excel noch CSV sind

                df_list.append(df)
                print(f"  -> {os.path.basename(file)} eingelesen.")
            except Exception as e:
                print(f"Fehler beim Lesen von {file}: {e}")

        if not df_list:
            continue

        # 4. Alle Jahre des Währungspaares zusammenfügen (egal ob Excel oder CSV)
        combined_df = pd.concat(df_list)

        # 5. Datum umwandeln und als Index setzen
        # 'errors="coerce"' wandelt Schrott-Zeilen (z.B. Copyright-Texte) in NaT um
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], format='%Y%m%d %H%M%S', errors='coerce')

        # Alle Zeilen ohne gültiges Datum löschen (wirft Textzeilen raus)
        combined_df.dropna(subset=['datetime'], inplace=True)

        # Doppelte Zeitstempel entfernen und chronologisch sortieren
        combined_df = combined_df[~combined_df['datetime'].duplicated(keep='first')]
        combined_df.set_index('datetime', inplace=True)
        combined_df.sort_index(inplace=True)

        # 6. Auf H1 resampeln und leere Phasen (Wochenenden) löschen
        df_h1 = combined_df.resample('1h').agg(aggregation_dict)
        df_h1.dropna(inplace=True)

        # 7. Speichern als CSV (Einleseformat von engine.load_all_data)
        output_file = os.path.join(output_folder, f"{pair}_H1_merged.csv")
        df_h1.index.name = 'time'
        df_h1.to_csv(output_file, index=True, header=True,
                     date_format='%Y-%m-%d %H:%M:%S', sep=',')
        print(f"  => gespeichert: {output_file}  ({len(df_h1)} Stundenkerzen)")


if __name__ == "__main__":
    process_all_pairs(INPUT_FOLDER, OUTPUT_FOLDER)
