# Datenbeschaffung: MT5 Data Engine

Dieses Modul ist Teil eines algorithmischen Trading-Bots und dient der automatisierten Beschaffung von historischen H1-Marktdaten (1-Stunde-Chart) über die MetaTrader 5 (MT5) API. Es lädt historische Daten für ein definiertes Portfolio von 100 Assets (Forex, Krypto, Aktien, Indizes, Rohstoffe) herunter und speichert diese als bereinigte CSV-Dateien für das nachgelagerte Backtesting.

## Requirements

Um dieses Skript auszuführen, müssen folgende Voraussetzungen erfüllt sein:

1. **Betriebssystem:** Windows (oder eine Windows-VM unter Linux/Mac), da die offizielle `MetaTrader5` Bibliothek von MetaQuotes zwingend Windows-Abhängigkeiten (`.dll`) benötigt.
2. **MetaTrader 5 Terminal:** Muss installiert, geöffnet und mit einem (Demo-)Konto verbunden sein (z.B. Pepperstone).
3. **Python:** Version 3.8 oder höher.

### Benötigte Python-Bibliotheken

Installiere die erforderlichen Pakete über das Terminal:

```bash
pip install MetaTrader5 pandas
```
## Verwendung

1. Öffne das MetaTrader 5 Terminal und logge dich in dein Konto ein.
2. Führe das Python-Skript `data_engine.py` aus.
3. Das Skript verbindet sich automatisch mit dem offenen MT5-Terminal.
4. Die H1-Daten der letzten 6 Jahre werden heruntergeladen und im automatisch erstellten Ordner `mt5_h1_daten` als saubere `.csv`-Dateien gespeichert.

## Python Code (data_engine.py)

Das Skript nutzt ein Fallback-System, da sich teilweise die Asset-Kürzel unterscheiden je nach Broker (z.B. AAPL.US vs. AAPL). Diese könnte man nun nach belieben auf die gewünschten Assets abändern/anpassen. Das Programm lässt sich starten mit `python3 data_engine.py`.

```bash
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

def initialize_mt5():
    """Initialisiert die Verbindung zum aktiven MetaTrader 5 Terminal."""
    print("Initialisiere MetaTrader 5...")
    if not mt5.initialize():
        print(f"Fehler bei der Initialisierung: {mt5.last_error()}")
        return False
    return True

def download_asset_data(symbol_list, years_back=6, output_folder="mt5_h1_daten"):
    """
    Lädt H1-Daten für ein Asset herunter und speichert sie als CSV.
    symbol_list: Liste möglicher Ticker-Namen für dasselbe Asset (Fallback-Logik).
    """
    for symbol in symbol_list:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            continue  # Name nicht gefunden im Broker-Terminal, probiere den nächsten Alias
            
        # Symbol im "Market Watch" sichtbar machen, falls nötig
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        utc_to = datetime.now()
        utc_from = utc_to - timedelta(days=365 * years_back)
        
        print(f"Lade H1-Daten für {symbol} herunter...")
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
        
        if rates is None or len(rates) == 0:
            print(f"Keine Daten für {symbol} gefunden.")
            return False
            
        # Daten aufbereiten
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Ordner erstellen, falls nicht vorhanden
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # CSV speichern (Dateiname wird basierend auf dem ersten Wunsch-Kürzel bereinigt)
        clean_name = symbol_list[0].replace(".US", "").replace(".US-24", "")
        file_path = os.path.join(output_folder, f"{clean_name}_H1.csv")
        df.to_csv(file_path)
        
        print(f"Erfolgreich gespeichert: {file_path}")
        return True
        
    print(f"Überspringe {symbol_list[0]}: Keiner der Alias-Namen existiert im MT5.")
    return False

if __name__ == "__main__":
    if initialize_mt5():
        
        # Format: [Wunschname, Fallback 1, Fallback 2, ...]
        assets_to_download = [
            # --- Indizes ---
            ["EUSTX50", "EU50"], ["US30"], ["NAS100"], ["US500"], ["US2000"], 
            ["JPN225"], ["VIX"], ["FRA40"], ["SPA35"], ["HK50"], ["UK100"], 
            ["GER40"], ["AUS200"],
            
            # --- Rohstoffe ---
            ["XTIUSD", "SpotCrude", "USOIL"], ["XBRUSD", "SpotBrent", "UKOIL"],
            ["XNGUSD", "NatGas", "NGAS"], ["XCUUSD", "Copper"], ["Corn"], 
            ["Wheat"], ["Sugar"], ["Coffee"], ["XAUUSD"], ["XAGUSD"], 
            ["XPTUSD"], ["XPDUSD"],
            
            # --- Kryptowährungen ---
            ["ATMUSD", "ATOM"], ["NERUSD", "NEAR"], ["UNIUSD"], ["XRPUSD"], 
            ["ADAUSD"], ["SOLUSD"], ["LTCUSD"], ["BCHUSD"], ["BTCUSD"], 
            ["ETHUSD"], ["BNBUSD"], ["DOTUSD"], ["LINKUSD"], ["AVAXUSD"], 
            ["DOGEUSD"],
            
            # --- US Aktien ---
            ["UNH.US", "UNH.US-24", "UNH"], ["LMT.US", "LMT.US-24", "LMT"], 
            ["NKE.US", "NKE.US-24", "NKE"], ["JPM.US", "JPM.US-24", "JPM"], 
            ["GS.US", "GS.US-24", "GS"], ["HD.US", "HD.US-24", "HD"], 
            ["V.US", "V.US-24", "V"], ["TSLA.US", "TSLA.US-24", "TSLA"], 
            ["AAPL.US", "AAPL.US-24", "AAPL"], ["MA.US", "MA.US-24", "MA"], 
            ["BAC.US", "BAC.US-24", "BAC"], ["BA.US", "BA.US-24", "BA"], 
            ["SBUX.US", "SBUX.US-24", "SBUX"], ["CAT.US", "CAT.US-24", "CAT"], 
            ["CVX.US", "CVX.US-24", "CVX"], ["AMD.US", "AMD.US-24", "AMD"], 
            ["QCOM.US", "QCOM.US-24", "QCOM"], ["DIS.US", "DIS.US-24", "DIS"], 
            ["META.US", "META.US-24", "META"], ["MSFT.US", "MSFT.US-24", "MSFT"], 
            ["AMZN.US", "AMZN.US-24", "AMZN"], ["GOOGL.US", "GOOGL.US-24", "GOOGL"], 
            ["NFLX.US", "NFLX.US-24", "NFLX"], ["PEP.US", "PEP.US-24", "PEP"], 
            ["PFE.US", "PFE.US-24", "PFE"], ["INTC.US", "INTC.US-24", "INTC"], 
            ["WMT.US", "WMT.US-24", "WMT"], ["JNJ.US", "JNJ.US-24", "JNJ"], 
            ["KO.US", "KO.US-24", "KO"], ["XOM.US", "XOM.US-24", "XOM"], 
            ["NVDA.US", "NVDA.US-24", "NVDA"], ["IBM.US", "IBM.US-24", "IBM"],

            # --- Forex ---
            ["USDCHF"], ["CADCHF"], ["AUDCHF"], ["GBPCHF"], ["NZDCHF"], 
            ["NZDUSD"], ["AUDUSD"], ["GBPCAD"], ["NZDCAD"], ["GBPUSD"], 
            ["EURUSD"], ["EURAUD"], ["EURGBP"], ["USDCAD"], ["EURNZD"], 
            ["EURCAD"], ["GBPNZD"], ["AUDNZD"], ["GBPAUD"], ["AUDCAD"], 
            ["AUDJPY"], ["EURCHF"], ["NZDJPY"], ["EURJPY"], ["CADJPY"], 
            ["CHFJPY"], ["GBPJPY"], ["USDJPY"]
        ]
        
        print(f"Starte Download für {len(assets_to_download)} Assets (Zeitraum: 6 Jahre)...")
        
        erfolgreich = 0
        for asset_aliases in assets_to_download:
            if download_asset_data(asset_aliases, years_back=6):
                erfolgreich += 1
            
        mt5.shutdown()
        print(f"\nDownload abgeschlossen! {erfolgreich} von {len(assets_to_download)} Assets wurden als CSV gespeichert.")
```
