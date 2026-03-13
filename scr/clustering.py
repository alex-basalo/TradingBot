import yfinance as yf
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score

class DatenVerarbeiter:
    """Verantwortlich für das Herunterladen der Daten und die Feature-Extraktion."""
    
    def __init__(self, ticker_liste: list[str], start_datum: str, end_datum: str):
        """
        Initialisiert den Verarbeiter mit den Tickersymbolen und dem Zeitraum.
        """
        self.ticker_liste = ticker_liste
        self.start_datum = start_datum
        self.end_datum = end_datum

    def daten_herunterladen(self) -> pd.DataFrame:
        """
        Lädt die historischen Schlusskurse für alle definierten Ticker herunter.
        Schließt Lücken (z.B. Wochenenden) durch Forward- und Backward-Fill.
        """
        print(f"Lade Daten für {len(self.ticker_liste)} Assets herunter...")
        daten = yf.download(self.ticker_liste, start=self.start_datum, end=self.end_datum, progress=False)
        
        if 'Close' in daten.columns:
            schlusskurse = daten['Close']
        else:
            schlusskurse = daten
            
        # Lücken füllen, um traditionelle Märkte (Wochenend-Pause) mit Krypto (24/7) anzugleichen
        schlusskurse = schlusskurse.ffill().bfill()
        return schlusskurse

    def features_extrahieren(self, schlusskurse: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Berechnet tägliche Renditen, Volatilität und Momentum und normalisiert diese.
        Gibt ein 3D-Numpy-Array (für tslearn) und eine Liste der gültigen Ticker zurück.
        """
        print("Extrahiere und normalisiere Features...")
        features_liste = []
        gueltige_ticker = []

        for ticker in self.ticker_liste:
            if ticker not in schlusskurse.columns:
                continue
                
            serie = schlusskurse[ticker]
            if serie.isna().all():
                continue

            # Finanzkennzahlen berechnen
            df_features = pd.DataFrame(index=schlusskurse.index)
            df_features['Rendite'] = serie.pct_change()
            df_features['Volatilitaet'] = df_features['Rendite'].rolling(window=20).std()
            df_features['Momentum'] = serie.pct_change(periods=20)

            # Entstandene NaN-Werte (durch rollierende Fenster) bereinigen
            df_features = df_features.bfill().ffill().fillna(0)

            # Daten normalisieren (Mittelwert=0, Varianz=1), wichtig für DTW
            skalierer = StandardScaler()
            skalierte_features = skalierer.fit_transform(df_features)
            
            features_liste.append(skalierte_features)
            gueltige_ticker.append(ticker)

        # Umwandlung in 3D-Array: (Anzahl_Assets, Zeitstempel, Anzahl_Features)
        X = np.array(features_liste)
        return X, gueltige_ticker


class ZeitreihenClusterer:
    """Führt das K-Means Clustering mit Dynamic Time Warping (DTW) durch und evaluiert es."""
    
    def __init__(self, min_k: int = 3, max_k: int = 15):
        """
        Legt die minimale und maximale Anzahl zu testender Cluster fest.
        """
        self.min_k = min_k
        self.max_k = max_k

    def optimale_cluster_finden(self, X: np.ndarray) -> tuple[int, TimeSeriesKMeans]:
        """
        Iteriert über verschiedene Werte für k und ermittelt das beste Modell
        anhand des höchsten Silhouette Scores.
        """
        bestes_k = self.min_k
        bester_score = -1.0
        bestes_modell = None
        
        print(f"\nEvaluiere optimale Cluster-Anzahl zwischen k={self.min_k} und k={self.max_k}...")
        
        for k in range(self.min_k, self.max_k + 1):
            # n_jobs=-1 nutzt alle Prozessorkerne für maximale Performance
            modell = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42, n_jobs=-1)
            labels = modell.fit_predict(X)
            
            # Silhouette Score berechnen (benötigt mindestens 2 verschiedene Cluster)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels, metric="dtw")
                print(f"k={k:2d} | Silhouette Score: {score:.4f}")
                
                if score > bester_score:
                    bester_score = score
                    bestes_k = k
                    bestes_modell = modell
            else:
                print(f"k={k:2d} | Keine eindeutigen Cluster gebildet.")
                
        print(f"\n--> Bester Parameter: k={bestes_k} mit einem Score von {bester_score:.4f}")
        return bestes_k, bestes_modell


class ErgebnisExportierer:
    """Verantwortlich für das Speichern der Clustering-Ergebnisse in verschiedene Dateiformate."""
    
    @staticmethod
    def speichern(ticker_liste: list[str], labels: np.ndarray, basis_dateiname: str = "cluster_ergebnisse"):
        """
        Speichert die Zuweisungen als CSV (für Excel/Thesis) und als JSON (für den Trading-Bot).
        """
        # Daten in ein Pandas DataFrame umwandeln
        df_ergebnisse = pd.DataFrame({
            'Asset': ticker_liste,
            'Cluster_ID': labels
        })
        
        # Nach Clustern sortieren, damit die Liste übersichtlich ist
        df_ergebnisse = df_ergebnisse.sort_values(by='Cluster_ID').reset_index(drop=True)
        
        # 1. Als CSV speichern
        csv_pfad = f"{basis_dateiname}.csv"
        df_ergebnisse.to_csv(csv_pfad, index=False)
        print(f"\nErfolgreich gespeichert: {csv_pfad}")
        
        # 2. Als JSON speichern (Dictionary: Cluster_ID -> Liste von Assets)
        cluster_dict = {}
        for cluster_id, gruppe in df_ergebnisse.groupby('Cluster_ID'):
            # Konvertierung cluster_id zu int, da JSON keine numpy-Datentypen akzeptiert
            cluster_dict[f"Cluster_{int(cluster_id)}"] = gruppe['Asset'].tolist()
            
        json_pfad = f"{basis_dateiname}.json"
        with open(json_pfad, 'w', encoding='utf-8') as f:
            json.dump(cluster_dict, f, indent=4)
        print(f"Erfolgreich gespeichert: {json_pfad}")

class ClusterAnalyst:
    """
    Übersetzt die mathematischen Cluster-Zentren wieder in reale, interpretierbare 
    Kennzahlen (Volatilität, Momentum, Rendite), um die Zuweisung zu begründen.
    """
    
    @staticmethod
    def begruendung_extrahieren(schlusskurse: pd.DataFrame, ticker_liste: list[str], labels: np.ndarray):
        print("\n--- BEGRÜNDUNG DER CLUSTER (Echte Feature-Werte) ---")
        
        statistiken = []
        
        for ticker, label in zip(ticker_liste, labels):
            serie = schlusskurse[ticker].dropna()
            if len(serie) < 20:
                continue
                
            # Exakte Berechnung der Features, die auch dem Algorithmus zugeführt wurden
            rendite = serie.pct_change()
            vola = rendite.rolling(window=20).std()
            momentum = serie.pct_change(periods=20)
            
            # Durchschnitt dieser Features über den gesamten Zeitraum
            statistiken.append({
                'Asset': ticker,
                'Cluster_ID': label,
                'Ø_Tagesrendite': rendite.mean(),
                'Ø_Volatilitaet': vola.mean(),
                'Ø_Momentum': momentum.mean()
            })
            
        df_stats = pd.DataFrame(statistiken)
        
        # Durchschnitt der Features pro Cluster berechnen (lesbares "Cluster-Zentrum")
        cluster_begruendung = df_stats.groupby('Cluster_ID').agg({
            'Ø_Volatilitaet': 'mean',
            'Ø_Momentum': 'mean',
            'Ø_Tagesrendite': 'mean',
            'Asset': 'count'
        }).rename(columns={'Asset': 'Anzahl_Assets'})
        
        # Formatierung in Prozent für deine Arbeit
        for col in ['Ø_Volatilitaet', 'Ø_Momentum', 'Ø_Tagesrendite']:
            cluster_begruendung[col] = (cluster_begruendung[col] * 100).round(3).astype(str) + '%'
            
        print(cluster_begruendung.to_string())

if __name__ == "__main__":
    TICKER_LISTE = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
        "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
        "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X",
        "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
        "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC",
        "JPM", "V", "MA", "BAC", "GS", "WMT", "HD", "IBM", "NKE", "KO", "PEP", "SBUX",
        "JNJ", "UNH", "PFE", "BA", "CAT", "LMT", "XOM", "CVX", "DIS", "QCOM",  
        "^GSPC", "^NDX", "^DJI", "^RUT", "^GDAXI", "^FTSE", "^FCHI", "^STOXX50E", "^IBEX", 
        "^N225", "^AXJO", "^HSI", "^VIX",
        "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "KC=F", "SB=F",
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOT-USD", "AVAX-USD", 
        "LINK-USD", "UNI-USD", "LTC-USD", "BCH-USD", "ATOM-USD", "DOGE-USD", "NEAR-USD"
    ]

    START_DATUM = "2023-01-01"
    END_DATUM = "2026-01-01"

    # 1. Datenaufbereitung
    verarbeiter = DatenVerarbeiter(ticker_liste=TICKER_LISTE, start_datum=START_DATUM, end_datum=END_DATUM)
    preis_daten = verarbeiter.daten_herunterladen()
    X_daten, validierte_assets = verarbeiter.features_extrahieren(preis_daten)
    
    print(f"Datenstruktur bereit für Clustering: {X_daten.shape} (Assets, Tage, Features)")

    # 2. Modell-Evaluierung und Clustering
    clusterer = ZeitreihenClusterer(min_k=3, max_k=12) # max_k anpassbar
    optimales_k, finales_modell = clusterer.optimale_cluster_finden(X_daten)

    # 3. Ergebnisse exportieren und Evidenz generieren
    if finales_modell is not None:
        finale_labels = finales_modell.predict(X_daten)
        
        # Speichere die rohen Zuweisungen für den Bot
        ErgebnisExportierer.speichern(validierte_assets, finale_labels, "bot_cluster_config")
        
        #Generiere die datengestützte Evidenz für deine Thesis
        ClusterAnalyst.begruendung_extrahieren(preis_daten, validierte_assets, finale_labels)
