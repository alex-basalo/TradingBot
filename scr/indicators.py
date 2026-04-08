import pandas as pd
import pandas_ta as ta
import numpy as np

class IndicatorBaukasten:
    """
    Modul 3: Der vollständige Indikatoren-Baukasten für den Trading-Bot.
    Beinhaltet alle wissenschaftlich recherchierten Werkzeuge für die Kategorien A, B und C.
    """

    @staticmethod
    def add_category_a_zones(df: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """Kategorie A: Zonen-Identifikation (S/R)"""
        df_ind = df.copy()
        
        if indicator_name == 'bollinger':
            length = kwargs.get('length', 20)
            std = kwargs.get('std', 2.0)
            bbands = ta.bbands(df_ind['close'], length=length, std=std)
            if bbands is not None:
                df_ind['zone_upper'] = bbands.iloc[:, 2]
                df_ind['zone_lower'] = bbands.iloc[:, 0]
                
        elif indicator_name == 'keltner':
            length = kwargs.get('length', 20)
            scalar = kwargs.get('scalar', 1.5)
            kc = ta.kc(df_ind['high'], df_ind['low'], df_ind['close'], length=length, scalar=scalar)
            if kc is not None:
                df_ind['zone_upper'] = kc.iloc[:, 2]
                df_ind['zone_lower'] = kc.iloc[:, 0]
                
        elif indicator_name == 'vwap':
            # VWAP inkl. Standardabweichungs-Bänder
            # Da keine Intraday-Ankerpunkte auf den H1-Daten vorliegen, wird ein rollierender VWAP genutzt
            length = kwargs.get('length', 24) # z.B. 24h VWAP
            std = kwargs.get('std', 1.5)
            
            typical_price = (df_ind['high'] + df_ind['low'] + df_ind['close']) / 3
            vp = typical_price * df_ind['volume']
            
            vwap_line = vp.rolling(window=length).sum() / df_ind['volume'].rolling(window=length).sum()
            vwap_std = typical_price.rolling(window=length).std()
            
            df_ind['zone_upper'] = vwap_line + (vwap_std * std)
            df_ind['zone_lower'] = vwap_line - (vwap_std * std)
            
        elif indicator_name == 'ichimoku':
            # Die Kumo-Wolke (Senkou Span A und B) als dynamische Zone
            tenkan = kwargs.get('tenkan', 9)
            kijun = kwargs.get('kijun', 26)
            senkou = kwargs.get('senkou', 52)
            
            ichi = ta.ichimoku(df_ind['high'], df_ind['low'], df_ind['close'], 
                               tenkan=tenkan, kijun=kijun, senkou=senkou)[0]
            if ichi is not None:
                # Span A und Span B bilden die Wolke. Ordnung erfolgt mathematisch nach Oben/Unten
                span_a = ichi[f'ISA_{tenkan}']
                span_b = ichi[f'ISB_{kijun}']
                df_ind['zone_upper'] = np.maximum(span_a, span_b)
                df_ind['zone_lower'] = np.minimum(span_a, span_b)
                
        elif indicator_name == 'pivot':
            # Rollierende Pivot Points (z.B. basierend auf den letzten 24 Stunden)
            length = kwargs.get('length', 24)
            roll_high = df_ind['high'].rolling(window=length).max().shift(1)
            roll_low = df_ind['low'].rolling(window=length).min().shift(1)
            roll_close = df_ind['close'].shift(1)
            
            pivot = (roll_high + roll_low + roll_close) / 3
            # R1 (Resistance) und S1 (Support)
            df_ind['zone_upper'] = (2 * pivot) - roll_low
            df_ind['zone_lower'] = (2 * pivot) - roll_high

        return df_ind

    @staticmethod
    def add_category_b_momentum(df: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """Kategorie B: Signal-Bestätigung (Momentum / Volumen)"""
        df_ind = df.copy()
        
        if indicator_name == 'rsi':
            length = kwargs.get('length', 14)
            df_ind['momentum_main'] = ta.rsi(df_ind['close'], length=length)
            
        elif indicator_name == 'adaptive_rsi':
            # Eigenentwicklung: Periodenlänge passt sich dynamisch an die ATR an
            min_len = kwargs.get('min_len', 5)
            max_len = kwargs.get('max_len', 25)
            
            # Volatilitäts-Niveau (Percentil der letzten 100 Kerzen)
            atr = df_ind['atr_14']
            atr_rank = atr.rolling(window=100).rank(pct=True)
            
            # Interpolation: Hohe Vola = längerer RSI (Glättung), niedrige Vola = kurzer RSI (reaktionsschnell)
            # Werte runden, um sie als Integer-Längen nutzen zu können
            dyn_lengths = np.round(min_len + (max_len - min_len) * atr_rank).fillna(min_len).astype(int)
            
            # Vektorisierter Trick: Alle RSI Längen vorberechnen und dann dynamisch picken
            rsi_matrix = {i: ta.rsi(df_ind['close'], length=i) for i in range(min_len, max_len + 1)}
            rsi_df = pd.DataFrame(rsi_matrix)
            
            # Für jede Zeile den RSI mit der spezifischen dyn_length auswählen
            idx = np.arange(len(df_ind))
            df_ind['momentum_main'] = rsi_df.values[idx, dyn_lengths.values - min_len]
            
        elif indicator_name == 'macd':
            fast = kwargs.get('fast', 12)
            slow = kwargs.get('slow', 26)
            signal = kwargs.get('signal', 9)
            macd = ta.macd(df_ind['close'], fast=fast, slow=slow, signal=signal)
            if macd is not None:
                # MACD Histogramm als primäres Momentum
                df_ind['momentum_main'] = macd.iloc[:, 1] 
                
        elif indicator_name == 'stoch':
            k = kwargs.get('k', 14)
            stoch = ta.stoch(df_ind['high'], df_ind['low'], df_ind['close'], k=k)
            if stoch is not None:
                df_ind['momentum_main'] = stoch.iloc[:, 0]
                
        elif indicator_name == 'mfi':
            length = kwargs.get('length', 14)
            df_ind['momentum_main'] = ta.mfi(df_ind['high'], df_ind['low'], df_ind['close'], df_ind['volume'], length=length)
            
        return df_ind

    @staticmethod
    def add_category_c_trend(df: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """Kategorie C: Trend-Filter (Optional)"""
        df_ind = df.copy()
        
        if indicator_name == 'ema_cross':
            fast = kwargs.get('fast', 50)
            slow = kwargs.get('slow', 200)
            ema_fast = ta.ema(df_ind['close'], length=fast)
            ema_slow = ta.ema(df_ind['close'], length=slow)
            df_ind['trend_direction'] = np.where(ema_fast > ema_slow, 1, -1)
            
        elif indicator_name == 'adx':
            length = kwargs.get('length', 14)
            adx = ta.adx(df_ind['high'], df_ind['low'], df_ind['close'], length=length)
            if adx is not None:
                df_ind['trend_direction'] = adx.iloc[:, 0]
                
        elif indicator_name == 'psar':
            af = kwargs.get('af', 0.02)
            max_af = kwargs.get('max_af', 0.2)
            psar = ta.psar(df_ind['high'], df_ind['low'], df_ind['close'], af=af, max_af=max_af)
            if psar is not None:
                # 1 wenn Preis über PSAR (Uptrend), -1 wenn darunter
                df_ind['trend_direction'] = np.where(df_ind['close'] > psar.iloc[:, 0], 1, -1)
                
        return df_ind
