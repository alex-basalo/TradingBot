# Kriterien zur Asset-Auswahl für den Trading-Bot

Für die Entwicklung des algorithmischen Trading-Bots wurde ein diversifiziertes Universum von 100 Finanzwerten aus fünf Anlageklassen gewählt. Die Auswahl erfolgte nicht zufällig, sondern strikt nach Kriterien der maximalen **Liquidität, Marktkapitalisierung und globalen Repräsentativität**, um eine reibungslose Orderausführung (minimaler Slippage) und verlässliche Support/Resistance-Zonen zu gewährleisten.

| Anlageklasse | Auswahlkriterium | Referenz |
| :--- | :--- | :--- |
| **Forex** | Auswahl der "Majors" und "Crosses" basierend auf dem global höchsten täglichen Handelsvolumen. Hochliquide Währungen garantieren die geringsten Spreads beim Broker (z. B. Pepperstone). | [BIS Triennial Central Bank Survey](https://www.bis.org/statistics/rpfx22.htm) |
| **Aktien** | Fokussierung auf die Unternehmen mit der weltweit höchsten Marktkapitalisierung (Market Cap). Diese Mega-Caps (z. B. Apple, Microsoft) weisen eine enorme Liquidität auf und sind kaum anfällig für Preismanipulationen. | [CompaniesMarketCap](https://companiesmarketcap.com/) |
| **Kryptowährungen** | Auswahl strikt nach höchster Marktkapitalisierung und 24-Stunden-Handelsvolumen. Illiquide Tokens (Altcoins) wurden ausgeschlossen, da deren Preis-Action zu erratisch für verlässliche Zonen-Strategien ist. | [CoinMarketCap](https://coinmarketcap.com/) |
| **Rohstoffe** | Auswahl der volumenstärksten Rohstoff-Futures (Gold, Öl, Kupfer) nach ihrer weltwirtschaftlichen Bedeutung. Diese Märkte bieten durch institutionelle Marktteilnehmer stabile technische Muster. | [CME Group Commodity Markets](https://www.cmegroup.com/markets/commodities.html) |
| **Indizes** | Geografische Diversifikation zur Abbildung des weltweiten Marktgeschehens. Ausgewählt wurden die Leitindizes der größten Volkswirtschaften (USA, Europa, Asien), die als Derivate extrem liquide handelbar sind. | [TradingView Major Indices](https://www.tradingview.com/markets/indices/quotes-major/) |
