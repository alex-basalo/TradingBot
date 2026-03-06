# **Hinweise für Implementierung aus Literaturrecherche**

1. *Bot-Architektur*: Time-Driven (Schleife) vs. Event-Driven (Websocket). Entscheidung u.a. abhängig von Broker
2. *Backtesting*: Trading-View Strategietester vs. direkte Backtesting-Implementierung in python
3. *Datenbeschaffung*: "open-crypto"-Bibliothek und "yfinance"-Bibliothek für historische Daten; "tslearn", "scikit-learn" und "pandas" für Algorithmus 
4. *Transaktionskosten*: Backtesting muss Fees, Slippage, Spreads berücksichtigen
5. *Deployment*: Docker-Container auf VPS
6. *Risikomanagement*: Zentrale Fragen sind 1. Positionsgrößenberechnung (Kelly-Kriterium, 1% Risk, dynamisch), 2. Stop-loss/Take-profit-Berechnung (dynamisch, ATR-basiert, fix), 3. ggf. Asset-Gewichtung

---

**Quellen**

*Für 1.* 

https://github.com/ccxt/ccxt

https://github.com/freqtrade/freqtrade

https://openapi.ctrader.com/

*Für 2.*

Jensen, Theis Ingerslev and Kelly, Bryan T. and Malamud, Semyon and Pedersen, Lasse Heje, Machine Learning and the Implementable Efficient Frontier (June 19, 2024). Swiss Finance Institute Research Paper No. 22-63, Available at SSRN: https://ssrn.com/abstract=4187217 or http://dx.doi.org/10.2139/ssrn.4187217 

https://github.com/polakowo/vectorbt

https://github.com/mementum/backtrader

*Für 3.*

The Python Package open-crypto: A Cryptocurrency Data
Collector
Steffen Günther, Christian Fieberg, and Thorsten Poddig

https://github.com/ranaroussi/yfinance

https://github.com/tslearn-team/tslearn

https://scikit-learn.org/stable/modules/clustering.html

*Für 4.*

Jensen, Theis Ingerslev and Kelly, Bryan T. and Malamud, Semyon and Pedersen, Lasse Heje, Machine Learning and the Implementable Efficient Frontier (June 19, 2024). Swiss Finance Institute Research Paper No. 22-63, Available at SSRN: https://ssrn.com/abstract=4187217 or http://dx.doi.org/10.2139/ssrn.4187217 

*Für 5.*

https://github.com/freqtrade/freqtrade/tree/develop/docker

*Für 6.*

Parametric Portfolio Policies: Exploiting Characteristics in the Cross Section of Equity Returns
Michael W. Brandt, Pedro Santa-Clara, Rossen Valkanov

R. Setiawan and M. S. Hakim, "Diversified Crypto Assets Portfolio Optimization Using K-Means Clustering Algorithm And The Efficient Frontier," 2023 IEEE Technology & Engineering Management Conference - Asia Pacific (TEMSCON-ASPAC), Bengaluru, India, 2023, pp. 1-6, doi: 10.1109/TEMSCON-ASPAC59527.2023.10531468.

https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas

