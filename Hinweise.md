**Hinweise für Implementierung aus Literaturrecherche**

1. *Bot-Architektur*: Time-Driven (Schleife) vs. Event-Driven (Websocket). Entscheidung u.a. abhängig von Broker
2. *Backtesting*: Trading-View Strategietester vs. direkte Backtesting-Implementierung in python
3. *Datenbeschaffung*: "open-crypto"-Bibliothek und "yfinance"-Bibliothek für historische Daten; "tslearn", "scikit-learn" und "pandas" für Algorithmus 
4. *Transaktionskosten*: Backtesting muss Fees, Slippage, Spreads berücksichtigen
5. *Deployment*: Docker-Container auf VPS
6. *Risikomanagement*: Zentrale Fragen sind 1. Positionsgrößenberechnung (Kelly-Kriterium, 1% Risk, dynamisch), 2. Stop-loss/Take-profit-Berechnung (dynamisch, ATR-basiert, fix), 3. ggf. Asset-Gewichtung
