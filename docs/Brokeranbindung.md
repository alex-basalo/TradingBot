# Broker-Anbindung via cTrader Open API

## 1. Systemvoraussetzungen und Registrierung
Um die Schnittstelle zu reproduzieren, müssen folgende asynchrone Zugangsparameter vorliegen:
1. **cTrader ID (cTID)**: Verknüpft mit einem CFD-Broker (z. B. Pepperstone Demo-Konto).
2. **Open API Applikation**: Erstellt im Spotware Developer Portal (`openapi.ctrader.com`) im Status "Active".
3. **Credentials**: `Client ID` und `Client Secret` der Applikation.
4. **Access Token**: Generiert über den API-Playground für das spezifische Zielkonto.
5. **cTID Trader Account ID**: Die interne API-Kontonummer (nicht identisch mit dem Broker-Login).

## 2. Lokale Umgebung (Ubuntu / Linux)
Aufgrund systemweiter Restriktionen für Python-Pakete (PEP 668) unter modernen Linux-Distributionen wird die Ausführung in einer virtuellen Umgebung (`venv`) zwingend empfohlen.

```bash
# Erstellung und Aktivierung der virtuellen Umgebung
python3 -m venv .venv
source .venv/bin/activate
# Installation der benötigten Abhängigkeiten
pip install ctrader-open-api service_identity
```

## 3. Implementierung (Python)
Der folgende Code demonstriert den Verbindungsaufbau, die Authentifizierungskette (Applikation -> Konto) und den Abruf von Kontodaten in einer objektorientierten Struktur, basierend auf dem asynchronen twisted-Framework.
Speichere die Datei als `broker_connection.py`.

```bash
import logging
from ctrader_open_api import Client, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from twisted.internet import reactor

# Logging-Konfiguration für professionelles Output-Management
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CTraderConnection:
    """
    Verwaltet die asynchrone Verbindung und Authentifizierung zur cTrader Open API.
    """

    def __init__(self, client_id: str, client_secret: str, access_token: str, account_id: int):
        """
        Initialisiert die Verbindungsdaten und den cTrader-Client.

        Args:
            client_id (str): Die Client ID der Spotware Applikation.
            client_secret (str): Das Client Secret der Spotware Applikation.
            access_token (str): Der OAUTH Access Token für das spezifische Konto.
            account_id (int): Die interne cTID Trader Account ID.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        
        # Initialisierung des Clients auf den Demo-Servern
        self.client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)

    def start(self):
        """Startet den Twisted-Reactor und initiiert den Verbindungsaufbau."""
        logger.info("Starte cTrader API Service...")
        self.client.startService()
        reactor.run()

    def _on_error(self, failure):
        """Callback für kritische Verbindungs- oder Laufzeitfehler."""
        logger.error(f"Kritischer Fehler aufgetreten: {failure}")
        reactor.stop()

    def _on_connected(self, client):
        """Callback nach erfolgreichem TCP-Verbindungsaufbau zum Server."""
        logger.info("TCP-Verbindung hergestellt. Sende App-Authentifizierung...")
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        
        deferred = client.send(req)
        deferred.addCallback(self._on_app_authenticated, client)
        deferred.addErrback(self._on_error)

    def _on_app_authenticated(self, result, client):
        """Callback nach erfolgreicher Authentifizierung der Applikation."""
        logger.info("Applikation verifiziert. Sende Konto-Authentifizierung...")
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        
        deferred = client.send(req)
        deferred.addCallback(self._on_account_authenticated, client)
        deferred.addErrback(self._on_error)

    def _on_account_authenticated(self, result, client):
        """Callback nach erfolgreicher Authentifizierung des Zielkontos."""
        if result.payloadType == 2142:
            error = ProtoOAErrorRes()
            error.ParseFromString(result.payload)
            logger.error(f"Authentifizierung fehlgeschlagen: {error.errorCode} - {error.description}")
            reactor.stop()
            return

        logger.info(f"Konto {self.account_id} erfolgreich authentifiziert. Rufe Kontodaten ab...")
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        
        deferred = client.send(req)
        deferred.addCallback(self._on_trader_received, client)
        deferred.addErrback(self._on_error)

    def _on_trader_received(self, result, client):
        """Callback zur Verarbeitung der asynchronen Kontodaten-Antwort."""
        if result.payloadType == 2122:
            response = ProtoOATraderRes()
            response.ParseFromString(result.payload)
            # Umrechnung von der kleinsten Einheit (z. B. Cent) in die Basiswährung
            balance = response.trader.balance / 100.0
            logger.info(f"Verbindungstest abgeschlossen. Virtuelles Guthaben: {balance:.2f}")
        else:
            logger.warning(f"Unerwarteter Payload empfangen (Typ: {result.payloadType}).")
            
        logger.info("Beende asynchronen Event-Loop.")
        reactor.stop()

    def _on_disconnected(self, client, reason):
        """Callback bei Trennung der TCP-Verbindung."""
        logger.info("Verbindung zum Server wurde getrennt.")


if __name__ == "__main__":
    # Konfiguration der Zugangsparameter
    CLIENT_ID = "DEINE_CLIENT_ID"
    CLIENT_SECRET = "DEIN_CLIENT_SECRET"
    ACCESS_TOKEN = "DEIN_ACCESS_TOKEN"
    ACCOUNT_ID = 12345678  # Als Integer übergeben

    # Instanziierung und Start des Broker-Clients
    bot_connection = CTraderConnection(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        access_token=ACCESS_TOKEN,
        account_id=ACCOUNT_ID
    )
    bot_connection.start()
```

## 4. Ausführung und Test

Um die Pipeline zu prüfen, wird das Skript innerhalb der aktivierten virtuellen Umgebung gestartet:
```Bash

python3 broker_connection.py
```
