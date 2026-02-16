from channels.generic.websocket import AsyncJsonWebsocketConsumer

class WalletConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send_json({"event": "connected"})
