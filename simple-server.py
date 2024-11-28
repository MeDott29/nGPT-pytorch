import asyncio
import json
import time
import random
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SandplotServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.active_connections = set()

    def generate_sample_data(self):
        """Generate sample visualization data"""
        return {
            "text_hash": f"hash_{random.randint(1000, 9999)}",
            "dataset": "sample_dataset",
            "timestamp": time.time(),
            "params": {
                "density": random.uniform(0.25, 0.75),
                "point_size": random.randint(1, 3),
                "prime_factors": [13, 17, 19],
                "color_variation": 0.1
            }
        }

    async def handle_client(self, websocket):
        """Handle individual WebSocket client connections"""
        try:
            self.active_connections.add(websocket)
            logger.info(f"New client connected. Total connections: {len(self.active_connections)}")
            
            while True:
                # Generate and send new data every 2 seconds
                data = self.generate_sample_data()
                await websocket.send(json.dumps(data))
                await asyncio.sleep(2)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.active_connections.remove(websocket)

    async def start_server(self):
        """Start the WebSocket server"""
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

async def main():
    """Main entry point"""
    server = SandplotServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main())