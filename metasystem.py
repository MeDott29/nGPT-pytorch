import asyncio
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import datasets
import websockets

@dataclass
class DatasetConfig:
    """Configuration for available datasets and their characteristics."""
    name: str                    # Dataset name in Hugging Face
    subset: str                  # Subset or version of the dataset
    text_field: str             # Field containing the main text content
    estimated_avg_size: int     # Average bytes per entry (for optimization)

class DataFetcher:
    """Handles the fetching and processing of data from Hugging Face datasets."""
    
    def __init__(self):
        # Configure available datasets with their specifications
        self.available_datasets = [
            DatasetConfig("wikipedia", "20220301.en", "text", 2000),
            DatasetConfig("code_search_net", "python", "content", 1500),
            DatasetConfig("github_issues", "all", "body", 1000),
            DatasetConfig("oscar", "unshuffled_deduplicated_en", "text", 3000)
        ]
        self.dataset_cache = {}

    async def get_random_chunk(self, target_size_mb: float = 1.0) -> Tuple[str, DatasetConfig]:
        """
        Fetches a random chunk of text data of approximately target_size_mb megabytes.
        Returns both the text and the source dataset configuration.
        """
        # Select a random dataset configuration
        dataset_config = random.choice(self.available_datasets)
        
        try:
            # Calculate how many samples we need based on average size
            target_bytes = int(target_size_mb * 1024 * 1024)
            estimated_samples = target_bytes // dataset_config.estimated_avg_size
            
            # Load dataset with streaming for memory efficiency
            dataset = datasets.load_dataset(
                dataset_config.name,
                dataset_config.subset,
                streaming=True
            )
            
            # Collect text chunks until we reach target size
            text_chunks = []
            current_size = 0
            
            for item in dataset["train"].take(estimated_samples * 2):  # Extra samples for safety
                text = item.get(dataset_config.text_field, "")
                if text:
                    text_chunks.append(text)
                    current_size += len(text.encode('utf-8'))
                    
                    if current_size >= target_bytes:
                        break
            
            return " ".join(text_chunks), dataset_config
            
        except Exception as e:
            print(f"Error fetching from {dataset_config.name}: {e}")
            return f"Error fetching dataset {dataset_config.name}", dataset_config

@dataclass
class SandplotParameters:
    """Parameters for generating a sandplot visualization."""
    density: float              # Point density (0.0 to 1.0)
    point_size: int            # Size of each point in pixels
    prime_factors: List[int]   # Prime numbers for color generation
    color_variation: float     # Random variation in colors (0.0 to 1.0)

@dataclass
class ChainMessage:
    """Represents a message in our blockchain-inspired system."""
    text_hash: str                    # SHA-256 hash of the text content
    dataset_info: DatasetConfig       # Source dataset information
    timestamp: float                  # Unix timestamp of creation
    sandplot_params: SandplotParameters  # Parameters for visualization
    previous_hash: str                # Hash of the previous message
    message_signature: Optional[str]   # For future cryptographic verification

class MetaSystemDatabase:
    """Handles all database operations for the meta-system."""
    
    def __init__(self, db_path: str = "metasystem.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_tables()
        
    def setup_tables(self):
        """Creates the necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_subset TEXT NOT NULL,
                timestamp REAL NOT NULL,
                previous_hash TEXT NOT NULL,
                sandplot_params TEXT NOT NULL,
                signature TEXT
            )
        ''')
        
        # Table for tracking data processing statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                timestamp REAL PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                bytes_processed INTEGER NOT NULL,
                processing_time REAL NOT NULL
            )
        ''')
        
        self.conn.commit()

class MetaSandplotSystem:
    """Main system that coordinates data fetching, processing, and visualization."""
    
    def __init__(self):
        self.db = MetaSystemDatabase()
        self.data_fetcher = DataFetcher()
        self.active_connections = set()
        
    def generate_sandplot_params(self, text: str) -> SandplotParameters:
        """
        Generates visualization parameters based on text characteristics.
        Uses the text's hash to ensure consistent parameters for the same text.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        return SandplotParameters(
            density=(int(text_hash[:2], 16) % 50 + 25) / 100,  # 25-75%
            point_size=(int(text_hash[2:4], 16) % 3) + 1,      # 1-3 pixels
            prime_factors=[13, 17, 19],                        # For color generation
            color_variation=0.1                                # 10% color variation
        )

    async def process_new_data(self) -> Optional[ChainMessage]:
        """Fetches and processes new data, creating a chain message."""
        # Fetch random data chunk
        text, dataset_config = await self.data_fetcher.get_random_chunk()
        
        # Generate text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Get previous message hash
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT hash FROM messages ORDER BY timestamp DESC LIMIT 1")
        previous_hash = cursor.fetchone()
        previous_hash = previous_hash[0] if previous_hash else "0" * 64
        
        # Create new chain message
        message = ChainMessage(
            text_hash=text_hash,
            dataset_info=dataset_config,
            timestamp=time.time(),
            sandplot_params=self.generate_sandplot_params(text),
            previous_hash=previous_hash,
            message_signature=None
        )
        
        # Store message in database
        self._store_message(message, text)
        
        return message

    def _store_message(self, message: ChainMessage, text: str):
        """Stores a message and its associated data in the database."""
        cursor = self.db.conn.cursor()
        cursor.execute('''
            INSERT INTO messages (
                hash, content, dataset_name, dataset_subset, 
                timestamp, previous_hash, sandplot_params, signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.text_hash,
            text,
            message.dataset_info.name,
            message.dataset_info.subset,
            message.timestamp,
            message.previous_hash,
            json.dumps(asdict(message.sandplot_params)),
            message.message_signature
        ))
        self.db.conn.commit()

    async def run_server(self, host: str = "localhost", port: int = 8765):
        """Runs the WebSocket server for real-time client updates."""
        async def client_handler(websocket, path):
            """Handles individual client connections."""
            self.active_connections.add(websocket)
            try:
                while True:
                    message = await self.process_new_data()
                    if message:
                        # Convert message to JSON-friendly format
                        message_data = {
                            "text_hash": message.text_hash,
                            "dataset": message.dataset_info.name,
                            "timestamp": message.timestamp,
                            "params": asdict(message.sandplot_params)
                        }
                        await websocket.send(json.dumps(message_data))
                    await asyncio.sleep(10)  # Wait between updates
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.active_connections.remove(websocket)

        async with websockets.serve(client_handler, host, port):
            print(f"MetaSandplot server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever

async def main():
    """Main entry point for the MetaSandplot system."""
    system = MetaSandplotSystem()
    try:
        await system.run_server()
    except KeyboardInterrupt:
        print("\nShutting down MetaSandplot system...")

if __name__ == "__main__":
    asyncio.run(main())