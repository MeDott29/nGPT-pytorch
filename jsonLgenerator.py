import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import aiofiles
import datasets
import websockets

@dataclass
class DatasetConfig:
    """Configuration for available datasets and their characteristics."""
    name: str                    # Dataset name in Hugging Face
    subset: str                  # Subset or version of the dataset
    text_field: str             # Field containing the main text content
    estimated_avg_size: int     # Average bytes per entry (for optimization)

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
    text_hash: str                     # SHA-256 hash of the text content
    text_content: str                  # The actual text content
    dataset_name: str                  # Name of source dataset
    dataset_subset: str                # Subset of source dataset
    timestamp: float                   # Unix timestamp of creation
    previous_hash: str                 # Hash of the previous message
    sandplot_params: SandplotParameters  # Parameters for visualization
    message_signature: Optional[str]    # For future cryptographic verification

class JSONLDatabase:
    """Handles all database operations using JSONL files."""
    
    def __init__(self, data_dir: str = "metasystem_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.messages_file = self.data_dir / "messages.jsonl"
        self.stats_file = self.data_dir / "statistics.jsonl"
        self.cache_size = 1000  # Number of recent messages to keep in memory
        self.message_cache = []
        self._initialize_files()

    def _initialize_files(self):
        """Creates data files if they don't exist."""
        for file in [self.messages_file, self.stats_file]:
            if not file.exists():
                file.touch()

    async def append_message(self, message: ChainMessage):
        """Appends a new message to the JSONL file."""
        async with aiofiles.open(self.messages_file, mode='a') as f:
            # Convert message to dictionary, ensuring all types are JSON serializable
            message_dict = {
                'text_hash': message.text_hash,
                'text_content': message.text_content,
                'dataset_name': message.dataset_name,
                'dataset_subset': message.dataset_subset,
                'timestamp': message.timestamp,
                'previous_hash': message.previous_hash,
                'sandplot_params': asdict(message.sandplot_params),
                'message_signature': message.message_signature
            }
            await f.write(json.dumps(message_dict) + '\n')
        
        # Update cache
        self.message_cache.append(message)
        if len(self.message_cache) > self.cache_size:
            self.message_cache.pop(0)

    async def get_latest_message(self) -> Optional[ChainMessage]:
        """Returns the most recent message from the chain."""
        if self.message_cache:
            return self.message_cache[-1]
        
        try:
            async with aiofiles.open(self.messages_file, mode='r') as f:
                last_line = ''
                async for line in f:
                    if line.strip():
                        last_line = line
                
                if last_line:
                    data = json.loads(last_line)
                    return ChainMessage(
                        text_hash=data['text_hash'],
                        text_content=data['text_content'],
                        dataset_name=data['dataset_name'],
                        dataset_subset=data['dataset_subset'],
                        timestamp=data['timestamp'],
                        previous_hash=data['previous_hash'],
                        sandplot_params=SandplotParameters(**data['sandplot_params']),
                        message_signature=data['message_signature']
                    )
        except Exception as e:
            print(f"Error reading latest message: {e}")
        return None

    async def add_statistics(self, stats: Dict):
        """Records processing statistics to JSONL file."""
        async with aiofiles.open(self.stats_file, mode='a') as f:
            await f.write(json.dumps({**stats, 'timestamp': time.time()}) + '\n')

class DataFetcher:
    """Handles the fetching and processing of data from Hugging Face datasets."""
    
    def __init__(self):
        self.available_datasets = [
            DatasetConfig("wikipedia", "20220301.en", "text", 2000),
            DatasetConfig("code_search_net", "python", "content", 1500),
            DatasetConfig("github_issues", "all", "body", 1000),
            DatasetConfig("oscar", "unshuffled_deduplicated_en", "text", 3000)
        ]

    async def get_random_chunk(self, target_size_mb: float = 1.0) -> Tuple[str, DatasetConfig]:
        dataset_config = random.choice(self.available_datasets)
        
        try:
            target_bytes = int(target_size_mb * 1024 * 1024)
            estimated_samples = target_bytes // dataset_config.estimated_avg_size
            
            dataset = datasets.load_dataset(
                dataset_config.name,
                dataset_config.subset,
                streaming=True
            )
            
            text_chunks = []
            current_size = 0
            
            for item in dataset["train"].take(estimated_samples * 2):
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

class MetaSandplotSystem:
    """Main system that coordinates data fetching, processing, and visualization."""
    
    def __init__(self):
        self.db = JSONLDatabase()
        self.data_fetcher = DataFetcher()
        self.active_connections = set()
        
    def generate_sandplot_params(self, text: str) -> SandplotParameters:
        """Generates visualization parameters based on text characteristics."""
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
        previous_message = await self.db.get_latest_message()
        previous_hash = previous_message.text_hash if previous_message else "0" * 64
        
        # Create new chain message
        message = ChainMessage(
            text_hash=text_hash,
            text_content=text,
            dataset_name=dataset_config.name,
            dataset_subset=dataset_config.subset,
            timestamp=time.time(),
            previous_hash=previous_hash,
            sandplot_params=self.generate_sandplot_params(text),
            message_signature=None
        )
        
        # Store message
        await self.db.append_message(message)
        
        # Record statistics
        await self.db.add_statistics({
            'dataset_name': dataset_config.name,
            'bytes_processed': len(text.encode('utf-8')),
            'processing_time': time.time()
        })
        
        return message

    async def run_server(self, host: str = "localhost", port: int = 8765):
        """Runs the WebSocket server for real-time client updates."""
        async def client_handler(websocket, path):
            self.active_connections.add(websocket)
            try:
                while True:
                    message = await self.process_new_data()
                    if message:
                        message_data = {
                            "text_hash": message.text_hash,
                            "dataset": message.dataset_name,
                            "timestamp": message.timestamp,
                            "params": asdict(message.sandplot_params)
                        }
                        await websocket.send(json.dumps(message_data))
                    await asyncio.sleep(10)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.active_connections.remove(websocket)

        async with websockets.serve(client_handler, host, port):
            print(f"MetaSandplot server running on ws://{host}:{port}")
            await asyncio.Future()

async def main():
    """Main entry point for the MetaSandplot system."""
    system = MetaSandplotSystem()
    try:
        await system.run_server()
    except KeyboardInterrupt:
        print("\nShutting down MetaSandplot system...")

if __name__ == "__main__":
    asyncio.run(main())