from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import torch
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Data structure for training metrics that will be sent to the client"""
    current_loss: float
    best_loss: float
    batch_number: int
    total_batches: int
    attention_norm: float
    mlp_norm: float
    validation_loss: Optional[float]
    gpu_memory: float
    samples: List[str]
    learning_rate: float
    training_time: float

class TrainingManager:
    """Manages the training state and metrics"""
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = datetime.now()
        
        # Training state
        self.current_loss = float('inf')
        self.best_loss = float('inf')
        self.batch_number = 0
        self.attention_norm = 0.0
        self.mlp_norm = 0.0
        self.validation_loss = None
        self.samples = []
        self.is_training = False
        
        # Initialize model and optimizer
        self.setup_model()
        self.setup_data()
    
    def setup_model(self):
        """Initialize the transformer model with normalization"""
        self.model = nGPT(
            num_tokens=256,
            dim=512,
            depth=8,
            tied_embedding=True,
            add_value_residual=True,
            attn_norm_qk=False,
            manual_norm_weights=not self.config['USE_PARAMETRIZE']
        ).to(self.device)
        
        # Initialize learnable eigen learning rates
        self.alpha_attention = torch.nn.Parameter(torch.ones(1))
        self.alpha_mlp = torch.nn.Parameter(torch.ones(1))
        
        self.optim = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['LEARNING_RATE']
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['USE_AMP'])
    
    def setup_data(self):
        """Setup data loaders with proper batching"""
        # Data loading code remains similar to original
        pass
    
    async def train_step(self) -> TrainingMetrics:
        """Execute a single training step and return metrics"""
        self.model.train()
        
        for _ in range(self.config['GRAD_ACCUM_EVERY']):
            data = next(self.train_loader)
            data = data.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config['USE_AMP']):
                # Track normalization metrics
                attention_norm = torch.norm(self.model.attention_weights)
                mlp_norm = torch.norm(self.model.mlp_weights)
                
                loss = self.model(data, return_loss=True)
            
            self.scaler.scale(loss / self.config['GRAD_ACCUM_EVERY']).backward()
            
            # Update metrics
            self.current_loss = loss.item()
            self.best_loss = min(self.best_loss, self.current_loss)
            self.attention_norm = attention_norm.item()
            self.mlp_norm = mlp_norm.item()
        
        # Optimizer step
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad()
        
        # Generate sample occasionally
        if self.batch_number % self.config['GENERATE_EVERY'] == 0:
            sample = await self.generate_sample()
            self.samples.append(sample)
            self.samples = self.samples[-5:]  # Keep last 5 samples
        
        # Get current metrics
        metrics = TrainingMetrics(
            current_loss=self.current_loss,
            best_loss=self.best_loss,
            batch_number=self.batch_number,
            total_batches=self.config['NUM_BATCHES'],
            attention_norm=self.attention_norm,
            mlp_norm=self.mlp_norm,
            validation_loss=self.validation_loss,
            gpu_memory=torch.cuda.max_memory_allocated() / 1024**3 
                      if torch.cuda.is_available() else 0,
            samples=self.samples,
            learning_rate=self.config['LEARNING_RATE'],
            training_time=(datetime.now() - self.start_time).total_seconds()
        )
        
        self.batch_number += 1
        return metrics
    
    async def generate_sample(self) -> str:
        """Generate a text sample from the current model state"""
        self.model.eval()
        with torch.no_grad():
            # Sample generation code remains similar to original
            pass
        return generated_text

# FastAPI application
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global training manager instance
training_manager: Optional[TrainingManager] = None

@app.on_event("startup")
async def startup_event():
    """Initialize training manager on startup"""
    global training_manager
    
    config = {
        'NUM_BATCHES': int(1e5),
        'BATCH_SIZE': 4,
        'GRAD_ACCUM_EVERY': 4,
        'LEARNING_RATE': 1e-3,
        'VALIDATE_EVERY': 100,
        'GENERATE_EVERY': 500,
        'GENERATE_LENGTH': 512,
        'SEQ_LEN': 512,
        'USE_AMP': True,
        'USE_PARAMETRIZE': True
    }
    
    training_manager = TrainingManager(config)
    logger.info("Training manager initialized")

@app.websocket("/ws/training")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    
    try:
        while True:
            # Check if we should stop training
            if training_manager.batch_number >= training_manager.config['NUM_BATCHES']:
                await websocket.close()
                break
            
            # Execute training step and send metrics
            metrics = await training_manager.train_step()
            await websocket.send_json(asdict(metrics))
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/training/control")
async def control_training(command: dict):
    """Endpoint for controlling the training process"""
    global training_manager
    
    if command['action'] == 'start':
        training_manager.is_training = True
        return {"status": "Training started"}
    elif command['action'] == 'stop':
        training_manager.is_training = False
        return {"status": "Training stopped"}
    elif command['action'] == 'generate_sample':
        sample = await training_manager.generate_sample()
        return {"sample": sample}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
