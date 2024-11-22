import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HanoiTransformer(nn.Module):
    def __init__(self, num_disks=4):
        super().__init__()
        self.num_disks = num_disks
        state_size = num_disks * 3  # State representation for 3 pegs
        
        # Encoder now outputs the correct dimension for attention
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        
        self.move_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9)
        )
        
    def forward(self, x):
        # Encode the state
        x = self.encoder(x)
        
        # Reshape for attention: [batch_size, seq_len=1, embedding_dim=32]
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply self-attention (x is now properly shaped)
        attn_out, _ = self.attention(x, x, x)
        
        # Remove sequence dimension and predict move
        x = attn_out.squeeze(1)
        return self.move_predictor(x)

def make_move(state, move, num_disks=4):
    """Make a move in the Hanoi tower state"""
    new_state = state.copy()
    source, target = move
    
    # Find and move the top disk
    source_top = -1
    for disk in range(num_disks):
        if new_state[source * num_disks + disk] == 1:
            source_top = disk
            new_state[source * num_disks + disk] = 0
            break
            
    # Place on target peg
    for disk in range(num_disks):
        if new_state[target * num_disks + disk] == 0:
            new_state[target * num_disks + disk] = 1
            break
            
    return new_state

def generate_hanoi_dataset(num_disks=4, num_samples=1000):
    """Generate Towers of Hanoi moves dataset"""
    def get_valid_moves(state):
        valid = []
        for source in range(3):
            # Find top disk on source peg
            source_top = -1
            for disk in range(num_disks):
                if state[source * num_disks + disk] == 1:
                    source_top = disk
                    break
            if source_top == -1:
                continue
                
            for target in range(3):
                if source != target:
                    # Find top disk on target peg
                    target_top = -1
                    for disk in range(num_disks):
                        if state[target * num_disks + disk] == 1:
                            target_top = disk
                            break
                            
                    # Move is valid if target peg is empty or top disk is larger
                    if target_top == -1 or target_top > source_top:
                        valid.append((source, target))
        return valid

    X, y = [], []
    
    # Generate samples
    for _ in range(num_samples):
        # Random initial state
        state = np.zeros(num_disks * 3)
        for disk in range(num_disks):
            peg = np.random.randint(3)
            state[peg * num_disks + disk] = 1
            
        valid_moves = get_valid_moves(state)
        if valid_moves:
            move = valid_moves[np.random.randint(len(valid_moves))]
            move_idx = move[0] * 3 + move[1]
            
            X.append(state)
            y.append(move_idx)
    
    # Convert list to numpy array first to avoid the slow operation warning
    X = np.array(X)
    y = np.array(y)
    return torch.FloatTensor(X), torch.LongTensor(y)

def train_hanoi_model(model, num_epochs=100, batch_size=32):
    """Train the Hanoi model"""
    X, y = generate_hanoi_dataset()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        # Random batch
        indices = torch.randperm(len(X))[:batch_size]
        batch_X = X[indices]
        batch_y = y[indices]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

def solve_hanoi(model, initial_state):
    """Use the model to solve Towers of Hanoi"""
    state = initial_state.clone()
    moves = []
    
    for _ in range(50):  # Max moves limit
        with torch.no_grad():
            # Add batch dimension if not present
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            
            output = model(state)
            move_idx = output.argmax().item()
            
            source = move_idx // 3
            target = move_idx % 3
            moves.append((source, target))
            
            # Update state (remove batch dimension for make_move)
            state = torch.FloatTensor(make_move(state.squeeze(0).numpy(), (source, target)))
            
            # Check if solved (all disks on last peg)
            if state[-4:].sum() == 4:
                break
                
    return moves

# Example usage
if __name__ == "__main__":
    model = HanoiTransformer()
    train_hanoi_model(model)
    
    # Test with initial state (all disks on first peg)
    initial_state = torch.zeros(12)
    initial_state[:4] = 1
    
    solution = solve_hanoi(model, initial_state)
    print("Solution moves:", solution)