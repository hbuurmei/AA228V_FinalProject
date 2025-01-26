import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

MEMORY_SIZE = 1000000
BATCH_SIZE = 32
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
GAMMA = 0.95
ALPHA = 0.001


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )
    
    def forward(self, x):
        return self.layers(x)


class MLPAgent:
    def __init__(self, exploration_rate=EXPLORATION_MAX):
        self.exploration_rate = exploration_rate
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = MLP()
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            return self.model(state).cpu().numpy()[0]
    
    def batch_predict(self, X):
        self.model.eval()
        with torch.no_grad():
            # Handle both tuple and array inputs
            if isinstance(state, tuple):
                state = state[0]
            X = torch.FloatTensor(X).to(self.device)
            return self.model(X).cpu().numpy()
    
    def fit(self, state, target):
        self.model.train()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(state)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
    
    def batch_fit(self, X, y):
        self.model.train()
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
    
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return int(random.randrange(2))
        q_values = self.predict(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        batch = random.sample(self.memory, BATCH_SIZE)
        states = []
        targets = []
        
        for state, action, reward, next_state, done in batch:
            if done:
                target_q = reward
            else:
                target_q = reward + GAMMA * np.amax(self.predict(next_state))
                
            target = self.predict(state)
            target[action] = target_q
            
            states.append(state)
            targets.append(target)
        
        states = np.array(states)
        targets = np.array(targets)
        self.batch_fit(states, targets)
        
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, weights_only=True))
        self.model.eval()
