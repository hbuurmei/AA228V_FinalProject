import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
from utils import MLP


class RLAgent:
    """
    Reinforcement Learning Agent using Deep Q-Learning.
    """
    def __init__(self, agent_config, seed=0):
        torch.manual_seed(seed)
        self.extract_config(agent_config)
        self.exploration_rate = self.exploration_max
        self.memory = deque(maxlen=self.memory_size)
        self.model = MLP(input_dim=self.input_dim, hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_config(self, config):
        """
        Extract config dictionary and set all values as class attributes.
        """
        for key, value in config.items():
            setattr(self, key, value)
    
    def predict(self, state):
        """
        Predict Q-values for a given state.
        """
        self.model.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.model(state)
    
    def batch_predict(self, X):
        """
        Predict Q-values for a batch of states.
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            return self.model(X)
    
    def fit(self, state, target):
        """
        Fit the model to a single state and target.
        """
        self.model.train()
        state = torch.FloatTensor(state).to(self.device)
        target = torch.FloatTensor(target).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(state)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
    
    def batch_fit(self, X, y):
        """
        Fit the model to a batch of states and targets.
        """
        self.model.train()
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
    
    def act(self, state):
        """
        Choose an action based on the current state.
        """
        if random.random() < self.exploration_rate:
            return int(random.randrange(2))
        q_values = self.predict(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, terminated):
        """
        Store the experience in memory.
        """
        self.memory.append((state, action, reward, next_state, terminated))
    
    def experience_replay(self):
        """
        Experience replay to train the model.
        """
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        
        for state, action, reward, next_state, terminated in batch:
            if terminated:
                target_q = reward
            else:
                target_q = reward + self.gamma * torch.amax(self.predict(next_state))
                
            target = self.predict(state)
            target[action] = target_q
            
            states.append(state)
            targets.append(target)
        
        states = np.array(states)
        targets = np.array(targets)
        self.batch_fit(states, targets)
        
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)
    
    def save_model(self, filename):
        """
        Save model weights to a file.
        """
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, filename):
        """
        Load model weights from a file.
        """
        self.model.load_state_dict(torch.load(filename, weights_only=True))
        self.model.eval()


def train_rl_agent(agent_config, env_config):
    """
    Train a Reinforcement Learning agent in a gym environment.
    """
    env = gym.make(env_config["name"], render_mode="human" if env_config["render"] else None)
    agent = RLAgent(agent_config)
    scores = []
    
    for episode in range(agent_config["max_episodes"]):
        state = env.reset()[0]
        score = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, terminated)
            agent.experience_replay()
            
            state = next_state
            score += reward

        scores.append(score)

        # Take average of last 100 episodes
        avg_score = np.mean(scores[-100:])
        
        if episode % 50 == 0:
            print(f"Episode {episode} Score: {score} Average Score: {avg_score:.2f} Exploration: {agent.exploration_rate:.3f}")
        
        # Save model if we achieve the target score
        if avg_score >= env_config["target_score"]:
            print(f"Environment solved in {episode} episodes!")
            agent.save_model("data/models/expert_policy.pt")
            break
    
    env.close()
    return agent, scores
