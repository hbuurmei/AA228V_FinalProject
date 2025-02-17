import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import MLP, policy_rollout


class ILAgent:
    """
    Imitation Learning Agent with differentiable classifier.
    """
    def __init__(self, agent_config, seed=0):
        torch.manual_seed(seed)
        self.extract_config(agent_config)
        self.model = MLP(input_dim=self.input_dim, hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.CrossEntropyLoss()
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
        Predict class probabilities for a batch of states.
        """
        self.model.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.model(state)
        
    def batch_predict(self, X):
        """
        Predict class probabilities for a batch of states.
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
        target = torch.LongTensor(target).to(self.device)
        
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
        y = torch.LongTensor(y).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        """
        Choose an action based on the current state.
        """
        logits = self.predict(state)
        return torch.argmax(logits).item()

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


def get_dataset_from_model(model, env_config, episodes):
    """
    Collect states and actions from a model in an environment.
    """
    env = gym.make(env_config["name"])
    
    X = []
    y = []

    for _ in range(episodes):
        state = env.reset()[0]
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = model.act(state)
            next_state, _, terminated, truncated, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    env.close()

    return np.array(X), np.array(y)


def label_dataset_with_model(model, X):
    """
    Get labels for a dataset using a model.
    """
    batch_logits = model.batch_predict(X)
    y = torch.argmax(batch_logits, dim=-1)
    return y


def train_il_agent(agent_config, expert, env_config):
    """
    Train an Imitation Learning agent on expert data.
    """
    # Collect dataset from the expert for both training and testing
    X_train, y_train = get_dataset_from_model(expert, env_config, episodes=agent_config["train_data_episodes"])
    np.savez_compressed("data/datasets/expert_data_train.npz", X=X_train, y=y_train)
    X_test, y_test = get_dataset_from_model(expert, env_config, episodes=agent_config["test_data_episodes"])
    np.savez_compressed("data/datasets/expert_data_test.npz", X=X_test, y=y_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=agent_config["batch_size"], shuffle=True)

    # We train multiple models to do data attribution
    for seed in tqdm(range(agent_config["num_models"]), desc="Training BC models"):
        agent0 = ILAgent(agent_config, seed=seed)
        
        # Fit the model to the training data, i.e. behavior cloning
        for _ in range(agent_config["max_epochs"]):
            for batch in train_loader:
                X_batch, y_batch = batch
                agent0.batch_fit(X_batch, y_batch)
        agent0.save_model(f"data/models/checkpoints/{agent_config["method"]}_policy_{seed}.pt")
    
    if agent_config["method"] == "BC":
        # Behavioral Cloning (BC)
        agent = agent0
    
    elif agent_config["method"] == "AO":
        # Alternating Optimization (AO)
        
        # Initialize to behavior cloning policy and tracking variables
        policy = agent0
        best_reward = -np.inf
        best_model = policy

        for _ in tqdm(range(50), desc="AO Iterations"):
            # Collect states using current policy
            X, _ = get_dataset_from_model(policy, env_config, episodes=agent_config["train_data_episodes"])

            # Get expert labels for visited states
            y = label_dataset_with_model(expert, X)

            # Train updated policy
            new_agent = ILAgent(agent_config)
            for _ in range(agent_config["max_epochs"]):
                for batch in train_loader:
                    X_batch, y_batch = batch
                    new_agent.batch_fit(X_batch, y_batch)
            policy = new_agent

            # Evaluate and track best policy
            avg_reward = policy_rollout(policy, env_config, N=100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = policy
        
        agent = best_model

    elif agent_config["method"] == "DA":
        # Data Aggregation (DA)

        # Initialize to behavior cloning policy and tracking variables
        policy = agent0
        best_reward = -np.inf
        best_model = policy

        for _ in tqdm(range(50), desc="DA Iterations"):
            # Collect states using current policy
            X, _ = get_dataset_from_model(policy, env_config, episodes=agent_config["train_data_episodes"])

            # Get expert labels for visited states
            y = label_dataset_with_model(expert, X)

            # Aggregate the data
            X = np.concatenate([X_train, X])
            y = np.concatenate([y_train, y])

            # Train updated policy
            new_agent = ILAgent(agent_config)
            for _ in range(agent_config["max_epochs"]):
                for batch in train_loader:
                    X_batch, y_batch = batch
                    new_agent.batch_fit(X_batch, y_batch)
            policy = new_agent

            # Evaluate and track best policy
            avg_reward = policy_rollout(policy, env_config, N=100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = policy
        
        agent = best_model

    else:
        raise ValueError("Invalid IL method. Please choose BC, AO, or DA.")

    # Evaluate the resulting policy
    avg_reward = policy_rollout(agent, env_config, N=100)
    print(f"Average reward of {agent_config["method"]} agent: {avg_reward:.2f}")

    # Save the model
    agent.save_model(f"data/models/{agent_config["method"]}_policy.pt")
    return agent, avg_reward
