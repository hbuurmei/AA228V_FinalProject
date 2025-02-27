import yaml
import gymnasium as gym
import numpy as np
import torch.nn as nn


def is_in_hull(point, hull, eps=None):
    """
    Check if point is in convex hull within epsilon tolerance.

    >>> import torch
    >>> from scipy.spatial import ConvexHull
    >>> pts = torch.rand(100, 4)
    >>> hull = ConvexHull(pts)
    >>> all(is_in_hull(pts[i,:], hull) for i in range(pts.shape[0]))
    True
    """
    point = np.asarray(point)
    eps = (eps if eps else np.sqrt(np.finfo(point.dtype).eps))
    equations = hull.equations  # (normal_vector, offset) for each facet
    return np.all(np.dot(equations[:, :-1], point) + equations[:, -1] <=eps)


class MLP(nn.Module):
    """
    A generic multi-layer perceptron (MLP).
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list, 
                 output_dim: int, 
                 activation=nn.ReLU):
        super(MLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        
        # Add final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        # Pack all layers into a Sequential container
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


def policy_rollout(agent, env_config, N=1, render=False):
    """
    Evaluate a policy by rolling it out in an environment.
    """
    env = gym.make(env_config["name"], render_mode="human" if render else None)
    total_reward = 0

    for _ in range(N):
        state, _ = env.reset()
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_reward += episode_reward
        
    env.close()
    return total_reward / N


def load_config(config_path):
    """
    Load configuration file from path.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_dataset_from_model(model, env_config, episodes, max_steps=500):
    """
    Collect states, actions and next states from a model in an environment.
    """
    env = gym.make(env_config["name"])
    
    X = []   # states
    A = []   # actions
    Xp = []  # next states

    for _ in range(episodes):
        state = env.reset()[0]
        terminated = truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < max_steps:
            steps += 1
            action = model.act(state)
            next_state, _, terminated, truncated, _ = env.step(action)

            X.append(state)
            A.append(action)
            Xp.append(next_state)

            state = next_state
    env.close()

    return np.array(X), np.array(A), np.array(Xp)
