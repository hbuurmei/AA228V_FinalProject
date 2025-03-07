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
    A generic multi-layer perceptron (MLP) with optional uncertainty prediction.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list, 
                 output_dim: int, 
                 activation=nn.ReLU,
                 predict_uncertainties=False):
        super(MLP, self).__init__()

        layers = []
        in_dim = input_dim
        
        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        
        # Output layer(s)
        if predict_uncertainties:
            # Single output head for both mean and log variance
            layers.append(nn.Linear(in_dim, 2 * output_dim))
        else:
            # Standard output head for mean only
            layers.append(nn.Linear(in_dim, output_dim))

        # Pack all layers into a Sequential container
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class UncertaintyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Single output head predicting both mean and log variance
        self.output_head = nn.Linear(hidden_dim, 2 * output_dim)
        self.output_dim = output_dim
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = self.output_head(features)
        
        # Split the outputs into mean and log variance
        mean = outputs[:, :self.output_dim]
        log_var = outputs[:, self.output_dim:]
        
        return mean, log_var


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


def get_dataset_from_hyperrectangle(env_config, num_samples=1000):
    """
    Create a dataset by sampling states from a hyperrectangle and random actions,
    then collecting the corresponding next states.
    """
    env = gym.make(env_config["name"])

    states_min = env_config["states_min"]
    states_max = env_config["states_max"]
    state_bounds = np.array([states_min, states_max]).T
    
    X = []   # states
    A = []   # actions
    Xp = []  # next states
    
    samples_collected = 0
    while samples_collected < num_samples:
        # Sample a random state from the hyperrectangle
        sampled_state = np.array([
            np.random.uniform(low=bounds[0], high=bounds[1]) 
            for bounds in state_bounds
        ])
        
        # Sample a random action (0 or 1)
        action = np.random.choice([0, 1])
        
        try:
            # Reset environment and force the state
            env.reset()
            env.unwrapped.state = sampled_state
            
            # Take the action and observe the next state
            next_state, _, _, _, _ = env.step(action)
            
            # If the step was successful, add it to our dataset
            X.append(sampled_state)
            A.append(action)
            Xp.append(next_state)
            
            samples_collected += 1
                
        except Exception as e:
            # Some sampled states might not be valid for the environment
            print(f"Skipping invalid state: {sampled_state}. Error: {e}")
            continue
    
    env.close()
    
    return np.array(X), np.array(A), np.array(Xp)


def make_extra_data(config):
    """
    Generate synthetic data based on the provided configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with the following keys:
        - centroid: list or np.ndarray, center point of the shape
        - eps: float, radius (for spherical) or half-width (for rectangular)
        - shape: str, 'rectangular' or 'spherical'
        - dim: int, dimensionality of the data
        - label: int, label to assign to all generated points
        - num_samples: int, number of samples to generate
    
    Returns:
    --------
    X : np.ndarray
        Generated feature data with shape (num_samples, dim)
    y : np.ndarray
        Labels for the generated data, all set to config['label']
    """
    centroid = np.array(config['centroid'])
    eps = config['eps']
    shape = config['shape']
    dim = config['dim']
    label = config['label']
    num_samples = config['num_samples']
    
    # Validate inputs
    assert len(centroid) == dim, f"Centroid dimension ({len(centroid)}) must match specified dimension ({dim})"
    assert shape in ['rectangular', 'spherical'], f"Shape must be 'rectangular' or 'spherical', got {shape}"
    
    # Initialize data array
    X = np.zeros((num_samples, dim))
    
    if shape == 'rectangular':
        # For rectangular shape, sample uniformly within a hypercube
        for i in range(num_samples):
            # Sample each dimension uniformly within [centroid[d] - eps, centroid[d] + eps]
            for d in range(dim):
                X[i, d] = np.random.uniform(centroid[d] - eps, centroid[d] + eps)
    
    elif shape == 'spherical':
        # For spherical shape, sample uniformly within a hypersphere
        for i in range(num_samples):
            while True:
                # Sample from a cube and reject if outside the sphere
                point = np.random.uniform(-eps, eps, dim)
                if np.linalg.norm(point) <= eps:  # Check if point is within the sphere
                    X[i] = centroid + point
                    break
    
    # Create labels array (all the same value)
    y = np.full(num_samples, label, dtype=int)
    
    return X, y


def extract_states_in_region(X_train, A_train, Xp_train, region_bounds):
    """
    Extract states that fall within a specified rectangular region and remove them from the original dataset.
    """
    # Ensure region_bounds matches the number of state dimensions
    assert len(region_bounds) == X_train.shape[1], "Number of region bounds must match number of state dimensions"
    
    # Initialize mask to track which states are inside the region
    mask = np.ones(X_train.shape[0], dtype=bool)
    
    # Check each dimension
    for dim, (min_val, max_val) in enumerate(region_bounds):
        # Update mask to identify points within bounds for this dimension
        mask = mask & (X_train[:, dim] >= min_val) & (X_train[:, dim] <= max_val)
    
    # Extract states within the region
    extracted_states = X_train[mask]
    
    # Get the remaining states, actions and next states
    remaining_states = X_train[~mask]
    remaining_actions = A_train[~mask]
    remaining_next_states = Xp_train[~mask]
    
    return extracted_states, remaining_states, remaining_actions, remaining_next_states
