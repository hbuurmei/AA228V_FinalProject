import numpy as np
import gymnasium as gym
from utils import load_config


def discretize_state(state, state_bounds, num_bins):
    """
    Convert continuous state to discrete indices.
    """
    indices = []
    for i, (lo, hi), bins in zip(range(4), state_bounds, num_bins):
        indices.append(np.clip(int((state[i] - lo) * bins / (hi - lo)), 0, bins-1))
    return tuple(indices)


def value_iteration(env_config):
    """
    Perform value iteration to compute the optimal value function.
    """

    # State bounds
    states_min = env_config["states_min"]
    states_max = env_config["states_max"]
    state_bounds = np.array([states_min, states_max]).T

    # Discretization (number of points per dimension)
    num_bins = [15, 15, 21, 21]
    assert len(num_bins) == env_config["state_dim"]

    # Create discretized state space
    grid = []
    for _, (lo, hi), bins in zip(range(4), state_bounds, num_bins):
        grid.append(np.linspace(lo, hi, bins))

    # Initialize value function
    V = np.zeros([n for n in num_bins])

    env = gym.make(env_config["name"], sutton_barto_reward=True)
    env.reset()

    # Initialize value function
    V = -1*np.ones([n for n in num_bins])

    # Value iteration
    gamma = 0.8
    theta = 0.1
    n_samples = 1  # Number of samples per action to account for stochasticity

    while True:
        delta = 0
        for indices in np.ndindex(V.shape):
            v = V[indices]
            state = np.array([grid[d][i] for d, i in enumerate(indices)])

            # Try both actions and store values in dict
            action_values = {}
            for action in [0, 1]:
                env.reset()
                # Set environment state
                env.env.env.env.state = state.copy()
                next_state, reward, terminated, truncated, _ = env.step(action)

                next_indices = discretize_state(next_state, state_bounds, num_bins)
                action_values[action] = reward + gamma * V[next_indices]

            # Take max value
            V[indices] = max(action_values.values())
            delta = max(delta, abs(v - V[indices]))

        print(delta)
        if delta < theta:
            break

    env.close()

    # Save results
    np.savez_compressed("data/datasets/value_iteration.npz", V=V, **{f"grid_dim_{i}": arr for i, arr in enumerate(grid)})


if __name__ == "__main__":
    env_config = load_config("config/env/cartpole.yaml")
    value_iteration(env_config)
