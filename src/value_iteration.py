import numpy as np
import gymnasium as gym
import pickle

# State bounds (position, velocity, angle, angular velocity)
state_bounds = np.array([
    [-4.8, 4.8],     # cart position
    [-4.0, 4.0],     # cart velocity
    [-0.5, 0.5],     # pole angle
    [-4.0, 4.0]      # pole angular velocity
])

# Discretization (number of points per dimension)
# num_bins = [20, 20, 20, 20]
num_bins = [15, 15, 21, 21]

# Create discretized state space
grid = []
for dim, (lo, hi), bins in zip(range(4), state_bounds, num_bins):
    grid.append(np.linspace(lo, hi, bins))

# Initialize value function
V = np.zeros([n for n in num_bins])

env = gym.make('CartPole-v1', sutton_barto_reward=True)
env.reset()

# State bounds and discretization setup as before...

def discretize_state(state):
    """Convert continuous state to discrete indices"""
    indices = []
    for i, (lo, hi), bins in zip(range(4), state_bounds, num_bins):
        indices.append(np.clip(int((state[i] - lo) * bins / (hi - lo)), 0, bins-1))
    return tuple(indices)

# Initialize value function
V = -1*np.ones([n for n in num_bins])

# Value iteration
gamma = 0.8
# theta = 0.001
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
            env.env.env.env.state = state.copy()  # Set environment state
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_indices = discretize_state(next_state)
            action_values[action] = reward + gamma * V[next_indices]

        # Take max value
        V[indices] = max(action_values.values())
        delta = max(delta, abs(v - V[indices]))

    print(delta)
    if delta < theta:
        break

env.close()
np.save("../data/models/qvalues", V)
with open("../data/models/grid_vals.pkl", "wb") as f:
    pickle.dump(grid, f)
