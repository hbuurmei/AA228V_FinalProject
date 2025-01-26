import gymnasium as gym
import numpy as np


def get_dataset_from_model(config, model, episodes):
    env = gym.make(config["name"])
    
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


class DTAgent:
    def __init__(self, dt):
        self.dt = dt
    
    def act(self, state):
        if isinstance(state, tuple):
            state = state[0]
        return self.dt.predict([state])[0]
    
    def predict(self, X):
        return self.dt.predict(X)
