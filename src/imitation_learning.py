import gymnasium as gym
import numpy as np


def get_dataset_from_model(config, model, episodes):
    env = gym.make(config["name"])
    
    X = []
    y = []

    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        
        while not done:
            action = model.act(state)
            next_state, _, done, _, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    env.close()

    return np.array(X), np.array(y)
