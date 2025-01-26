import gymnasium as gym
import numpy as np


def get_dataset_from_model(config, model, episodes, verbose=False):
    env = gym.make(config["name"])
    
    X = []
    y = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = model.act(state)
            next_state, _, terminated, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    
    env.close()

    X = np.array(X)
    y = np.array(y)

    return X, y
