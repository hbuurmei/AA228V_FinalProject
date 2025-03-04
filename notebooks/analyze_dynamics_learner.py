import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analyze Dynamics Learner""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import torch
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import sys; sys.path.append("src")
    from dynamics_learning import DynamicsLearner
    from utils import load_config
    return DynamicsLearner, gym, load_config, mo, np, plt, sys, torch


@app.cell
def _(DynamicsLearner, load_config):
    # Import the learned dynamics model
    dl_config = load_config("config/train/dynamics_learner_cartpole.yaml")
    dynamics_learner = DynamicsLearner(dl_config)
    dynamics_learner.load_model("data/models/dynamics_learner.pt")
    return dl_config, dynamics_learner


@app.cell
def _(dynamics_learner, gym, load_config, np):
    # We initiate the states at the origin and sample random actions, to compare predicted states and true states
    env_config = load_config("config/env/cartpole.yaml")
    env = gym.make(env_config["name"])

    # We evaluate multiple episodes
    episodes = 5
    max_steps = 500

    X_true = []  # true states
    X_pred = []  # predicted states
    A_rand = []  # actions

    for _ in range(episodes):
        state_true = env.reset()[0]
        state_pred = state_true

        X_true_ep = []
        X_pred_ep = []
        A_rand_ep = []

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            steps += 1
            act_rand = np.random.choice([0, 1])
            next_state_true, _, terminated, truncated, _ = env.step(act_rand)
            dl_output = dynamics_learner.predict(state_pred.reshape(1, -1), np.array(act_rand).reshape(1, -1)).numpy().squeeze()
            next_state_pred = dl_output[:4]
            next_state_uncertainty = dl_output[4:]

            X_true_ep.append(state_true)
            X_pred_ep.append(state_pred)
            A_rand_ep.append(act_rand)

            state_true = next_state_true
            state_pred = next_state_pred

        X_true.append(np.array(X_true_ep))
        X_pred.append(np.array(X_pred_ep))
        A_rand.append(np.array(A_rand_ep))

    env.close()
    return (
        A_rand,
        A_rand_ep,
        X_pred,
        X_pred_ep,
        X_true,
        X_true_ep,
        act_rand,
        dl_output,
        env,
        env_config,
        episodes,
        max_steps,
        next_state_pred,
        next_state_true,
        next_state_uncertainty,
        state_pred,
        state_true,
        steps,
        terminated,
        truncated,
    )


@app.cell(hide_code=True)
def _(X_pred, X_true, episodes, np, plt):
    # We plot the true state propagation and the prediction for several episodes
    state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

    fig, axes = plt.subplots(episodes, 4, figsize=(16, 4*episodes))
    if episodes == 1:
        axes = axes.reshape(1, -1)

    for ep in range(episodes):
        x_true = X_true[ep]
        x_pred = X_pred[ep]
        steps_range = np.arange(len(x_true))

        for state_idx in range(4):
            ax = axes[ep, state_idx]
            ax.plot(steps_range, x_true[:, state_idx], 'b-', label='True State', linewidth=2)        
            ax.plot(steps_range, x_pred[:, state_idx], 'r--', label='Predicted State', linewidth=2)
            ax.set_title(f'Episode {ep+1}: {state_names[state_idx]}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('State Value')

            # Only add legend to the first row to avoid clutter
            if ep == 0:
                ax.legend()            
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gcf()
    return (
        ax,
        axes,
        ep,
        fig,
        state_idx,
        state_names,
        steps_range,
        x_pred,
        x_true,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
