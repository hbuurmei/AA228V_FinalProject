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
    import pandas as pd
    import altair as alt
    import torch
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import sys; sys.path.append("src")
    from dynamics_learning import DynamicsLearner
    from utils import load_config
    return (
        DynamicsLearner,
        alt,
        gym,
        load_config,
        mo,
        np,
        pd,
        plt,
        sys,
        torch,
    )


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
def _(np):
    attribution_scores = np.load("data/scores/dynamics_learner_scores.npz")["scores"]
    training_data = np.load("data/datasets/dl_data_train.npz")
    target_data = np.load("data/datasets/dl_target_rollouts.npz")

    print(f"""
            Number of training data points: {attribution_scores.shape[0]}
            Number of target data points: {attribution_scores.shape[1]}
            Percentage of nonzero elements in DA scores matrix: {np.count_nonzero(attribution_scores) / np.prod(attribution_scores.shape) * 100:.2f}%
    """)
    return attribution_scores, target_data, training_data


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=200-1, step=1, show_value=True)
    slider
    return (slider,)


@app.cell
def _(attribution_scores, np, plt, slider):
    plt.figure()
    plt.hist(np.log10(np.abs(attribution_scores[:, slider.value])))
    plt.gcf()
    return


@app.cell
def _(attribution_scores, np):
    np.min(attribution_scores[:, 0]), np.max(attribution_scores[:, 0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### First metric: **entropy**

        First we convert the attribution scores to valid probability, i.e. the probability of training sample $i$ is equal to $p_i=\frac{\left|s_i\right|}{\sum_j\left|s_j\right|}$, where $s_i$ denotes the corresponding score.

        Then, we compute the entropy according to $H=-\sum_{i=1}^N p_i \log p_i$, where $N$ is the size of our training dataset.

        _Higher entropy_: influence is spread across many training points.
        _Lower entropy_: influence is concentrated on a few training points.### First metric: **entropy**

        First we convert the attribution scores to valid probability, i.e. the probability of training sample $i$ is equal to $p_i=\frac{\left|s_i\right|}{\sum_j\left|s_j\right|}$, where $s_i$ denotes the corresponding score.

        Then, we compute the entropy according to $H=-\sum_{i=1}^N p_i \log p_i$, where $N$ is the size of our training dataset.

        _Higher entropy_: influence is spread across many training points.
        _Lower entropy_: influence is concentrated on a few training points.
        """
    )
    return


@app.cell
def _(attribution_scores, np):
    def attribution_entropy(scores):
        """
        Compute entropy of attribution scores.
        """
        abs_scores = np.abs(scores)
        probs = abs_scores / np.sum(abs_scores)

        # Avoid log(0) by setting p log p = 0 when p = 0
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        return entropy


    entropies = [attribution_entropy(attribution_scores[:, i]) 
                 for i in range(attribution_scores.shape[1])]
    print(attribution_entropy(attribution_scores))
    print(np.array(entropies).mean())
    return attribution_entropy, entropies


@app.cell(hide_code=True)
def _(entropies, plt):
    plt.hist(entropies, bins=20, edgecolor="black", alpha=0.75)
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attribution Entropy Across Targets")
    return


@app.cell
def _(attribution_scores, np, plt, target_data, training_data):
    from matplotlib.lines import Line2D

    inspection_id = 5
    scores = attribution_scores[:, inspection_id]

    abs_scores = np.abs(scores)
    top_indices = np.argsort(abs_scores)[-10:]

    print(f"Top 10 scores: {scores[top_indices]}")

    top_training_data = training_data["X"][top_indices]

    min_idx = np.argsort(scores)[:10]  # top min
    max_idx = np.argsort(scores)[-10:][::-1]  # top max (descending)

    target_label = target_data["A"][inspection_id]

    axis1 = 0
    axis2 = 2

    plt.figure(figsize=(5, 3))
    plt.scatter(
        training_data["X"][:, axis1],
        training_data["X"][:, axis2],
        c=training_data["A"],
        cmap="coolwarm",
        alpha=0.01,
        marker="o"
    )
    plt.scatter(
        training_data["X"][min_idx, axis1],
        training_data["X"][min_idx, axis2],
        c=training_data["A"][min_idx],
        cmap="coolwarm",
        s=70,
        marker="^"
    )
    plt.scatter(
        training_data["X"][max_idx, axis1],
        training_data["X"][max_idx, axis2],
        c=training_data["A"][max_idx],
        cmap="coolwarm",
        s=70,
        marker="^"
    )
    plt.scatter(
        target_data["X"][inspection_id, axis1],
        target_data["X"][inspection_id, axis2],
        c=target_data["A"][inspection_id],
        cmap="coolwarm",
        s=90,
        marker="X"
    )
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Target point"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Training data"),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Important points"),
    ]
    plt.xlabel(r"$v\ [\text{m sec}^{-1}]$")
    plt.ylabel(r"$\omega\ [\text{sec}^{-1}]$")
    yticks = plt.yticks()[0]
    plt.yticks(yticks, [f"{np.rad2deg(x):.0f}°" for x in yticks])
    # plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.gcf()
    return (
        Line2D,
        abs_scores,
        axis1,
        axis2,
        inspection_id,
        legend_elements,
        max_idx,
        min_idx,
        scores,
        target_label,
        top_indices,
        top_training_data,
        yticks,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
