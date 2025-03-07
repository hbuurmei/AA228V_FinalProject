import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Training Data Inspection

        The data used for training the IL agents consists of state-action pairs, $\mathcal{D}=\left\{\left(s_i, a_i \\right) \\right\}_{i=1}^N$, where $a_i=\pi^\star (s_i)$.

        For the _CartPole-V1_ environment, the four states correspond to, in order, $\{x, v, \\theta, \omega \}$. Also, the maximum episode length for that environment is equal to $500$.

        In this notebook we want to inspect this data and compute quantities such as data coverage of the ground truth recoverable states.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt


@app.cell
def _(np):
    # Import training data
    expert_data_train = np.load("data/datasets/expert_data_train.npz")
    states_train = expert_data_train["X"]
    actions_train = expert_data_train["y"]
    N_train = states_train.shape[0]
    print(f"Number of data points in training dataset: {N_train}")
    print(f"Note: if none of the episodes were terminated early, this corresponds to {N_train//500} episodes")
    return N_train, actions_train, expert_data_train, states_train


@app.cell
def _(np, states_train):
    # Check basic statistics of visited states
    stats = {
        "mean": np.mean(states_train, axis=0),
        "std": np.std(states_train, axis=0),
        "min": np.min(states_train, axis=0),
        "max": np.max(states_train, axis=0),
        "range": np.ptp(states_train, axis=0)
    }
    stats
    return (stats,)


@app.cell(hide_code=True)
def _(actions_train, np, plt):
    # Show split in actions
    unique_actions, counts = np.unique(actions_train, return_counts=True)

    plt.figure(figsize=(3, 3))
    plt.bar(unique_actions, counts, tick_label=[f"{int(a)}" for a in unique_actions])
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Distribution of Actions")
    return counts, unique_actions


@app.cell(hide_code=True)
def _(plt, states_train):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot positional phase space
    axes[0].scatter(states_train[:, 0], states_train[:, 1])
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$v$")
    axes[0].set_title("Training data positional phase space")

    # Plot angular phase space
    axes[1].scatter(states_train[:, 2], states_train[:, 3])
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$\omega$")
    axes[1].set_title("Training data angular phase space")
    plt.tight_layout()
    fig
    return axes, fig


@app.cell(hide_code=True)
def _(plt, states_train):
    # Plot states in 3D (disregarding position)

    from mpl_toolkits.mplot3d import Axes3D

    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')
    ax.scatter(states_train[:, 1], states_train[:, 2], states_train[:, 3], s=1)
    ax.set_xlabel(r'$v$')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$\omega$')
    plt.title("3D State Space Coverage")
    return Axes3D, ax, fig3d


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Coverage""")
    return


@app.cell
def _(states_train):
    # We first just compute the volume of the convex hull of the states in the training data
    from scipy.spatial import ConvexHull

    hull = ConvexHull(states_train)
    print(f"Convex hull volume of training data (states): {hull.volume}")
    return ConvexHull, hull


if __name__ == "__main__":
    app.run()
