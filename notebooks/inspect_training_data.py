import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Training Data Inspection

        The data used for training the IL agents consists of state-action pairs, $\mathcal{D}=\left\{\left(s_i, a_i \\right) \\right\}_{i=1}^N$, where $a_i=\pi^\star (s_i)$.

        In this notebook we want to inspect this data and compute quantities such as data coverage and compare with ground truth recoverable states.
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
    expert_dataset = np.load('data/datasets/expert_dataset.npz')
    states = expert_dataset['X']
    actions = expert_dataset['y']
    return actions, expert_dataset, states


@app.cell
def _(np, states):
    # Check basic statistics of visited states
    stats = {
        "mean": np.mean(states, axis=0),
        "std": np.std(states, axis=0),
        "min": np.min(states, axis=0),
        "max": np.max(states, axis=0),
        "range": np.ptp(states, axis=0)
    }
    stats
    return (stats,)


@app.cell
def _(actions, np, plt):
    # Show split in actions
    unique_actions, counts = np.unique(actions, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar(unique_actions, counts, tick_label=[f"{int(a)}" for a in unique_actions])
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Distribution of Actions')
    return counts, unique_actions


@app.cell
def _(plt, states):
    # Plot states in 3D (position left out)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states[:, 1], states[:, 2], states[:, 3], s=1)
    ax.set_xlabel(r'$\dot{x}$')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$\dot{\theta}$')
    plt.title("3D State Space Coverage")
    return Axes3D, ax, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Coverage""")
    return


@app.cell
def _(np, states, stats):
    from scipy.spatial import KDTree


    def create_grid(stats, num_points=15):
        """
        Create a grid of points covering the space defined by stats['min'] and stats['max'].

        Parameters:
        - stats: A dictionary with 'min' and 'max' keys representing min and max values for each dimension.
        - num_points: Number of points per dimension to create the grid.

        Returns:
        - ndarray: A grid of shape (num_points^dim, dim).
        """
        bounds = np.array([stats["min"], stats["max"]]).T  # Shape (4, 2) if there are 4 dimensions
        grid = np.array(np.meshgrid(*[np.linspace(b[0], b[1], num_points) for b in bounds])).reshape(len(bounds), -1).T
        return grid


    def dispersion(S, V):
        """
        Compute the dispersion metric: the maximum minimum distance from points in S to V.
        
        Parameters:
        - S: Grid of points covering the space of interest.
        - V: Reference set of points.
        
        Returns:
        - float: The dispersion value.
        """
        tree = KDTree(V)
        min_distances, _ = tree.query(S)
        return np.max(min_distances)


    grid = create_grid(stats)
    dispersion_value = dispersion(grid, states)
    dispersion_value
    return KDTree, create_grid, dispersion, dispersion_value, grid


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
