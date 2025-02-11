import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Visualizing recoverable states

        We try to visualize the 4d state space, and construct a set of "recoverable" states.
        Later, we will use this to find states that are in theory recoverable, but our policy doesn't manage to recover them.

        To determine "recoverability", we have learned a value function as a sort of "ground truth". For the value function, we consider a discretized state set

        $$
        s_1 \in [-4.8, 4.8]\\
        s_2 \in [-4.0, 4.0]\\
        s_3 \in [-0.5, 0.5]\\
        s_4 \in [-4.0, 4.0]
        $$

        Using the cartpole environment where reward `r=0` for a successful step and `r=-1` for a terminating (failure) step, we create a value function `V[s_1, s_2, s_3, s_4]`, which we try to visualize below.
        We can then define recoverability as having a value close to zero, i.e. there is an action which will not eventually lead us to failure.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import pickle, math
    import marimo as mo
    return math, mo, np, pickle, plt, sp


@app.cell
def _(np, pickle):
    # load value function and discretization grid
    V = np.load("data/qvalues.npy")
    with open('data/grid_vals.pkl', 'rb') as f:
        grid = pickle.load(f)
    return V, f, grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Three value regimes

        We find that the value function has three regimes:

        1. **Immediate termination:** Every action leads to immediate termination. The value is $-5$ because of our discount factor $\gamma=4/5$ and therefore following the Bellman equilibrium $v = \max r + \gamma v$ we have $v = {(1-\gamma)}^{-1} r = -5$.
        2. **Eventual termination:** Here, the states don't terminate immediately, but failure is inevitable, and the value is $\gamma^k \cdot (-5)$ for some number of steps $k$.
        3. **Recovery:** Here we can recover the state and can therefore achieve a value of $0$ (or close to zero, due to Value Iteration convergence).
        """
    )
    return


@app.cell(hide_code=True)
def _(V, plt):
    fig, ax = plt.subplots(1,1)
    ax.hist(V.flatten())
    plt.xlabel("State Value (0 is best)")
    plt.ylabel("Count")
    plt.title("Histogram of recoverable states")
    fig
    return ax, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A slice of recoverability
        Due to the 4d nature of the states, we can't visualize them all.
        However, we can "slice" the data, for example by plotting only values for which the cart is centered ($s_1=0$) and still ($s_2=0$).

        Below we plot the value for the grid of pole angles and angular velocities. Recall that *only* values close to $0$, i.e. dark blue, are recoverable.
        """
    )
    return


@app.cell(hide_code=True)
def _(V, grid, math, np, plt):
    # Fix indices for position and velocity (5,5)
    fixed_i, fixed_j = math.floor(len(grid[0]) / 2), math.floor(len(grid[1]) / 2)
    # fixed_i, fixed_j = 4, 0

    # Create 2D view of value function
    slice_2d = V[fixed_i, fixed_j, :, :]

    # Create meshgrid for angle and angular velocity
    angle_grid, ang_vel_grid = grid[2], grid[3]
    X, Y = np.meshgrid(angle_grid, ang_vel_grid)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, slice_2d.T, cmap="RdYlBu", shading="nearest", vmax=0)
    plt.colorbar(label="State value")
    plt.xlabel("Pole Angle")
    plt.ylabel("Pole Angular Velocity")
    xticks = plt.xticks()[0]
    plt.xticks(xticks, [f"{np.rad2deg(x):.2f}°" for x in xticks])
    plt.title(
        f"Recoverable States (pos={grid[0][fixed_i]:.2f}, vel={grid[1][fixed_j]:.2f})"
    )
    plt.show()
    return (
        X,
        Y,
        ang_vel_grid,
        angle_grid,
        fixed_i,
        fixed_j,
        slice_2d,
        xticks,
    )


@app.cell(hide_code=True)
def _(V, grid, np):
    import sys; sys.path.append("src")
    from set_utils import is_in_hull

    pts = []
    for indices in np.ndindex(V.shape):
        if V[indices] > -0.1:
            pts.append(np.array([
                grid[0][indices[0]],
                grid[1][indices[1]],
                grid[2][indices[2]],
                grid[3][indices[3]],]))
    return indices, is_in_hull, pts, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How much of the space is recoverable?

        We can now compare the volume of the convex hull around all recoverable states to the total volume of considered states.
        Interestingly, we find that only about 10% (!) of the states are recoverable. This may require further investigation.
        """
    )
    return


@app.cell
def _(pts):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    total_vol = (2*4.8) * (2*4.0) * (2*0.5) * (2*4.0)
    print(f"""
    hull vol: {hull.volume:.2f}
    total vol: {total_vol:.2f}
    fraction: {hull.volume / total_vol:.2f}
    """)
    return ConvexHull, hull, total_vol


if __name__ == "__main__":
    app.run()
