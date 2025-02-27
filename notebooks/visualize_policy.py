import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Visualizing the policy
    We visualize the expert and imitation agent policies by visualizing each agents' decision boundary through a contour plot across a state meshgrid. Blue indicates action 0 (move left), and yellow indicates action 1 (move right).
    We also plot a scatter plot of imitation agent training data located near the tuple $(x, x')$ (with range $\epsilon$).

    We initially notice that the policies seem very similar.
    However, as we move away from the initial centered state, we notice that the expert agent starts to have nonsensicle decision boundaries, whereas the imitation agent stays the same.
    In particular, for example for $(x=-1.8, x'=1.0)$ we find that a new decision boundary appears, which leads to the inverted pendulum falling. 
    We conclude that the imitation policy actually generalizes better, and overfits less to the original data than the expert agent, despite having the same architecture.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, slider_log10eps, slider_x, slider_xdot):
    mo.vstack([
        mo.md(f"x     : {slider_x}"),
        mo.md(f"x'    : {slider_xdot}"),
        mo.md(f"logeps: {slider_log10eps} -> eps: {np.pow(10, slider_log10eps.value):.2f}")
    ], align="start")
    return


@app.cell(hide_code=True)
def _(agent_exp, agent_imi, mo, plot_agent):
    mo.hstack([
        mo.vstack([mo.md("# Expert agent"), plot_agent(agent_exp)]),
        mo.vstack([mo.md("# Imitation agent"), plot_agent(agent_imi)]),
    ])
    return


@app.cell(hide_code=True)
def _(mo, np):
    slider_x    = mo.ui.slider(steps=np.linspace(-2.4, 2.4, 51), show_value=True, value=0)
    slider_xdot = mo.ui.slider(steps=np.linspace(-3, 3, 51), show_value=True, value=0)
    slider_log10eps = mo.ui.slider(start=-3, step=0.25, stop=-0, show_value=True, value=-1)

    mo.md("### Backend")
    return slider_log10eps, slider_x, slider_xdot


@app.cell(hide_code=True)
def _():
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import pickle, math
    import marimo as mo

    import yaml
    import sys; sys.path.append("src")
    from utils import load_config
    from imitation_learning import ILAgent
    from reinforcement_learning import RLAgent

    training_data = np.load("data/datasets/expert_data_train.npz")["X"]

    env_config = load_config("config/env/cartpole.yaml")
    print(f"Current environment: {env_config["name"]}")
    return (
        ILAgent,
        RLAgent,
        env_config,
        load_config,
        math,
        mo,
        np,
        pickle,
        plt,
        sp,
        sys,
        training_data,
        yaml,
    )


@app.cell(hide_code=True)
def _(ILAgent, load_config):
    agent_imi = ILAgent(load_config("config/train/il_agent_cartpole.yaml"))
    agent_imi.load_model("data/models/BC_policy.pt")
    agent_imi.device = "cpu"
    agent_imi.model.to(agent_imi.device)
    agent_imi
    return (agent_imi,)


@app.cell(hide_code=True)
def _(RLAgent, load_config):
    agent_exp = RLAgent(load_config("config/train/rl_agent_cartpole.yaml"))
    agent_exp.load_model("data/models/expert_policy.pt")
    agent_exp.device = "cpu"
    agent_exp.model.to(agent_exp.device)
    agent_exp.exploration_rate=0
    agent_exp
    return (agent_exp,)


@app.cell(hide_code=True)
def _(np, plt, slider_log10eps, slider_x, slider_xdot, training_data):
    def plot_agent(agent):
        # Fix indices for position and velocity (5,5)
        #fixed_i, fixed_j = math.floor(len(grid[0]) / 2), math.floor(len(grid[1]) / 2)
        s1 = slider_x.value
        s2 = slider_xdot.value
        
        # Create meshgrid for angle and angular velocity
        angle_grid = np.linspace(np.deg2rad(-12), np.deg2rad(12), 20)
        ang_vel_grid = np.linspace(np.deg2rad(-35), np.deg2rad(35), 20)#grid[2], grid[3]
        X, Y = np.meshgrid(angle_grid, ang_vel_grid)
        Z = np.array([[agent.act(np.array([s1, s2, xi, yi])) for xi, yi in zip(x_row, y_row)] 
                      for x_row, y_row in zip(X, Y)])
        
        # Plot
        plt.figure(figsize=(5, 3))
        plt.contourf(X, Y, Z)
        eps = np.pow(10, slider_log10eps.value)
        idx = np.linalg.norm(training_data[:, 0:2] - np.array([s1, s2]), axis=1) < eps
        if np.sum(idx) > 0:
            plt.scatter(training_data[idx, :][:, 2], training_data[idx, :][:, 3], color="red")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\omega\ [\text{sec}^{-1}]$")
        #xticks = np.deg2rad(np.linspace(-24, 24, 9))
        xticks = np.deg2rad(np.linspace(-12, 12, 9))
        yticks = np.deg2rad(np.linspace(-230/4, 230/4, 9))
        plt.xticks(xticks, [f"{np.rad2deg(x):.0f}°" for x in xticks])
        plt.yticks(yticks, [f"{np.rad2deg(x):.0f}°" for x in yticks])
        plt.title(f"Decision boundary @ x={s1:.2f}, x'={s2:.2f}")
        plt.xlim(np.deg2rad(-12), np.deg2rad(12))
        plt.ylim(np.deg2rad(-3*12), np.deg2rad(3*12))
        return plt.gcf()
    return (plot_agent,)


@app.cell(hide_code=True)
def _(np):
    def middle(x, y):
        np.round((x+y)/2)
    return (middle,)


if __name__ == "__main__":
    app.run()
