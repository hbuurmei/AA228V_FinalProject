import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sys; sys.path.append("src")
    from dynamics_learning import DynamicsLearner
    from utils import load_config
    return DynamicsLearner, load_config, mo, np, plt, sys


@app.cell
def _(DynamicsLearner, load_config):
    # Import the learned dynamics model
    dl_config = load_config("config/train/dynamics_learner_cartpole.yaml")
    dynamics_learner = DynamicsLearner(dl_config).load_model("data/models/dynamics_learner.pt")
    return dl_config, dynamics_learner


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
