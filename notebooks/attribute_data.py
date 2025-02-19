import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Data Attribution for Simple Policy""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(np):
    attribution_scores = np.load("data/scores/BC_policy_scores.npz")["scores"]
    return (attribution_scores,)


@app.cell
def _(attribution_scores):
    attribution_scores
    return


@app.cell
def _(attribution_scores, np):
    np.count_nonzero(attribution_scores)
    return


@app.cell
def _(attribution_scores, np):
    np.prod(attribution_scores.shape)
    return


@app.cell
def _(attribution_scores, np):
    np.count_nonzero(attribution_scores)/np.prod(attribution_scores.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
