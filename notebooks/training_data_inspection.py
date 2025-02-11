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
def _():
    # Import training data

    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
