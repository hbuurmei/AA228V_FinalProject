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
    from mpl_toolkits.mplot3d import Axes3D
    return Axes3D, mo, np, plt


@app.cell
def _(np):
    # Load the relevant data
    attribution_scores = np.load("data/scores/BC_policy_scores.npz")["scores"]
    training_data = np.load("data/datasets/expert_data_train.npz")
    target_data = np.load("data/datasets/BC_target_rollouts.npz")
    return attribution_scores, target_data, training_data


@app.cell(hide_code=True)
def _(attribution_scores, np):
    print(f"""
            Number of training data points: {attribution_scores.shape[0]}
            Number of target data points: {attribution_scores.shape[1]}
            Percentage of nonzero elements in DA scores matrix: {np.count_nonzero(attribution_scores) / np.prod(attribution_scores.shape) * 100:.2f}%
    """)
    return


@app.cell
def _(attribution_scores, np, plt):
    plt.hist(np.log10(np.abs(attribution_scores[:, 0])+1e-6))
    plt.show()
    return


@app.cell
def _():
    """
    from matplotlib.lines import Line2D


    inspection_id = 20
    min_idx = np.argsort(attribution_scores[:, inspection_id])[:10]  # top 10 min
    max_idx = np.argsort(attribution_scores[:, inspection_id])[-10:][::-1]  # top 10 max (descending)

    target_label = target_data["y"][inspection_id]

    # Select filtered indices
    if target_label == 0:
        min_filtered_idx = min_idx[training_data["y"][min_idx] == 1]
        max_filtered_idx = max_idx[training_data["y"][max_idx] == 0]
    else:
        min_filtered_idx = min_idx[training_data["y"][min_idx] == 0]
        max_filtered_idx = max_idx[training_data["y"][max_idx] == 1]

    # Colors for target data
    target_colors = 'red' if target_label == 0 else 'blue'

    # Colors for training data
    colors_min = np.where(training_data["y"][min_filtered_idx] == 0, 'red', 'blue')
    colors_max = np.where(training_data["y"][max_filtered_idx] == 0, 'red', 'blue')

    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')

    # Plot all training data faded out in the background for reference
    colors_training = np.where(training_data["y"] == 0, 'red', 'blue')
    ax.scatter(
        training_data["X"][:, 1],
        training_data["X"][:, 2],
        training_data["X"][:, 3],
        c=colors_training,
        alpha=0.01,
        s=3,
        label="Training data",
        marker='o'
    )
    # Plot target data
    ax.scatter(
        target_data["X"][inspection_id, 1],
        target_data["X"][inspection_id, 2],
        target_data["X"][inspection_id, 3],
        c=target_colors,
        alpha=0.85,
        s=60,
        label="Target data",
        marker='X'
    )

    # Plot high-influence points
    ax.scatter(
        training_data["X"][min_filtered_idx, 1],
        training_data["X"][min_filtered_idx, 2],
        training_data["X"][min_filtered_idx, 3],
        c=colors_min,
        alpha=0.95,
        s=60,
        label="High -influence",
        marker='^'
    )

    ax.scatter(
        training_data["X"][max_filtered_idx, 1],
        training_data["X"][max_filtered_idx, 2],
        training_data["X"][max_filtered_idx, 3],
        c=colors_max,
        alpha=0.95,
        s=60,
        label="High +influence",
        marker='^'
    )

    # ax.set_xlim([-0.35, 0.35])
    # ax.set_ylim([-0.035, 0.035])
    # ax.set_zlim([-0.2, 0.2])
    ax.set_xlabel(r'$v$')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$\omega$')
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Target point"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Training data"),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Important points"),
    ]
    plt.legend(handles=legend_elements)
    mo.mpl.interactive(plt.gcf())
    """
    return


@app.cell
def _(attribution_scores, np, plt, target_data, training_data):
    from matplotlib.lines import Line2D


    inspection_id = 5
    min_idx = np.argsort(attribution_scores[:, inspection_id])[:10]  # top 10 min
    max_idx = np.argsort(attribution_scores[:, inspection_id])[-10:][::-1]  # top 10 max (descending)

    target_label = target_data["y"][inspection_id]

    # Select filtered indices
    if target_label == 0:
        min_filtered_idx = min_idx[training_data["y"][min_idx] == 1]
        max_filtered_idx = max_idx[training_data["y"][max_idx] == 0]
    else:
        min_filtered_idx = min_idx[training_data["y"][min_idx] == 0]
        max_filtered_idx = max_idx[training_data["y"][max_idx] == 1]

    plt.figure(figsize=(5, 3))
    plt.scatter(
        training_data["X"][:, 1],
        training_data["X"][:, 3],
        c=training_data["y"],
        cmap="coolwarm",
        alpha=0.002,
        marker="o"
    )
    plt.scatter(
        training_data["X"][min_filtered_idx, 1],
        training_data["X"][min_filtered_idx, 3],
        c=training_data["y"][min_filtered_idx],
        cmap="coolwarm",
        s=70,
        marker="^"
    )
    plt.scatter(
        training_data["X"][max_filtered_idx, 1],
        training_data["X"][max_filtered_idx, 3],
        c=training_data["y"][max_filtered_idx],
        cmap="coolwarm",
        s=70,
        marker="^"
    )
    plt.scatter(
        target_data["X"][inspection_id, 1],
        target_data["X"][inspection_id, 3],
        c=target_data["y"][inspection_id],
        cmap="coolwarm",
        s=90,
        marker="X"
    )
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Target point"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Training data"),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, alpha=0.75, label="Important points"),
    ]
    plt.xlim([-0.7, 0.7])
    plt.xlabel(r"$v\ [\text{m sec}^{-1}]$")
    plt.ylabel(r"$\omega\ [\text{sec}^{-1}]$")
    yticks = plt.yticks()[0]
    plt.yticks(yticks, [f"{np.rad2deg(x):.0f}°" for x in yticks])
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    # plt.savefig("IL_DA_result.svg", format="svg")
    plt.gcf()
    return (
        Line2D,
        inspection_id,
        legend_elements,
        max_filtered_idx,
        max_idx,
        min_filtered_idx,
        min_idx,
        target_label,
        yticks,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### First metric: **entropy**

        First we convert the attribution scores to valid probability, i.e. the probability of training sample $i$ is equal to $p_i=\frac{\left|s_i\right|}{\sum_j\left|s_j\right|}$, where $s_i$ denotes the corresponding score.

        Then, we compute the entropy according to $H=-\sum_{i=1}^N p_i \log p_i$, where $N$ is the size of our training dataset.

        _Higher entropy_: influence is spread across many training points.
        _Lower entropy_: influence is concentrated on a few training points.
        """
    )
    return


@app.cell
def _(attribution_scores, np, target_data):
    def attribution_entropy(scores):
        """
        Compute entropy of attribution scores.
        """
        abs_scores = np.abs(scores)
        probs = abs_scores / np.sum(abs_scores)

        # Avoid log(0) by setting p log p = 0 when p = 0
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        return entropy


    entropies = [attribution_entropy(attribution_scores[:, target_id]) 
                 for target_id in range(target_data["X"].shape[0])]
    return attribution_entropy, entropies


@app.cell(hide_code=True)
def _(entropies, plt):
    plt.boxplot(entropies, vert=False)
    plt.xlabel("Entropy")
    plt.title("Spread of Attribution Entropy Across Targets")
    return


@app.cell(hide_code=True)
def _(entropies, plt):
    plt.hist(entropies, bins=20, edgecolor="black", alpha=0.75)
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attribution Entropy Across Targets")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Second metric:** Gini coefficient**

        We compute the Gini coefficient according to $G=\frac{\sum_{i=1}^N \sum_{j=1}^N\left|s_i-s_j\right|}{2 N \sum_{i=1}^N |s_i|}$.

        _Low coefficient_ ($G=0$): uniform influence across all data points.
        _High coefficient_ ($G=0$): highly concentrated influence.
        """
    )
    return


@app.cell
def _(attribution_scores, np, target_data):
    def gini_coefficient(scores):
        """
        Compute the Gini coefficient of attribution scores.
        """
        abs_scores = np.abs(scores)
        if np.sum(abs_scores) == 0:
            return 0.0

        sorted_scores = np.sort(abs_scores)
        N = len(sorted_scores)
        cumulative_sum = np.cumsum(sorted_scores)

        # Compute Gini using the standard formula
        gini_coeff = (2 / N) * np.sum((np.arange(1, N + 1) / N) * sorted_scores) - (np.sum(sorted_scores) / N)
        gini_coeff /= np.mean(sorted_scores)

        return gini_coeff


    gini_coeffs = [gini_coefficient(attribution_scores[:, target_id]) 
                   for target_id in range(target_data["X"].shape[0])]
    return gini_coefficient, gini_coeffs


@app.cell(hide_code=True)
def _(gini_coeffs, plt):
    plt.boxplot(gini_coeffs, vert=False)
    plt.xlabel("Gini coefficient")
    plt.title("Spread of Attribution Gini Coefficient Across Targets")
    return


@app.cell(hide_code=True)
def _(gini_coeffs, plt):
    plt.hist(gini_coeffs, bins=20, edgecolor="black", alpha=0.75)
    plt.xlabel("Gini coefficient")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attribution Gini Coefficient Across Targets")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Third metric: **effective number of influential data points**

        This metric aim to quantify how many training points effectively contribute to a policy’s decision.
        It's inspired by the inverse participation ratio (IPR) from physics.

        It is computed as $N_{eff} = \\frac{1}{\sum_{i=1}^N p_i^2}$, where $p_i$ is defined as before.
        """
    )
    return


@app.cell
def _(attribution_scores, np, target_data):
    def effective_num_influential_points(scores):
        """
        Compute the effective number of influential training points.
        """
        abs_scores = np.abs(scores)
        if np.sum(abs_scores) == 0:
            return 0.0

        probs = abs_scores / np.sum(abs_scores)
        N_eff = 1 / np.sum(probs ** 2)

        return N_eff


    N_eff_values = [effective_num_influential_points(attribution_scores[:, target_id]) 
                    for target_id in range(target_data["X"].shape[0])]
    return N_eff_values, effective_num_influential_points


@app.cell(hide_code=True)
def _(N_eff_values, plt):
    plt.hist(N_eff_values, bins=20, edgecolor="black", alpha=0.75)
    plt.xlabel(r"$N_{eff}$")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attribution Effective Number of Data Points Across Targets")
    return


if __name__ == "__main__":
    app.run()
