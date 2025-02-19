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
def _(attribution_scores, np, plt, target_data, training_data):
    inspection_idx = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    min_imp_train_sample_idx = []
    max_imp_train_sample_idx = []
    for inspection_id in inspection_idx:
        min_imp_train_sample_idx.append(np.argmin(attribution_scores[:, inspection_id]))
        max_imp_train_sample_idx.append(np.argmax(attribution_scores[:, inspection_id]))

    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')
    ax.scatter(
        target_data["X"][inspection_idx, 1],
        target_data["X"][inspection_idx, 2],
        target_data["X"][inspection_idx, 3],
        c=target_data["y"][inspection_idx],
        cmap="coolwarm",
        alpha=0.4,
        s=10,
        label="Target data")
    ax.scatter(
        training_data["X"][min_imp_train_sample_idx, 1],
        training_data["X"][min_imp_train_sample_idx, 2],
        training_data["X"][min_imp_train_sample_idx, 3],
        c=training_data["y"][min_imp_train_sample_idx],
        cmap="coolwarm",
        alpha=0.4,
        s=10,
        label="High -influence")
    ax.scatter(
        training_data["X"][max_imp_train_sample_idx, 1],
        training_data["X"][max_imp_train_sample_idx, 2],
        training_data["X"][max_imp_train_sample_idx, 3],
        c=training_data["y"][max_imp_train_sample_idx],
        cmap="coolwarm",
        alpha=0.4,
        s=10,
        label="High +influence")
    ax.set_xlabel(r'$v$')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$\omega$')
    plt.title("Important Data Points Inspection")
    plt.legend()
    return (
        ax,
        fig3d,
        inspection_id,
        inspection_idx,
        max_imp_train_sample_idx,
        min_imp_train_sample_idx,
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
