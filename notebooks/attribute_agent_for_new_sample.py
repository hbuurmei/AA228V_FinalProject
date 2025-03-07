import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from pathlib import Path
    from tqdm import tqdm
    from trak import TRAKer
    import matplotlib.pyplot as plt
    import sys; sys.path.append("src")
    from utils import load_config, load_extra_data_config, make_extra_data, MLP
    from imitation_learning import ILAgent
    return (
        DataLoader,
        ILAgent,
        MLP,
        Path,
        TRAKer,
        TensorDataset,
        load_config,
        load_extra_data_config,
        make_extra_data,
        mo,
        np,
        plt,
        sys,
        torch,
        tqdm,
    )


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(load_extra_data_config, make_extra_data, torch):
    # Load extra data configuration
    extra_data_config = load_extra_data_config("config/extra_data/cartpole_tilt_left.yaml")
    print(f"Loaded extra data config: {extra_data_config.num_samples} samples at {extra_data_config.centroid}")

    # Generate synthetic data
    X_extra, A_extra = map(torch.tensor, make_extra_data(extra_data_config))
    print(f"Generated {len(X_extra)} synthetic data points with action label: {extra_data_config.label}")
    return A_extra, X_extra, extra_data_config


@app.cell
def _(A_extra, DataLoader, TensorDataset, X_extra, np, torch):
    data_train = np.load("data/datasets/expert_data_train.npz")
    data_target = np.load("data/datasets/BC_target_rollouts.npz")
    X_train = torch.from_numpy(data_train["X"]).float()
    A_train = torch.from_numpy(data_train["A"]).long()

    X_train = torch.concatenate([X_train, X_extra])
    A_train = torch.concatenate([A_train, A_extra])

    X_target = torch.from_numpy(data_target["X"]).float()
    A_target = torch.from_numpy(data_target["A"]).long()
    train_dataset = TensorDataset(X_train, A_train)
    target_dataset = TensorDataset(X_target, A_target)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=False)
    return (
        A_target,
        A_train,
        X_target,
        X_train,
        data_target,
        data_train,
        target_dataset,
        target_loader,
        train_dataset,
        train_loader,
    )


@app.cell
def _(Path, torch):
    checkpoints_dir = "data/models/BC_policy_ckpts"
    ckpt_files = list(Path(checkpoints_dir).rglob('*.pt'))
    ckpts = [torch.load(ckpt, map_location='cpu', weights_only=True) for ckpt in ckpt_files]
    return checkpoints_dir, ckpt_files, ckpts


@app.cell
def _(MLP, device, load_config):
    il_agent_config = load_config("config/train/il_agent_cartpole.yaml")
    model = MLP(input_dim=il_agent_config["input_dim"],
                hidden_dims=il_agent_config["hidden_dims"],
                output_dim=il_agent_config["output_dim"]).to(device).eval()
    return il_agent_config, model


@app.cell
def _():
    # Store results to folder
    trak_results_dir = f"data/trak_results/IL_agent"
    experiment_name = "IL_agent"
    return experiment_name, trak_results_dir


@app.cell
def _(TRAKer, device, model, train_dataset, trak_results_dir):
    # Initialize TRAKer object
    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=len(train_dataset),
                    save_dir=trak_results_dir,
                    device=str(device),
                    use_half_precision=False)
    return (traker,)


@app.cell
def _(ckpts, device, tqdm, train_loader, traker):
    for model_id, ckpt in enumerate(tqdm(ckpts)):
        # TRAKer loads the provided checkpoint and also associates
        # the provided (unique) model_id with the checkpoint.
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in train_loader:
            # TRAKer computes features corresponding to the batch of examples,
            # using the checkpoint loaded above.
            batch_dev = [batch[0].to(device), batch[1].to(device)]
            traker.featurize(batch=batch_dev, num_samples=batch[0].shape[0])

    # Tells TRAKer that we've given it all the information, at which point
    # TRAKer does some post-processing to get ready for the next step
    traker.finalize_features()
    return batch, batch_dev, ckpt, model_id


@app.cell
def _(train_loader):
    next(iter(train_loader))[1].shape
    return


@app.cell
def _(np, torch):
    batch_ref = [torch.tensor([[0.0, 0.0, np.deg2rad(9), np.deg2rad(15)]]).float(), torch.tensor([1]).long()]
    return (batch_ref,)


@app.cell
def _(batch_ref, ckpts, device, experiment_name, tqdm, traker):
    for model_id_, ckpt_ in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(exp_name=experiment_name,
                                        checkpoint=ckpt_,
                                        model_id=model_id_,
                                        num_targets=len(batch_ref[1]))
        batch_ref_dev = [batch_ref[0].to(device), batch_ref[1].to(device)]
        traker.score(batch=batch_ref_dev, num_samples=batch_ref[0].shape[0])
    scores = traker.finalize_scores(exp_name=experiment_name)
    return batch_ref_dev, ckpt_, model_id_, scores


@app.cell
def _(np, scores):
    np.min(scores), np.max(scores)
    return


@app.cell
def _(np, plt, scores):
    plt.hist(np.log10(np.abs(scores)+1e-7))
    plt.gcf()
    return


@app.cell
def _(np, scores):
    np.sort(scores[:, 0])
    return


@app.cell
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


@app.cell
def _(ILAgent, load_config):
    agent_imi = ILAgent(load_config("config/train/il_agent_cartpole.yaml"))
    agent_imi.load_model("data/models/BC_policy.pt")
    agent_imi.device = "cpu"
    agent_imi.model.to(agent_imi.device)
    agent_imi
    return (agent_imi,)


@app.cell
def _(mo, np):
    slider_x    = mo.ui.slider(steps=np.linspace(-2.4, 2.4, 51), show_value=True, value=0)
    slider_xdot = mo.ui.slider(steps=np.linspace(-3, 3, 51), show_value=True, value=0)
    slider_log10eps = mo.ui.slider(start=-3, step=0.25, stop=-0, show_value=True, value=-1)
    return slider_log10eps, slider_x, slider_xdot


@app.cell(hide_code=True)
def _(mo, np, slider_log10eps, slider_x, slider_xdot):
    mo.vstack([
        mo.md(f"x     : {slider_x}"),
        mo.md(f"x'    : {slider_xdot}"),
        mo.md(f"logeps: {slider_log10eps} -> eps: {np.pow(10, slider_log10eps.value):.2f}")
    ], align="start")
    return


@app.cell
def _(agent_imi, batch_ref, np, plot_agent, plt, scores, training_data):
    fig = plot_agent(agent_imi)
    plt.scatter(batch_ref[0][0, 2:3], batch_ref[0][0, 3:4], marker="x")

    idx_scores_to_plot = np.argsort(np.abs(scores[:, 0]))[-20:]
    plt.scatter(training_data[idx_scores_to_plot, 2:3], training_data[idx_scores_to_plot, 3:4], c=scores[idx_scores_to_plot, 0], marker='s')
    plt.colorbar()
    plt.gcf()
    return fig, idx_scores_to_plot


@app.cell
def _(batch_ref):
    batch_ref[0][0, 2:3]
    return


@app.cell
def _(X_train, load_config):
    training_data = X_train

    env_config = load_config("config/env/cartpole.yaml")
    return env_config, training_data


@app.cell
def _(np):
    np.rad2deg(0.5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
