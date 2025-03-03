import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from trak import TRAKer
from utils import load_config, MLP


def run_data_attribution(model_name):
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    if model_name == "dynamics_learner":
        data_train = np.load("data/datasets/dl_data_train.npz")
        data_target = np.load("data/datasets/dl_target_rollouts.npz")
        X_train = torch.from_numpy(data_train["X"]).float()
        A_train = torch.from_numpy(data_train["A"]).long()
        Xp_train = torch.from_numpy(data_train["Xp"]).float()
        inputs_train = torch.cat((X_train, A_train), dim=1)
        labels_train = Xp_train
        X_target = torch.from_numpy(data_target["X"]).float()
        A_target = torch.from_numpy(data_target["A"]).long()
        Xp_target = torch.from_numpy(data_target["Xp"]).float()
        inputs_target = torch.cat((X_target, A_target), dim=1)
        labels_target = Xp_target
        train_dataset = TensorDataset(inputs_train, labels_train)
        target_dataset = TensorDataset(inputs_target, labels_target)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
        target_loader = DataLoader(target_dataset, batch_size=50, shuffle=False)

    elif model_name == "IL_agent":
        data_train = np.load("data/datasets/expert_data_train.npz")
        data_target = np.load("data/datasets/BC_target_rollouts.npz")
        X_train = torch.from_numpy(data_train["X"]).float()
        A_train = torch.from_numpy(data_train["A"]).long()
        X_target = torch.from_numpy(data_target["X"]).float()
        A_target = torch.from_numpy(data_target["A"]).long()
        train_dataset = TensorDataset(X_train, A_train)
        target_dataset = TensorDataset(X_target, A_target)
        train_loader = DataLoader(train_dataset, batch_size=200, shuffle=False)
        target_loader = DataLoader(target_dataset, batch_size=100, shuffle=False)

    # Get the model checkpoints
    checkpoints_dir = "data/models/checkpoints"
    ckpt_files = list(Path(checkpoints_dir).rglob('*.pt'))
    ckpts = [torch.load(ckpt, map_location='cpu', weights_only=True) for ckpt in ckpt_files]

    # Load a model to evaluation
    il_agent_config = load_config("config/train/il_agent_cartpole.yaml")
    model = MLP(input_dim=il_agent_config["input_dim"],
                hidden_dims=il_agent_config["hidden_dims"],
                output_dim=il_agent_config["output_dim"]).to(device).eval()

    # Store results to folder
    trak_results_dir = "data/trak_results"
    experiment_name = f"{il_agent_config["method"]}_policy"

    # Initialize TRAKer object
    traker = TRAKer(model=model,
                    task='image_classification',  # NOTE: also works for non-image data
                    train_set_size=len(train_dataset),
                    save_dir=trak_results_dir,
                    device=str(device),
                    use_half_precision=False)

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        # TRAKer loads the provided checkpoint and also associates
        # the provided (unique) model_id with the checkpoint.
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in train_loader:
            # TRAKer computes features corresponding to the batch of examples,
            # using the checkpoint loaded above.
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    # Tells TRAKer that we've given it all the information, at which point
    # TRAKer does some post-processing to get ready for the next step
    traker.finalize_features()

    # Get the scores
    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(exp_name=experiment_name,
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(target_dataset))
        for batch in target_loader:
            traker.score(batch=batch, num_samples=batch[0].shape[0])
    scores = traker.finalize_scores(exp_name=experiment_name)
    print("Shape of scores matrix:", scores.shape)

    # Save the scores
    np.savez_compressed(f"data/scores/{experiment_name}_scores.npz", scores=scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', action='store_true', help='Attribution for dynamics learner, default is for IL agent')
    args = parser.parse_args()
    if args.dynamics:
        model_name = "dynamics_learner"
    else:
        model_name = "IL_agent"
    print(f"Running attribution for {model_name}")

    run_data_attribution(model_name)
