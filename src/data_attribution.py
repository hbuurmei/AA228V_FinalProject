import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from trak import TRAKer
from utils import load_config, MLP


def main():
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)
    print(f"Using device: {device}")
    
    # Load the expert data
    expert_data_train = np.load("data/datasets/expert_data_train.npz")
    expert_data_test = np.load("data/datasets/expert_data_test.npz")
    X_train = torch.from_numpy(expert_data_train["X"]).float()
    y_train = torch.from_numpy(expert_data_train["y"]).long()
    X_test = torch.from_numpy(expert_data_test["X"]).float()
    y_test = torch.from_numpy(expert_data_test["y"]).long()
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    # Initialize TRAKer object
    traker = TRAKer(model=model,
                    task='image_classification',  # NOTE: also works for non-image data
                    train_set_size=len(train_dataset),
                    save_dir=trak_results_dir,
                    device=device)

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
        traker.start_scoring_checkpoint(exp_name=f"{il_agent_config["method"]}_policy",
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(test_dataset))
        for batch in test_loader:
            traker.score(batch=batch, num_samples=batch[0].shape[0])
    scores = traker.finalize_scores(exp_name=f"{il_agent_config["method"]}_policy")
    print(type(scores), scores.shape)

    # Save the scores


if __name__ == "__main__":
    main()
