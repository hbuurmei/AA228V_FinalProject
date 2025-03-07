import numpy as np
from utils import load_config
from dynamics_learning import train_dynamics_learner
from data_attribution import run_data_attribution


def spread_measure(states):
    """
    Compute det of covariance matrix of states as a measure of spread.
    """
    cov_matrix = np.cov(states, rowvar=False)  # each columns represents a variable (state)
    det_cov = np.linalg.det(cov_matrix)

    # Scale according to number of states
    det_cov_scaled = det_cov ** (1 / (2 * states.shape[1]))

    return det_cov_scaled


def run_ood_experiment():
    """
    Run the out-of-distribution evaluation experiment.
    """
    # We run the experiment for a few different model sizes
    model_sizes = [
        [16]*2,  # small
        [64]*3,  # medium
        [128]*4  # large
    ]

    # Load env and default DL configs
    env_config = load_config("config/env/cartpole.yaml")
    dl_default_config = load_config("config/train/dynamics_learner_cartpole.yaml")

    for model_size in model_sizes:
        dl_config = dl_default_config.copy()
        dl_config["hidden_dims"] = model_size
        model_name = f"dynamics_learner_{'_'.join(str(size) for size in model_size)}"

        # Train dynamics learner
        train_dynamics_learner(dl_config, env_config, model_name=model_name, save_model=True)

        # Compute DA scores for OOD target data
        run_data_attribution("dynamics_learner", model_name)
        scores = np.load(f"data/scores/{model_name}_scores.npz")["scores"]

        # For each column, get the top 10 indices with largest absolute values
        top_indices = np.argsort(np.abs(scores), axis=0)[-10:]
        training_data = np.load("data/datasets/dl_data_train.npz")
        training_states = training_data["X"]
        target_data = np.load("data/datasets/dl_target_rollouts.npz")
        target_states = target_data["X"]

        # Shape: (10, num_states, num_target_data)
        important_states = np.zeros((10, target_states.shape[1], target_states.shape[0]))
        for i in range(target_states.shape[0]):
            indices = top_indices[:, i].squeeze().tolist()
            important_states[:, :, i] = training_states[indices]
        
        # Get spread measure of important states
        spreads = np.zeros(target_states.shape[0])
        for i in range(target_states.shape[0]):
            spreads[i] = spread_measure(important_states[:, :, i])
        print(f"Model size {model_size}: Mean spread of important states according to DA is {spreads.mean():.6f}")


if __name__ == "__main__":
    run_ood_experiment()
