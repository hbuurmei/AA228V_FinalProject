import argparse
from reinforcement_learning import RLAgent, train_rl_agent
from imitation_learning import ILAgent, train_il_agent
from dynamics_learning import DynamicsLearner, train_dynamics_learner
from utils import load_config, policy_rollout


def train_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_expert', action='store_true', help='Train expert policy')
    parser.add_argument('--train_student', action='store_true', help='Train student policy')
    parser.add_argument('--train_dynamics_learner', action='store_true', help='Train dynamics learner')
    parser.add_argument('--visualize_expert', action='store_true', help='Visualize expert policy rollout')
    args = parser.parse_args()

    # Load env, RL agent, IL agent and DL configs
    env_config = load_config("config/env/cartpole.yaml")
    rl_agent_config = load_config("config/train/rl_agent_cartpole.yaml")
    il_agent_config = load_config("config/train/il_agent_cartpole.yaml")
    dl_config = load_config("config/train/dynamics_learner_cartpole.yaml")

    # Train expert policy
    if args.train_expert:
        expert, _ = train_rl_agent(rl_agent_config, env_config)
        expert.exploration_rate = 0
    else:
        expert = RLAgent(rl_agent_config)
        expert.exploration_rate = 0
        expert.load_model("data/models/expert_policy.pt")

    # Visualize expert policy rollout
    if args.visualize_expert:
        policy_rollout(expert, env_config, render=True)
    
    # Train imitation learning agent
    if args.train_student:
        # Example of how to use extra_data_config
        # Uncomment and modify as needed
        extra_data_config = None
        
        # Example: Add synthetic data around a specific state with "push right" action
        # extra_data_config = {
        #     'centroid': [0, 0, 0.1, 0],  # [x, x_dot, theta, theta_dot] - slight tilt
        #     'eps': 0.05,                  # Small region around the centroid
        #     'shape': 'spherical',         # Sample from a sphere
        #     'dim': 4,                     # State dimension for cartpole
        #     'label': 1,                   # Action 1 (push right)
        #     'num_samples': 200            # Number of synthetic samples
        # }
        
        # Pass the extra_data_config as a separate parameter
        agent, _ = train_il_agent(il_agent_config, expert, env_config, extra_data_config)
    else:
        agent = ILAgent(il_agent_config)
        agent.load_model(f"data/models/{il_agent_config["method"]}_policy.pt")

    # Train dynamics learner
    if args.train_dynamics_learner:
        train_dynamics_learner(dl_config, env_config)
    else:
        dl = DynamicsLearner(dl_config)
        dl.load_model(f"data/models/dynamics_learner.pt")

if __name__ == "__main__":
    train_models()
