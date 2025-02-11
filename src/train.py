import torch
import argparse
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import sk2torch
from trak import TRAKer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier
from reinforcement_learning import MLPAgent
from imitation_learning import get_dataset_from_model, label_dataset_with_model, ILAgent


def train_expert(config, max_episodes=1000):
    env = gym.make(config["name"], render_mode="human" if config["render"] else None)
    agent = MLPAgent()
    scores = []
    
    for episode in range(max_episodes):
        state = env.reset()[0]
        score = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, terminated)
            agent.experience_replay()
            
            state = next_state
            score += reward

        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Average of last 100 episodes
        
        if episode % 50 == 0:
            print(f'Episode {episode} Score: {score} Average Score: {avg_score:.2f} Exploration: {agent.exploration_rate:.3f}')
        
        # Save model if we achieve the target score
        if avg_score >= config["target_score"]:
            print(f'Environment solved in {episode} episodes!')
            agent.save_model('data/models/expert_policy.pth')
            break
    
    env.close()
    return agent, scores


def policy_rollout(agent, config, N=1, render=False):
    env = gym.make(config["name"], render_mode="human" if render else None)
    total_reward = 0

    for _ in range(N):
        state, _ = env.reset()
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_reward += episode_reward
        
    env.close()
    return total_reward / N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_expert', action='store_true', help='Train expert policy')
    parser.add_argument('--visualize_expert', action='store_true', help='Visualize expert policy rollout')
    parser.add_argument('--IL_method', type=str, choices=['BC', 'AO', 'DA'], required=True,
                        help='Method to use: BC (Behavioral Cloning), AO (Alternating Optimization), or DA (Data Aggregation)')
    parser.add_argument('--classifier_type', type=str, choices=['MLP', 'DT'], required=True,
                        help='Classifier type: MLP (Multi-Layer Perceptron), or DT (Decision Tree)')
    args = parser.parse_args()

    # Define environment configuration
    config = {
        "name": "CartPole-v1",
        "render": False,
        "target_score": 400,
    }

    # Train expert policy
    if args.train_expert:
        expert, scores = train_expert(config)
        expert.exploration_rate = 0
    else:
        expert = MLPAgent(exploration_rate=0)
        expert.load_model("data/models/expert_policy.pth")

    # Visualize expert policy rollout
    if args.visualize_expert:
        policy_rollout(expert, config, render=True)

    # Collect initial dataset from the expert
    X0, y0 = get_dataset_from_model(config, expert, episodes=100)

    # Store this dataset for inspection
    np.savez_compressed("data/expert_dataset.npz", X=X0, y=y0)

    # Initial policy will be using behavior cloning
    if args.classifier_type == "MLP":
        clf0 = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(24, 24, 24))
    elif args.classifier_type == "DT":
        clf0 = DecisionTreeClassifier(ccp_alpha=0.02)
    clf0.fit(X0, y0)
    agent0 = ILAgent(clf0)

    if args.IL_method == "BC":
        # Behavioral Cloning (BC)
        agent = agent0

        if args.classifier_type == "DT":
            text_repr = export_text(agent.dt)
            print("BC with decision tree representation:")
            print(text_repr)
    
    elif args.IL_method == "AO":
        # Alternating Optimization (AO)
        
        # Initialize to behavior cloning policy and tracking variables
        policy = agent0
        best_reward = -np.inf
        best_model = policy

        for _ in tqdm(range(50), desc="AO Iterations"):
            # Collect states using current policy
            X, _ = get_dataset_from_model(config, policy, episodes=100)

            # Get expert labels for visited states
            y = label_dataset_with_model(expert, X)

            # Train updated policy
            if args.classifier_type == "MLP":
                clf = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(24, 24, 24))
            elif args.classifier_type == "DT":
                clf = DecisionTreeClassifier(ccp_alpha=0.02)
            clf.fit(X, y)
            policy = ILAgent(clf)

            # Evaluate and track best policy
            avg_reward = policy_rollout(policy, config, N=100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = policy
        
        agent = best_model

    elif args.IL_method == "DA":
        # Data Aggregation (DA)

        # Initialize to behavior cloning policy and tracking variables
        policy = agent0
        best_reward = -np.inf
        best_model = policy

        for _ in tqdm(range(50), desc="DA Iterations"):
            # Collect states using current policy
            X, _ = get_dataset_from_model(config, policy, episodes=100)

            # Get expert labels for visited states
            y = label_dataset_with_model(expert, X)

            # Aggregate the data
            X = np.concatenate([X0, X])
            y = np.concatenate([y0, y])

            # Train updated policy
            if args.classifier_type == "MLP":
                clf = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(24, 24, 24))
            elif args.classifier_type == "DT":
                clf = DecisionTreeClassifier(ccp_alpha=0.02)
            clf.fit(X, y)
            policy = ILAgent(clf)

            # Evaluate and track best policy
            avg_reward = policy_rollout(policy, config, N=100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = policy
        
        agent = best_model

    else:
        raise ValueError("Invalid method. Please choose BC, AO, or DA.")

    # Evaluate the resulting policy
    avg_reward = policy_rollout(agent, config, N=100)
    print(f"Average reward of {args.IL_method} agent with {args.classifier_type} classifier: {avg_reward:.2f}")

    # Visualize a rollout
    policy_rollout(agent, config, render=True)

    # Convert the trained classifier to a torch model
    torch_model = sk2torch.wrap(agent.clf)

    # Save the trained model
    torch.save(torch_model, "data/models/imitation_policy.pth")

    # Get TRAK scores
    # traker = TRAKer(model=torch_model,
    #                 task='image_classification',  # should also work for general classification tasks
    #                 train_set_size=len(loader_train.dataset))
