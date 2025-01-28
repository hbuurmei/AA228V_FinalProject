import argparse
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from sklearn import tree
from reinforcement_learning import MLPAgent
from imitation_learning import get_dataset_from_model, label_dataset_with_model, DTAgent


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
    parser.add_argument('--method', type=str, choices=['BC', 'AO', 'DA'], required=True,
                        help='Method to use: BC (Behavioral Cloning), AO (Alternating Optimization), or DA (Data Aggregation)')
    args = parser.parse_args()

    config = {
        "name": "CartPole-v1",
        "render": False,
        "target_score": 400,
    }
    if args.train_expert:
        expert, scores = train_expert(config)
        expert.exploration_rate = 0
    else:
        expert = MLPAgent(exploration_rate=0)
        expert.load_model("data/models/expert_policy.pth")

    if args.visualize_expert:
        policy_rollout(expert, config, render=True)

    # Collect initial dataset from the expert
    X0, y0 = get_dataset_from_model(config, expert, episodes=100)

    # Initial policy will be using behavior cloning
    dt0 = tree.DecisionTreeClassifier(ccp_alpha=0.02)
    dt0.fit(X0, y0)
    dt_agent0 = DTAgent(dt0)
    
    if args.method == "BC":
        # Behavioral Cloning (BC)
        bc_agent = dt_agent0
        text_repr = tree.export_text(bc_agent.dt)
        print("BC with decision tree representation:")
        print(text_repr)

        # Evaluate decision tree policy
        avg_reward = policy_rollout(bc_agent, config, N=100)
        print(f"Average reward of decision tree agent: {avg_reward:.2f}")

        # Visaulize a rollout
        policy_rollout(bc_agent, config, render=True)
    
    elif args.method == "AO":
        # Alternating Optimization (AO)
        
        # Initialize to behavior cloning policy and tracking variables
        policy = dt_agent0
        best_reward = -np.inf
        best_model = policy

        for _ in tqdm(range(50), desc="AO Iterations"):
            # Collect states using current policy
            X, _ = get_dataset_from_model(config, policy, episodes=100)

            # Get expert labels for visited states
            y = label_dataset_with_model(expert, X)

            # Train updated policy
            dt = tree.DecisionTreeClassifier(ccp_alpha=0.02)
            dt.fit(X, y)
            policy = DTAgent(dt)

            # Evaluate and track best policy
            avg_reward = policy_rollout(policy, config, N=100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = policy
        
        ao_agent = best_model

        # Evaluate the final policy
        avg_reward = policy_rollout(ao_agent, config, N=100)
        print(f"Average reward of the final policy: {avg_reward:.2f}")

        # Visaulize a rollout
        policy_rollout(ao_agent, config, render=True)

    elif args.method == "DA":
        # Data Aggregation (DA)

        # Initialize to behavior cloning policy and tracking variables
        policy = dt_agent0
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
            dt = tree.DecisionTreeClassifier(ccp_alpha=0.02)
            dt.fit(X, y)
            policy = DTAgent(dt)

            # Evaluate and track best policy
            avg_reward = policy_rollout(policy, config, N=100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = policy
        
        da_agent = best_model

        # Evaluate the final policy
        avg_reward = policy_rollout(da_agent, config, N=100)
        print(f"Average reward of the final policy: {avg_reward:.2f}")

        # Visaulize a rollout
        policy_rollout(da_agent, config, render=True)

    else:
        raise ValueError("Invalid method. Please choose BC, AO, or DA.")
