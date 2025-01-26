import argparse
import numpy as np
import gymnasium as gym
from sklearn import tree
from reinforcement_learning import MLPAgent
from imitation_learning import get_dataset_from_model, DTAgent


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
            agent.save_model('expert_policy.pth')
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
    args = parser.parse_args()

    config = {
        "name": "CartPole-v1",
        "render": False,
        "target_score": 450,
    }
    if args.train_expert:
        expert, scores = train_expert(config)
        expert.exploration_rate = 0
    else:
        expert = MLPAgent(exploration_rate=0)
        expert.load_model("data/models/expert_policy.pth")

    if args.visualize_expert:
        policy_rollout(expert, config, render=True)

    # Generate D from expert policy
    X, y = get_dataset_from_model(config, expert, episodes=100)

    # Behavior cloning with decision tree as student policy
    dt = tree.DecisionTreeClassifier(ccp_alpha=0.02)
    dt.fit(X, y)
    text_repr = tree.export_text(dt)
    print("BC with decision tree representation:")
    print(text_repr)
    dt_agent = DTAgent(dt)

    # Evaluate decision tree policy
    avg_reward = policy_rollout(dt_agent, config, N=100)
    print(f"Average reward of decision tree agent: {avg_reward:.2f}")
