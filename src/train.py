import argparse
import numpy as np
import gymnasium as gym
from sklearn import tree
from reinforcement_learning import MLPAgent
from imitation_learning import get_dataset_from_model


def train_expert(config, n_episodes=1000):
    if config["render"]:
        env = gym.make(config["name"], render_mode="human")
    else:
        env = gym.make(config["name"], render_mode=None)
    agent = MLPAgent()
    scores = []
    
    for episode in range(n_episodes):
        state = env.reset()[0]
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.experience_replay()
            
            state = next_state
            score += reward
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Average of last 100 episodes
        
        if episode % 10 == 0:
            print(f'Episode {episode} Score: {score} Average Score: {avg_score:.2f} Exploration: {agent.exploration_rate:.2f}')
        
        # Save model if we achieve the target score
        if avg_score >= config["episode_max_score"]:
            print(f'Environment solved in {episode} episodes!')
            agent.save_model('expert_policy.pth')
            break
    
    env.close()
    return agent, scores


def single_rollout(agent, config, render=False):
    if render:
        env = gym.make(config["name"], render_mode="human")
    else:
        env = gym.make(config["name"], render_mode=None)
    state = env.reset()[0]
    done = False
    while not done:
        action = agent.act(state)
        next_state, _, done, _, _ = env.step(action)
        state = next_state
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_expert', action='store_true', help='Train expert policy')
    parser.add_argument('--visualize_expert', action='store_true', help='Visualize expert policy rollout')
    args = parser.parse_args()
    
    config = {
        "name": "CartPole-v1",
        "render": False,
        "episode_max_score": 195,
    }
    if args.train_expert:
        agent, scores = train_expert(config)
        agent.exploration_rate = 0
    else:
        expert = MLPAgent(exploration_rate=0)
        expert.load_model("data/models/expert_policy.pth")

    if args.visualize_expert:
        single_rollout(expert, config, render=True)

    # Create D
    X, y = get_dataset_from_model(config, expert, episodes=100)

    # Fit decision tree to expert data
    dt = tree.DecisionTreeClassifier(ccp_alpha=0.02)
    dt.fit(X, y)
    text_repr = tree.export_text(dt)
    print("IL with decision tree representation:")
    print(text_repr)
