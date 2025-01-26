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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_expert', action='store_true', help='Train expert policy')
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

    # Create D
    X, y = get_dataset_from_model(config, expert, episodes=100)

    # Fit supervised model
    dt = tree.DecisionTreeClassifier(ccp_alpha=0.01)
    dt.fit(X, y)
