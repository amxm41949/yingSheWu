import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import wandb

from env import SnakeEnv
from model import SnakeNet, SnakeMLP

GAMMA = 0.99
LR = 3e-4
EPS_CLIP = 0.2
K_EPOCHS = 10
T_HORIZON = 2048
ENTROPY_COEF = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100000

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = SnakeMLP(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
    
    def update(self, states, actions, log_probs_old, rewards):
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        for _ in range(K_EPOCHS):
            pi = self.policy(states)
            dist = torch.distributions.Categorical(pi)
            new_log_probs = dist.log_prob(actions)
            # breakpoint()
            ratio = (new_log_probs - log_probs_old).exp()
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * returns
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = dist.entropy().mean()
            
            loss = policy_loss - ENTROPY_COEF * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=device))

# 初始化 wandb
wandb.init(project="snake-ppo", config={
    "board_size": 5,
    "num_epochs": 10000,
    "T_HORIZON": 200,
    "gamma": 0.99,
    "learning_rate": 3e-4
})

def train():
    os.makedirs("./snake/checkpoints/", exist_ok=True)

    env = SnakeEnv(5, 1, 5)
    state_dim = env.board_size
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim)
    reward_buffer = []
    rollout_lengths = []

    for episode in range(wandb.config["num_epochs"]):
        state = env.reset()
        memory = []
        episode_reward = 0

        with torch.no_grad():
            for t in range(wandb.config["T_HORIZON"]):
                state_tensor = state.unsqueeze(0).to(device)
                log_prob_dist = agent.policy(state_tensor)
                dist = torch.distributions.Categorical(log_prob_dist)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _ = env.step(action.item())
                memory.append([state, action, log_prob, reward])
                
                episode_reward += reward
                
                if done:
                    rollout_lengths.append(t + 1)
                    break

        states, actions, log_probs, rewards = zip(*memory)
        loss = agent.update(
            torch.stack(states).to(device),  # T, 3, H, W
            torch.stack(actions).squeeze().to(device),  # T
            torch.stack(log_probs).squeeze().to(device),  # T
            torch.stack(rewards).to(device)  # T
        )

        reward_buffer.append(episode_reward)

        # 记录训练数据到 wandb
        if episode % 100 == 0:
            avg_reward = sum(reward_buffer) / len(reward_buffer)
            avg_len = sum(rollout_lengths) / len(rollout_lengths)
            wandb.log({"avg_reward": avg_reward, "avg_length": avg_len})
            
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Len: {avg_len}")

            reward_buffer = []
            rollout_lengths = []

        # 保存模型
        if episode % 1000 == 0:
            model_path = f"./snake/checkpoints/ppo_model_{episode}.pth"
            agent.save(model_path)
            wandb.save(model_path)

if __name__ == "__main__":
    train()