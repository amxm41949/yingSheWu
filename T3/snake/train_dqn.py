import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import wandb

from env import SnakeEnv
from model import ReplayBuffer, SnakeMLP, SnakeNet

# 超参数
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0  
EPSILON_END = 0.05  
EPSILON_DECAY = 5000 
REPLAY_BUFFER_SIZE = 10000  
BATCH_SIZE = 64 
TARGET_UPDATE = 100  
EPOCHS = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_dim, action_dim):
        self.q_net = SnakeNet(state_dim, action_dim).to(device)
        self.target_q_net = SnakeNet(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 初始化目标网络
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.action_dim = action_dim

    def select_action(self, state):
        """ϵ-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1) 
        else:
            with torch.no_grad():
                state_tensor = state.to(device).unsqueeze(0)
                q_values = self.q_net(state_tensor)
                return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return  
        
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        # 计算 Q 目标值
        with torch.no_grad():
            target_q_values = self.target_q_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + GAMMA * target_q_values * (1 - dones)  # Q-learning 目标值

        # 计算当前 Q 值
        q_values = self.q_net(states).gather(1, actions)

        # 计算损失
        loss = nn.MSELoss()(q_values, targets)

        # 反向传播更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())

# 初始化 wandb
wandb.init(project="snake-dqn", config={
    "board_size": 5,
    "num_epochs": 10000,
    "epsilon_decay": EPSILON_DECAY,
    "gamma": GAMMA,
    "learning_rate": LR
})

def train():
    os.makedirs("./snake_dqn/checkpoints/", exist_ok=True)

    seed = 114514
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = SnakeEnv(5, 1, 5)
    state_dim = env.board_size
    action_dim = env.action_space.n
    
    agent = Agent(state_dim, action_dim)
    reward_buffer = []
    rollout_lengths = []

    for episode in range(wandb.config["num_epochs"]):
        state = env.reset()
        episode_reward = 0

        for t in range(500): 
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.update()

            if done:
                rollout_lengths.append(t + 1)
                break

        # epsilon decay
        agent.epsilon = max(EPSILON_END, agent.epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)

        reward_buffer.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = sum(reward_buffer) / len(reward_buffer)
            avg_len = sum(rollout_lengths) / len(rollout_lengths)
            wandb.log({"avg_reward": avg_reward, "avg_length": avg_len})
            
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Len: {avg_len}")

            reward_buffer = []
            rollout_lengths = []

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 1000 == 0:
            model_path = f"./snake/checkpoints/dqn_model_{episode}.pth"
            agent.save(model_path)
            wandb.save(model_path)

if __name__ == "__main__":
    train()
