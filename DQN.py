import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')   # 无交互后端，只保存图片，避免报错

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# 1.神经网络（稍大，但仍只有2层）
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.net(x)

# 2.经验回放池（不变）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    def __len__(self):
        return len(self.buffer)

# 3.超参数（调优）
ENV_NAME = 'CartPole-v1'
EPISODES = 500           # 增加训练回合
BATCH_SIZE = 64          # 适中
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998        # 缓慢衰减（每回合乘一次）
TAU = 0.005              # 软更新系数（使用软更新更稳定）
LR = 1e-3
MEMORY_CAPACITY = 10000
TARGET_UPDATE = 10       # 每10步软更新一次（实际软更新是每步都做，但这里频率不影响）

env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(obs_dim, act_dim).to(device)
target_net = DQN(obs_dim, act_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(MEMORY_CAPACITY)

eps = EPS_START
episode_rewards = []

# 4.训练循环
for episode in range(EPISODES):
    state, _ = env.reset()
    state = state.astype(np.float32)
    total_reward = 0
    for step in range(500):
        # ε-greedy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                action = q.argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = next_state.astype(np.float32) if not done else np.zeros_like(state)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # 训练
        if len(buffer) >= BATCH_SIZE:
            s, a, r, ns, d = buffer.sample(BATCH_SIZE)
            s = torch.tensor(s, device=device)
            a = torch.tensor(a, device=device).unsqueeze(1)
            r = torch.tensor(r, device=device)
            ns = torch.tensor(ns, device=device)
            d = torch.tensor(d, device=device)

            # 当前 Q 值
            q_vals = policy_net(s).gather(1, a).squeeze(1)
            # 目标 Q 值（软更新后的目标网络）
            with torch.no_grad():
                max_next_q = target_net(ns).max(1)[0]
                target = r + GAMMA * max_next_q * (1 - d)
            loss = nn.MSELoss()(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 软更新目标网络（每步都做）
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

        if done:
            break

    eps = max(EPS_END, eps * EPS_DECAY)
    episode_rewards.append(total_reward)

    if (episode+1) % 20 == 0:
        avg = np.mean(episode_rewards[-20:])
        print(f"Episode {episode+1}/{EPISODES}, Avg Reward (last 20): {avg:.1f}, eps: {eps:.3f}")

# 5.保存训练曲线
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN on CartPole')
plt.savefig('dqn_result.png', dpi=150)
print("训练曲线已保存为 dqn_result.png")