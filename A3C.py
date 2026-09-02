import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import time

# 1.超参数
ENV_NAME = 'CartPole-v1'
N_WORKERS = 4                # 并行 Worker 数量
N_STEPS = 5                  # 每个 Worker 收集 n-step 经验
GAMMA = 0.99
LR = 1e-3
ENTROPY_COEF = 0.01          # 熵正则系数
VALUE_LOSS_COEF = 0.5        # 价值损失系数
MAX_GLOBAL_STEPS = 50000     # 全局总步数
LOG_INTERVAL = 1000          # 打印日志的步数间隔

# 2.全局变量（用于统计）
global_step = 0
episode_rewards = []        # 存储每个回合的总奖励
episode_rewards_lock = threading.Lock()
print_lock = threading.Lock()

# 3.神经网络（Actor-Critic 共享）
class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# 4.全局网络与优化器
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

global_net = Net(obs_dim, act_dim)
global_net.share_memory()   # 允许多线程共享参数
optimizer = optim.Adam(global_net.parameters(), lr=LR)

# 5.打印函数（线程安全）
def safe_print(msg):
    with print_lock:
        print(msg)

# 6.Worker 线程
class Worker(threading.Thread):
    def __init__(self, worker_id, global_net, optimizer):
        super().__init__()
        self.worker_id = worker_id
        self.global_net = global_net
        self.optimizer = optimizer
        self.local_net = Net(obs_dim, act_dim)
        self.env = gym.make(ENV_NAME)

    def run(self):
        global global_step, episode_rewards
        state, _ = self.env.reset()
        state = state.astype(np.float32)
        done = False
        episode_reward = 0.0
        episode_len = 0

        while global_step < MAX_GLOBAL_STEPS:
            # 同步本地网络参数
            self.local_net.load_state_dict(self.global_net.state_dict())

            # 收集 n-step 经验
            states, actions, rewards, dones, values = [], [], [], [], []
            for _ in range(N_STEPS):
                state_t = torch.FloatTensor(state).unsqueeze(0)
                logits, value = self.local_net(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                next_state = next_state.astype(np.float32) if not done else None

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                episode_reward += reward
                episode_len += 1

                state = next_state if not done else self.env.reset()[0].astype(np.float32)
                if done:
                    # 记录该回合的奖励
                    with episode_rewards_lock:
                        episode_rewards.append(episode_reward)
                    # 打印该回合信息
                    current_step = global_step + len(states)  # 近似当前步数
                    safe_print(f"[Worker {self.worker_id}] Episode finished: Reward={episode_reward:.1f}, Length={episode_len}, Global Steps≈{current_step}")
                    episode_reward = 0.0
                    episode_len = 0
                    break

            # 计算 n-step 回报和优势
            R = 0.0
            if not done:
                with torch.no_grad():
                    _, last_value = self.global_net(torch.FloatTensor(state).unsqueeze(0))
                    R = last_value.item()

            returns = []
            advantages = []
            for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
                R = r + GAMMA * R * (1 - d)
                adv = R - v
                returns.append(R)
                advantages.append(adv)
            returns.reverse()
            advantages.reverse()

            # 转换为 Tensor
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)

            # 计算损失
            logits, values_pred = self.local_net(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)

            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = VALUE_LOSS_COEF * F.mse_loss(values_pred.squeeze(1), returns)
            entropy_loss = -ENTROPY_COEF * dist.entropy().mean()
            loss = policy_loss + value_loss + entropy_loss

            # 计算梯度并异步更新全局网络
            self.optimizer.zero_grad()
            loss.backward()
            for local_param, global_param in zip(self.local_net.parameters(), self.global_net.parameters()):
                if global_param.grad is not None:
                    global_param.grad.data.add_(local_param.grad.data)
            self.optimizer.step()

            # 更新全局步数
            global_step += len(states)

            # 每 LOG_INTERVAL 步打印一次平均奖励
            if global_step % LOG_INTERVAL < len(states):  # 避免重复打印
                with episode_rewards_lock:
                    if len(episode_rewards) >= 10:
                        avg_reward = np.mean(episode_rewards[-10:])
                    else:
                        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                safe_print(f"[Log] Global Steps: {global_step}, Avg Reward (last 10 eps): {avg_reward:.1f}")

# 7.启动所有 Worker
workers = [Worker(i, global_net, optimizer) for i in range(N_WORKERS)]
for w in workers:
    w.start()
for w in workers:
    w.join()

# 8.测试
safe_print("\n=== Training Finished ===")
safe_print("Running test...")
test_env = gym.make(ENV_NAME, render_mode=None)
state, _ = test_env.reset()
state = state.astype(np.float32)
total_reward = 0
for _ in range(500):
    with torch.no_grad():
        logits, _ = global_net(torch.FloatTensor(state).unsqueeze(0))
        action = logits.argmax().item()
    next_state, r, terminated, truncated, _ = test_env.step(action)
    total_reward += r
    if terminated or truncated:
        state, _ = test_env.reset()
        state = state.astype(np.float32)
    else:
        state = next_state.astype(np.float32)
test_env.close()
safe_print(f"🎯 Test Reward: {total_reward}")

# 9.保存奖励曲线
if episode_rewards:
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A3C on CartPole (with logging)')
    plt.savefig('a3c_result.png')
    safe_print("Training curve saved as a3c_result.png")