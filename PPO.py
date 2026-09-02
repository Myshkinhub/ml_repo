import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# 1.超参数
ENV_NAME = 'CartPole-v1'
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
LR = 3e-4
ENTROPY_COEF = 0.01
STEPS_PER_UPDATE = 1024   # 每次收集的步数
BATCH_SIZE = 64
PPO_EPOCHS = 10
MAX_EPISODES = 2000       # 最大训练批次

env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# 2.Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x).squeeze(1)

policy = ActorCritic()
optimizer = optim.Adam(policy.parameters(), lr=LR)

# 3.经验收集
def collect_rollout(env, policy, steps):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    state, _ = env.reset()
    state = state.astype(np.float32)
    for _ in range(steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = policy(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_state = next_state.astype(np.float32) if not done else None
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        values.append(value.item())
        state = next_state if not done else env.reset()[0].astype(np.float32)
    return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32),
            np.array(log_probs, dtype=np.float32), np.array(values, dtype=np.float32))

# 4.GAE 优势计算
def compute_gae(rewards, dones, values, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0.0 if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns

# 5.训练主循环
episode_rewards = []          # 存储每个实际回合的总奖励
batch_episode_avg = []        # 用于记录每批次的平均回合奖励（仅用于绘图）
total_steps = 0

for update in range(MAX_EPISODES):
    # 收集经验
    states, actions, rewards, dones, old_log_probs, values = collect_rollout(env, policy, STEPS_PER_UPDATE)
    total_steps += STEPS_PER_UPDATE

    # 从收集的奖励中提取每个回合的累计奖励
    ep_rew = 0.0
    for r, d in zip(rewards, dones):
        ep_rew += r
        if d:
            episode_rewards.append(ep_rew)
            ep_rew = 0.0

    # 计算优势并标准化
    advantages, returns = compute_gae(rewards, dones, values, GAMMA, GAE_LAMBDA)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO 更新
    indices = np.arange(len(states))
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, len(states), BATCH_SIZE):
            idx = indices[start:start+BATCH_SIZE]
            s = torch.FloatTensor(states[idx])
            a = torch.LongTensor(actions[idx])
            old_lp = torch.FloatTensor(old_log_probs[idx])
            adv = torch.FloatTensor(advantages[idx])
            ret = torch.FloatTensor(returns[idx])

            logits, vals = policy(s)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(a)
            ratio = torch.exp(new_lp - old_lp)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-CLIP_RANGE, 1+CLIP_RANGE) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(vals, ret)
            entropy_loss = -ENTROPY_COEF * dist.entropy().mean()
            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    # 记录本批次平均回合奖励（用于绘图）
    if episode_rewards:
        batch_episode_avg.append(np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards))

    # 打印进度
    if (update+1) % 10 == 0:
        avg_recent = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Update {update+1}, Total Steps: {total_steps}, Avg Reward (last 10 episodes): {avg_recent:.1f}")

    # 早停条件：最近10个回合平均奖励 ≥ 190
    if len(episode_rewards) >= 10 and np.mean(episode_rewards[-10:]) >= 190:
        print(f"✅ Solved at update {update+1}!")
        break

# 6.测试
test_env = gym.make(ENV_NAME, render_mode=None)   # 如需可视化，改为 'human'
state, _ = test_env.reset()
state = state.astype(np.float32)
test_reward = 0
for _ in range(500):
    with torch.no_grad():
        logits, _ = policy(torch.FloatTensor(state).unsqueeze(0))
        action = logits.argmax().item()
    next_state, r, terminated, truncated, _ = test_env.step(action)
    test_reward += r
    if terminated or truncated:
        state, _ = test_env.reset()
        state = state.astype(np.float32)
    else:
        state = next_state.astype(np.float32)
test_env.close()
print(f"🎯 Test Reward (single episode): {test_reward}")

# 7.绘图保存
plt.figure()
plt.plot(batch_episode_avg)
plt.xlabel('Update Batch (each = 1024 steps)')
plt.ylabel('Average Episode Reward (last 10)')
plt.title('PPO on CartPole')
plt.savefig('ppo.png')
print("训练曲线已保存为 ppo_corrected.png")
