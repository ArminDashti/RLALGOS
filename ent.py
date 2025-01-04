import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(reward).unsqueeze(1),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done).unsqueeze(1))
    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(1, keepdim=True)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.replay_buffer = ReplayBuffer(1_000_000)
        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().numpy()[0]
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state
        action = action
        reward = reward
        next_state = next_state
        done = done
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target = reward + (1 - done) * self.gamma * target_q
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        action_new, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, action_new)
        q2_new = self.critic2(state, action_new)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - min_q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        entropy_coef_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            uncertainty = (current_q1 - current_q2).abs().mean()
            if uncertainty < 0.1:
                self.log_alpha.fill_(-100)  # effectively minimizing entropy
            else:
                self.log_alpha.fill_(torch.log(torch.tensor(0.2)))
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action)

    episodes = 1000
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            episode_reward += reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
