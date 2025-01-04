import numpy as np
import random
from collections import deque, namedtuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

torch.autograd.set_detect_anomaly(True)

@dataclass
class Config:
    BUFFER_SIZE: int = 100000
    BATCH_SIZE: int = 64
    GAMMA: float = 0.99
    TAU: float = 1e-3
    LR_ACTOR: float = 1e-4
    LR_CRITIC: float = 1e-3
    UPDATE_EVERY: int = 1
    SEQ_LENGTHS: list = (8, 4, 2)
    HIDDEN_DIM: int = 256
    MAX_GRAD_NORM: float = 1.0
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED: int = 42

config = Config()

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", ["state","action","reward","next_state","done"])

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self, seq_length):
        if len(self.memory) < seq_length + 1:
            return None
        sequences = []
        for _ in range(self.batch_size):
            idx = random.randint(0, len(self.memory) - seq_length - 1)
            seq = [self.memory[i] for i in range(idx, idx + seq_length)]
            sequences.append(seq)
        states = torch.tensor(
            np.array([[exp.state for exp in seq] for seq in sequences]),
            dtype=torch.float32
        ).to(config.DEVICE)
        actions = torch.tensor(
            np.array([[exp.action for exp in seq] for seq in sequences]),
            dtype=torch.float32
        ).to(config.DEVICE)
        rewards = torch.tensor(
            np.array([[exp.reward for exp in seq] for seq in sequences]),
            dtype=torch.float32
        ).to(config.DEVICE)
        next_states = torch.tensor(
            np.array([[exp.next_state for exp in seq] for seq in sequences]),
            dtype=torch.float32
        ).to(config.DEVICE)
        dones = torch.tensor(
            np.array([[float(exp.done) for exp in seq] for seq in sequences]),
            dtype=torch.float32
        ).to(config.DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_limit):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, states, hidden=None):
        lstm_out, hidden = self.lstm(states, hidden)
        actions = torch.tanh(self.fc(lstm_out)) * self.action_limit
        return actions, hidden

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions, hidden=None):
        xu = torch.cat([states, actions], dim=2)
        lstm_out, hidden = self.lstm(xu, hidden)
        q_values = self.fc(lstm_out)
        return q_values, hidden

class OUNoise:
    def __init__(self, action_dim, seed=42, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class DRDPGAgent:
    def __init__(self, state_size, action_size, action_limit):
        self.state_size = state_size
        self.action_size = action_size
        self.action_limit = action_limit
        self.actor_local = Actor(state_size, action_size, config.HIDDEN_DIM, action_limit).to(config.DEVICE)
        self.actor_target = Actor(state_size, action_size, config.HIDDEN_DIM, action_limit).to(config.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)
        self.critic_local = Critic(state_size, action_size, config.HIDDEN_DIM).to(config.DEVICE)
        self.critic_target = Critic(state_size, action_size, config.HIDDEN_DIM).to(config.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        self.memory = ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)
        self.noise = OUNoise(action_size)
        self.t_step = 0

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, hidden_actor=None):
        self.actor_local.eval()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            action, hidden_actor = self.actor_local(state, hidden_actor)
        self.actor_local.train()
        action = action.squeeze(0).squeeze(0).cpu().numpy()
        noise = self.noise.sample()
        action = np.clip(action + noise, -self.action_limit, self.action_limit)
        return action, hidden_actor

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % config.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) >= config.BATCH_SIZE:
            self.learn()

    def learn(self):
        critic_loss_sum = 0
        actor_loss_sum = 0
        count = 0

        for seq_length in config.SEQ_LENGTHS:
            sample = self.memory.sample(seq_length)
            if sample is None:
                continue
            states, actions, rewards, next_states, dones = sample
            with torch.no_grad():
                actions_next, _ = self.actor_target(next_states)
                Q_targets_next, _ = self.critic_target(next_states, actions_next)
                Q_targets_next = Q_targets_next.squeeze(-1)
                Q_targets = rewards + config.GAMMA * Q_targets_next * (1 - dones)
            Q_expected, _ = self.critic_local(states, actions)
            Q_expected = Q_expected.squeeze(-1)
            critic_loss = nn.MSELoss()(Q_expected, Q_targets)
            actions_pred, _ = self.actor_local(states)
            Q_pred, _ = self.critic_local(states, actions_pred)
            actor_loss = -Q_pred.mean()
            critic_loss_sum += critic_loss
            actor_loss_sum += actor_loss
            count += 1

        if count > 0:
            critic_loss_avg = critic_loss_sum / count
            actor_loss_avg = actor_loss_sum / count
            self.critic_optimizer.zero_grad()
            critic_loss_avg.backward()
            nn.utils.clip_grad_norm_(self.critic_local.parameters(), config.MAX_GRAD_NORM)
            self.critic_optimizer.step()
            self.actor_optimizer.zero_grad()
            actor_loss_avg.backward()
            nn.utils.clip_grad_norm_(self.actor_local.parameters(), config.MAX_GRAD_NORM)
            self.actor_optimizer.step()
            self.soft_update(self.critic_target, self.critic_local, config.TAU)
            self.soft_update(self.actor_target, self.actor_local, config.TAU)

def train_agent(env_name, num_episodes, max_t=1000):
    env = gym.make(env_name)
    env.reset(seed=config.RANDOM_SEED)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_limit = env.action_space.high[0]
    agent = DRDPGAgent(state_size, action_size, action_limit)
    scores_deque = deque(maxlen=100)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        agent.noise.reset()
        hidden_actor = None
        total_reward = 0
        for t in range(max_t):
            action, hidden_actor = agent.select_action(state, hidden_actor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        scores_deque.append(total_reward)
        average_reward = np.mean(scores_deque)
        if episode % 10 == 0:
            print(f"Episode {episode}\tAverage Reward: {average_reward:.2f}")
        if average_reward >= 3000.0:
            print(f"Environment solved in {episode} episodes!")
            break
    env.close()

if __name__ == "__main__":
    train_agent("HalfCheetah-v5", num_episodes=1000)
