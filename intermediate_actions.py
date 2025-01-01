import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque, defaultdict

# Set all seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define actor network
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.fc(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)  # Clamp for numerical stability
        std = torch.exp(log_std)
        return mean, std

    def select_action(self, state):
        mean, std = self.forward(state)
        normal_dist = torch.distributions.Normal(mean, std)
        raw_action = normal_dist.sample()
        action = torch.tanh(raw_action).squeeze(0).detach().numpy()
        log_prob = normal_dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        return action, log_prob

# Define critic network
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.fc(state)
        return value

# Define exploration policy network
class ExplorationPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExplorationPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action)
        return action, log_prob

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states),
            torch.cat(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.cat(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# Initialize environment and parameters
def initialize_env_and_models(env_name, hidden_dim, lr):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, hidden_dim, action_dim)
    critic = Critic(state_dim, hidden_dim)
    exploration_policy = ExplorationPolicy(state_dim, hidden_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    exploration_optimizer = optim.Adam(exploration_policy.parameters(), lr=lr)
    return env, actor, critic, exploration_policy, actor_optimizer, critic_optimizer, exploration_optimizer

# Training loop
def train_actor_critic(env, actor, critic, exploration_policy, actor_optimizer, critic_optimizer, exploration_optimizer, buffer, num_episodes, max_steps, batch_size, gamma):
    for episode in range(num_episodes):
        state = env.reset(seed=42)[0]
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0

        for _ in range(max_steps):
            with torch.no_grad():
                action, log_prob = exploration_policy.sample_action(state)
                action = torch.tanh(action).squeeze(0).detach().cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            buffer.add((state, torch.FloatTensor(action).unsqueeze(0), reward, next_state, terminated or truncated))

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            with torch.no_grad():
                next_values = critic(next_states)
                targets = rewards + gamma * next_values * (1 - dones)

            values = critic(states)
            critic_loss = (targets - values).pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            advantages = (targets - values).detach()
            exploration_loss = -(advantages * log_prob).mean()
            exploration_optimizer.zero_grad()
            exploration_loss.backward()
            exploration_optimizer.step()

# Main execution
if __name__ == "__main__":
    ENV_NAME = 'HalfCheetah-v5'
    HIDDEN_DIM = 128
    LR = 1e-3
    NUM_EPISODES = 100
    MAX_STEPS = 1000
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    GAMMA = 0.99

    env, actor, critic, exploration_policy, actor_optimizer, critic_optimizer, exploration_optimizer = initialize_env_and_models(ENV_NAME, HIDDEN_DIM, LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    train_actor_critic(env, actor, critic, exploration_policy, actor_optimizer, critic_optimizer, exploration_optimizer, replay_buffer, NUM_EPISODES, MAX_STEPS, BATCH_SIZE, GAMMA)

    env.close()
