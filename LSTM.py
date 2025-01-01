import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# Set all seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define standard actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)

        self.critic_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        actor_features = self.actor_fc(state)
        mean = self.actor_mean(actor_features)
        log_std = self.actor_log_std(actor_features)
        std = torch.exp(log_std)

        value = self.critic_fc(state)
        return mean, std, value

# Initialize environment and parameters
env = gym.make('HalfCheetah-v5')
state_dim = env.observation_space.shape[0]  # Typically 17
action_dim = env.action_space.shape[0]      # Typically 6
hidden_dim = 128

# Initialize actor-critic network and optimizer
actor_critic = ActorCritic(state_dim, hidden_dim, action_dim)
optimizer = optim.Adam(actor_critic.parameters(), lr=1e-3)

# Training parameters
num_episodes = 100
gamma = 0.99  # Discount factor
max_steps_per_episode = 1000  # Limit the number of steps per episode

for episode in range(num_episodes):
    state = env.reset(seed=42)[0]
    state = torch.FloatTensor(state).unsqueeze(0)  # No LSTM, so no sequence dimension
    episode_reward = 0
    log_probs = []
    values = []
    rewards = []

    for _ in range(max_steps_per_episode):
        # Forward pass through the actor-critic network
        mean, std, value = actor_critic(state)
        normal_dist = torch.distributions.Normal(mean, std)
        raw_action = normal_dist.sample()
        action = torch.tanh(raw_action).squeeze(0).detach().numpy()  # Fixed action dimension issue

        # Interact with the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # Store log probability, value, and reward
        log_prob = normal_dist.log_prob(raw_action).sum(dim=-1)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = next_state
        episode_reward += reward

        if terminated or truncated:
            break

    # Compute discounted rewards-to-go
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + gamma * cumulative_reward
        returns.insert(0, cumulative_reward)
    returns = torch.FloatTensor(returns)

    # Compute advantages
    values = torch.cat(values)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    advantages = returns - values.detach()

    # Compute actor and critic losses
    actor_loss = -(torch.cat(log_probs) * advantages).sum()
    critic_loss = (returns - values).pow(2).sum()
    loss = actor_loss + critic_loss

    # Optimize the actor-critic network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close()
