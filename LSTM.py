import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import os
from gymnasium.wrappers import RecordVideo

# Set all seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define LSTM-based policy network
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]  # Output from the last LSTM cell
        mean = self.mean_layer(lstm_out)
        log_std = self.log_std_layer(lstm_out)
        std = torch.exp(log_std)
        return mean, std, hidden

# Define independent action prediction network
class ActionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        actions = self.output_layer(x)
        return actions

# Evaluate and save video
def evaluate(policy=None, env_name='HalfCheetah-v4', 
             render_mode='rgb_array', 
             seeds=[42], max_step=200, 
             name_prefix='half_cheetah_run', 
             save_dir=r'C:\Users\Armin\step_aware'):
    os.makedirs(save_dir, exist_ok=True)

    def only_first_episode_trigger(episode_id):
        return episode_id == 0

    env = gym.make(env_name, render_mode=render_mode)
    env = RecordVideo(env, video_folder=save_dir, name_prefix=name_prefix, episode_trigger=only_first_episode_trigger)

    total_rewards = []

    for seed in seeds:
        observation, info = env.reset()
        episode_reward = 0
        hidden = (torch.zeros(1, 1, policy.lstm.hidden_size), torch.zeros(1, 1, policy.lstm.hidden_size))

        for _ in range(max_step):
            if policy:
                policy.eval()
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
                action_mean, action_std, hidden = policy(obs_tensor, hidden)
                action = torch.tanh(action_mean).detach().numpy().squeeze()
                print(action.mean())
            else:
                action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Reward for seed {seed}: {episode_reward}")

    env.close()

    average_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    print(f"Average reward: {average_reward}")

# Initialize environment and parameters
env = gym.make('HalfCheetah-v4')
state_dim = env.observation_space.shape[0]  # Typically 17
action_dim = env.action_space.shape[0]      # Typically 6
hidden_dim = 128

# Initialize policy network, action predictor, and optimizer
policy = LSTMPolicy(state_dim, hidden_dim, action_dim)
action_predictor = ActionPredictor(state_dim, action_dim)
optimizer_policy = optim.Adam(policy.parameters(), lr=1e-3)
optimizer_predictor = optim.Adam(action_predictor.parameters(), lr=1e-3)

# Training parameters
num_episodes = 200
gamma = 0.99  # Discount factor
max_steps_per_episode = 1000  # Limit the number of steps per episode
total_reward_avg = 0
for episode in range(num_episodes):
    state = env.reset(seed=42)[0]
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
    hidden = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
    episode_reward = 0
    log_probs = []
    rewards = []
    predictor_loss = 0

    for _ in range(max_steps_per_episode):
        # Forward pass through the policy network
        mean, std, hidden = policy(state, hidden)
        normal_dist = torch.distributions.Normal(mean, std)
        raw_action = normal_dist.sample()
        action = torch.tanh(raw_action).squeeze(0)  # Ensures action is within [-1, 1]

        # Predict action using the independent network
        predicted_action = action_predictor(state.squeeze(0).squeeze(0))

        # Penalize the loss for the difference between LSTM policy actions and predicted actions
        predictor_loss += nn.MSELoss()(predicted_action, action.detach())

        # Interact with the environment
        next_state, reward, done, _, _ = env.step(action.detach().numpy())
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

        # Store log probability and reward
        log_prob = normal_dist.log_prob(raw_action).sum(dim=-1)
        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state
        episode_reward += reward

        if done:
            break

    # Compute discounted rewards
    discounted_rewards = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    discounted_rewards = torch.FloatTensor(discounted_rewards)

    # Normalize rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    # Compute policy loss
    policy_loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()

    # Backpropagation
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    optimizer_predictor.zero_grad()
    predictor_loss.backward()
    optimizer_predictor.step()

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Predictor Loss = {predictor_loss.item()}")
    total_reward_avg += episode_reward

print(total_reward_avg)

env.close()
