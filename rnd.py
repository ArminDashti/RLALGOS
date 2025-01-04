import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal
from collections import deque
import random

lambda_gae = 0.95

learning_rate = 3e-4
num_episodes = 500
gamma = 0.99
ensemble_size = 2
min_noise = 0.001
max_noise = 0.005
high_uncertainty_threshold = 1.0
entropy_coeff = 1.0
batch_size = 64
buffer_capacity = 10000

env_name = "HalfCheetah-v5"
env = gym.make(env_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std_clamped = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std_clamped)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_nets = [PolicyNetwork(state_dim, action_dim).to(device) for _ in range(ensemble_size)]
value_net = ValueNetwork(state_dim).to(device)

policy_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in policy_nets]
value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

buffer = ReplayBuffer(buffer_capacity)

def compute_uncertainty(state):
    with torch.no_grad():
        means = [net(state)[0] for net in policy_nets]
        variance = torch.var(torch.stack(means), dim=0)
        uncertainty = variance.mean(dim=1, keepdim=True)
    return uncertainty

def compute_gae(rewards, values, next_values, dones):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lambda_gae * gae * (1 - dones[i])
        advantages.insert(0, gae)
    return advantages

def train_actor_critic():
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards = []
        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            uncertainty = compute_uncertainty(state_tensor.unsqueeze(0))
            entropy_scale = torch.clamp(uncertainty, min=min_noise, max=max_noise)
            entropy_scale = torch.where(uncertainty > high_uncertainty_threshold,
                                        torch.tensor(1.0).to(device),
                                        entropy_scale)
            
            means = [net(state_tensor)[0] for net in policy_nets]
            stds = [net(state_tensor)[1] for net in policy_nets]
            actions = [Normal(mean, std).sample() for mean, std in zip(means, stds)]
            mean_action = torch.stack(means).mean(dim=0)
            action = torch.tanh(mean_action + torch.randn_like(mean_action) * entropy_scale)
            action_np = action.detach().cpu().numpy()
            
            value = value_net(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.push((
                state_tensor.detach().cpu(),
                action.detach().cpu(),
                reward,
                [Normal(mean, std).log_prob(a).sum(dim=-1).detach().cpu() for mean, std, a in zip(means, stds, actions)],
                value.detach().cpu(),
                done
            ))
            state = next_state
            episode_rewards.append(reward)

        if len(buffer) >= batch_size:
            for _ in range(len(buffer) // batch_size):
                transitions = buffer.sample(batch_size)
                states, actions, rewards, log_probs_list, values, dones = zip(*transitions)
                states = torch.stack(states).to(device)
                actions = torch.stack(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)
                values = torch.stack(values).squeeze().to(device)

                with torch.no_grad():
                    next_values = value_net(states)
                    next_values = torch.cat([next_values[1:], torch.zeros(1, 1).to(device)], dim=0).squeeze()

                advantages = compute_gae(rewards.detach().cpu().numpy(), values.detach().cpu().numpy(),
                                         next_values.detach().cpu().numpy(), dones.detach().cpu().numpy())
                advantages = torch.FloatTensor(advantages).to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                value_predictions = value_net(states).squeeze()
                returns = advantages + values
                value_loss = nn.MSELoss()(value_predictions, returns)

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                policy_losses = []
                entropy_terms = []
                for i, net in enumerate(policy_nets):
                    mean, std = net(states)
                    dist = Normal(mean, std)
                    log_probs = dist.log_prob(actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    policy_loss = -(log_probs * advantages).mean()
                    entropy_loss = -entropy_coeff * (entropy * entropy_scale.squeeze()).mean()
                    total_loss = policy_loss + entropy_loss
                    policy_optimizers[i].zero_grad()
                    total_loss.backward()
                    policy_optimizers[i].step()

        print(f"Episode {episode + 1}, Reward: {sum(episode_rewards):.2f}")

if __name__ == "__main__":
    train_actor_critic()
