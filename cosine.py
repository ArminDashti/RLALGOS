import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque

# Set all seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Cosine function for alpha adjustment
def cosine_alpha(n, A=1, B=1, C=0, D=0, k_a_reduced=0.005, k_w_reduced=0.002):
    return abs(A * np.exp(-k_a_reduced * n) * np.cos(B * np.exp(-k_w_reduced * n) * n + C) + D)

# Define Gaussian Policy Network
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, num_actions)
        self.log_std = nn.Linear(hidden_size, num_actions)
        action_scale = (action_space.high - action_space.low) / 2.
        action_bias = (action_space.high + action_space.low) / 2.
        self.register_buffer("action_scale", torch.FloatTensor(action_scale))
        self.register_buffer("action_bias", torch.FloatTensor(action_bias))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Define Q Network
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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

# Soft Actor-Critic (SAC) Agent
class SACAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, buffer_size=1000000, batch_size=256):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_count = 0

        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.policy = GaussianPolicy(num_inputs, num_actions, action_space=env.action_space).to(self.device)
        self.q1 = QNetwork(num_inputs, num_actions).to(self.device)
        self.q2 = QNetwork(num_inputs, num_actions).to(self.device)
        self.q1_target = QNetwork(num_inputs, num_actions).to(self.device)
        self.q2_target = QNetwork(num_inputs, num_actions).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.policy.forward(state)
                action = torch.tanh(mean) * self.policy.action_scale + self.policy.action_bias
        else:
            action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = ((self.alpha * log_probs) - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            self.episode_count += 1
            state = self.env.reset(seed=42)[0]
            episode_reward = 0
            done = False

            # Update alpha dynamically
            self.alpha = cosine_alpha(self.episode_count)

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add((torch.FloatTensor(state).unsqueeze(0),
                                        torch.FloatTensor(action).unsqueeze(0),
                                        reward, torch.FloatTensor(next_state).unsqueeze(0), done))

                state = next_state
                episode_reward += reward

                self.update_parameters()

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Alpha = {self.alpha}")

# Main execution
if __name__ == "__main__":
    ENV_NAME = 'HalfCheetah-v5'
    NUM_EPISODES = 100

    env = gym.make(ENV_NAME)
    agent = SACAgent(env)
    agent.train(NUM_EPISODES)

    env.close()
