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

# set_seed(42)

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.action_head(x))

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.temp_buffer = deque()
        self.best_episode_reward = float('-inf')

    def add(self, transition, episode_reward, episode):
        if episode > 10 and episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward
            self.buffer.extend(self.temp_buffer)
            self.temp_buffer.clear()
            self.buffer.append(transition)
        elif episode > 10:
            self.temp_buffer.append(transition)
        else:
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

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, env, gamma=0.99, lr=3e-4, buffer_size=1000000, batch_size=256):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.actor = Actor(num_inputs, num_actions).to(self.device)
        self.critic = Critic(num_inputs).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update Critic
        with torch.no_grad():
            target_value = rewards + self.gamma * (1 - dones) * self.critic(next_states)
        value = self.critic(states)
        critic_loss = F.mse_loss(value, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        policy_loss = -self.critic(states).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add((torch.FloatTensor(state).unsqueeze(0),
                                        torch.FloatTensor(action).unsqueeze(0),
                                        reward, torch.FloatTensor(next_state).unsqueeze(0), done),
                                       episode_reward, episode)

                state = next_state
                episode_reward += reward

                self.update_parameters()

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Main execution
if __name__ == "__main__":
    ENV_NAME = 'HalfCheetah-v5'
    NUM_EPISODES = 100

    env = gym.make(ENV_NAME)
    agent = ActorCriticAgent(env)
    agent.train(NUM_EPISODES)

    env.close()
