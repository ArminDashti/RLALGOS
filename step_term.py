#%%
import numpy as np
from ila.datasets.farama_minari import Dataset, add_nexts
from ila.utils.farama_gymnasium import evaluate_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader

dataset_id = "D4RL/door/human-v2"
minari = Dataset(dataset_id)
dataset = minari.to_dict()

# Initialize episode_done array with zeros
dataset['episode_done'] = np.zeros_like(dataset['states'])

# Extract states and dones from the dataset
states = dataset['states']
dones = dataset['dones']

# Find indices where episodes are done
done_indices = np.where(dones)[0]

# Map states to the nearest done state for each step
for i in range(len(states)):
    closest_done_idx = done_indices[done_indices >= i].min() if np.any(done_indices >= i) else done_indices[-1]
    dataset['episode_done'][i] = states[closest_done_idx]

dataset = add_nexts(dataset, ['states'])
#%%
print(dataset.keys())
# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, step=None):
        x = self.hidden(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state,):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + state_dim + state_dim + action_dim, 256)  # Includes both done states
        self.fc1 = nn.Linear(state_dim + action_dim, 256)  # Includes both done states
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Define custom dataset for Minari data
class MinariDataset(TorchDataset):
    def __init__(self, dataset):
        self.states = torch.tensor(dataset['states'], dtype=torch.float32)
        self.actions = torch.tensor(dataset['actions'], dtype=torch.float32)
        self.rewards = torch.tensor(dataset['rewards'], dtype=torch.float32)
        self.terminations = torch.tensor(dataset['terminations'], dtype=torch.float32)
        self.truncations = torch.tensor(dataset['truncations'], dtype=torch.float32)
        self.next_states = torch.tensor(dataset['next_states'], dtype=torch.float32)
        self.steps = torch.tensor(dataset['steps'], dtype=torch.float32)
        self.past_returns = torch.tensor(dataset['past_returns'], dtype=torch.float32)
        self.future_returns = torch.tensor(dataset['future_returns'], dtype=torch.float32)
        self.dones = torch.tensor(dataset['dones'], dtype=torch.float32)
        self.episode_done = torch.tensor(dataset['episode_done'], dtype=torch.float32)

    def __len__(self):
        return len(self.states)-1

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'terminations': self.terminations[idx],
            'truncations': self.truncations[idx],
            'next_states': self.next_states[idx],
            'steps': self.steps[idx],
            'past_returns': self.past_returns[idx],
            'future_returns': self.future_returns[idx],
            'dones': self.dones[idx],
            'episode_done': self.episode_done[idx]
        }

# Initialize dataset and dataloader
minari_dataset = MinariDataset(dataset)
dataloader = DataLoader(minari_dataset, batch_size=64, shuffle=True)

# Initialize actor and critic networks
actor = Actor(state_dim=39, action_dim=28)
critic = Critic(state_dim=39, action_dim=28)
actor.train()
critic.train()

# Define optimizers
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=0.0001)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.0001)

# Define discount factor
gamma = 0.99

# Define a target done state
target_done_state_base = torch.tensor([-0.11317386, -0.74615619,  0.45638067, -0.03016726, -0.20718082, -0.01188001,
                                        0.68812667,  0.03324709,  0.01746637,  0.13716846,  0.203915,    0.23010166,
                                        0.14622286, -0.06587624,  0.02318344,  0.364236,    0.36975146,  0.10940316,
                                       -0.24152849,  0.10579144,  0.28664274,  0.21281208, -0.05450402,  0.88553315,
                                        0.16213126, -0.21015972, -0.49618742,  1.79818367,  1.02190141,  0.33215968,
                                       -0.28247214,  0.21499075,  0.32090315, -0.25597729,  0.2101142,   0.01125653,
                                       -0.02649485,  0.00487655,  1.], dtype=torch.float32)

# Training loop
for epoch in range(200):
    total_actor_loss = 0
    total_critic_loss = 0
    for batch in dataloader:
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards'] / 21
        next_states = batch['next_states']
        dones = batch['dones']
        done_states = batch['episode_done']
        steps = batch['steps'] / 300

        # Expand target_done_state to match batch size
        target_done_state = target_done_state_base.expand(states.size(0), -1)

        with torch.no_grad():
            sampled_actions, log_probs = actor.sample_action(next_states)
            
            target_values = (rewards + gamma * critic(next_states, sampled_actions) * (1-dones))

        predicted_values = critic(states, actions)
        critic_loss = F.mse_loss(predicted_values, target_values)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()
        total_critic_loss += critic_loss.item()

        sampled_actions, _ = actor.sample_action(states)
        actor_loss = -critic(states, sampled_actions).mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()
        total_actor_loss += actor_loss.item()
    
    print('total_actor_loss', total_actor_loss)
    # print('total_critic_loss', total_critic_loss)
    minari_instance = minari.minari_instance
    env = minari.env()
    evaluate_policy(env, actor, 'c:/users/armin/step_aware', target_done_state_base)

# %%
