import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

wine = load_wine()
X, y = wine.data, wine.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

input_dim = X.shape[1]
output_dim = len(wine.target_names)
model = SimpleNN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()

class LearningRateEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, model, criterion, train_loader, device='cpu'):
        super(LearningRateEnv, self).__init__()
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.num_gradients = sum(p.numel() for p in self.model.parameters())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.num_gradients + 1,),
                                            dtype=np.float32)
        self.epoch = 0
        self.max_epochs = 20
        self.train_loader_iterator = iter(self.train_loader)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.model.train()
        self.total_loss = 0.0
        self.num_batches = 0
        self.train_loader_iterator = iter(self.train_loader)
        self.epoch += 1
        self.current_batch = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        lr_scalar = action[0]
        try:
            X_batch, y_batch = next(self.train_loader_iterator)
        except StopIteration:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = -self.total_loss / self.num_batches if self.num_batches > 0 else 0.0
            done = True
            info = {}
            return observation, reward, done, info
        self.model.zero_grad()
        outputs = self.model(X_batch)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
            else:
                gradients.append(torch.zeros_like(param).view(-1))
        gradients = torch.cat(gradients).detach().cpu().numpy()
        observation = np.concatenate([gradients, np.array([loss.item()])]).astype(np.float32)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= lr_scalar * param.grad
        reward = -loss.item()
        self.total_loss += loss.item()
        self.num_batches += 1
        done = False
        info = {}
        return observation, reward, done, info

env = LearningRateEnv(model, criterion, train_loader)

class LearningRateActor(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[64, 64]):
        super(LearningRateActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Sigmoid()
        )
    def forward(self, state, *args, **kwargs):
        return self.net(state)

class LearningRateCritic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[64, 64]):
        super(LearningRateCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
    def forward(self, state, *args, **kwargs):
        return self.net(state)

state_dim = env.observation_space.shape[0]
hidden_sizes = [128, 128]
actor = LearningRateActor(state_dim, hidden_sizes).to('cpu')
critic = LearningRateCritic(state_dim, hidden_sizes).to('cpu')
optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)

def compute_gae(rewards, dones, values, next_value, gamma=0.99, tau=0.95):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * (1 - dones[step]) * next_value - values[step]
        gae = delta + gamma * tau * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def ppo_update(actor, critic, optimizer, states, actions, log_probs_old, returns, advantages, clip_epsilon=0.2, epochs=10, batch_size=64):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).unsqueeze(1)
    returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
    advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            s, a, old_logp, R, A = batch
            mean = actor(s)
            std = torch.ones_like(mean) * 1e-2
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(a)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * A
            actor_loss = -torch.min(surr1, surr2).mean()
            value = critic(s)
            critic_loss = nn.MSELoss()(value, R)
            loss = actor_loss + 0.5 * critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_learning_rate_controller():
    max_epochs = 20
    max_steps = len(train_loader)
    for epoch in range(max_epochs):
        state, _ = env.reset()
        done = False
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean = actor(state_tensor)
            std = torch.ones_like(mean) * 1e-2
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = critic(state_tensor)
            action_np = action.detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action_np)
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            state = next_state
        next_value = 0 if done else critic(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).item()
        advantages, returns = compute_gae(rewards, dones, values, next_value)
        ppo_update(actor, critic, optimizer, states, actions, log_probs, returns, advantages)
        avg_reward = np.mean(rewards)
        print(f"Epoch {epoch+1}/{max_epochs}, Average Reward: {avg_reward:.4f}")

train_learning_rate_controller()

def apply_learned_lr():
    observation, _ = env.reset()
    episode_reward = 0.0
    optimizer_model = optim.SGD(model.parameters(), lr=1.0)
    model.train()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer_model.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
            else:
                gradients.append(torch.zeros_like(param).view(-1))
        gradients = torch.cat(gradients).detach().cpu().numpy()
        current_loss = loss.item()
        state = np.concatenate([gradients, np.array([current_loss])]).astype(np.float32)
        action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()[0][0]
        lr_scalar = action
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data *= lr_scalar
        optimizer_model.step()
        episode_reward += -current_loss
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {current_loss:.4f}, LR Scalar: {lr_scalar:.4f}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

apply_learned_lr()
