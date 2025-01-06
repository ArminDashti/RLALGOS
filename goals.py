import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def sample_action(mean, std):
    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()
    action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-7)
    return action, log_prob.sum(dim=-1, keepdim=True)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        action, log_prob = sample_action(mean, std)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        s, a, r, s2, d = zip(*samples)
        return (torch.cat(s),
                torch.cat(a),
                torch.tensor(r, dtype=torch.float32),
                torch.cat(s2),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

def update_target_network(source_net, target_net, tau=0.005):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def update_actor_critic(batch_size, actor, critic, target_critic, actor_opt, critic_opt, memory):
    if len(memory) < batch_size:
        return None, None
    s, a, r, s2, d = memory.sample(batch_size)
    with torch.no_grad():
        next_vals = target_critic(s2).squeeze()
        targets = r + (1 - d) * 0.99 * next_vals
    vals = critic(s).squeeze()
    critic_loss = nn.MSELoss()(vals, targets)
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.5)
    critic_opt.step()
    vals_no_grad = vals.detach()
    adv = targets - vals_no_grad
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    actions, log_probs = actor.sample_action(s)
    actor_loss = -(log_probs.squeeze() * adv).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.5)
    actor_opt.step()
    update_target_network(critic, target_critic)
    return actor_loss.item(), critic_loss.item()

def train(env, actor, critic, target_critic, actor_opt, critic_opt, memory, max_episodes, batch_size):
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob = actor.sample_action(state_tensor)
            action_np = action.detach().cpu().numpy().squeeze()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            total_reward += reward
            memory.add(state_tensor, action.detach(), reward,
                       torch.tensor(next_state, dtype=torch.float32).unsqueeze(0), done)
            state = next_state
        actor_loss, critic_loss = update_actor_critic(batch_size, actor, critic,
                                                      target_critic, actor_opt,
                                                      critic_opt, memory)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, "
              f"Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    env = gym.make("HalfCheetah-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    target_critic = Critic(state_dim)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    actor_opt = optim.Adam(actor.parameters(), lr=1e-5)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)
    memory = ReplayBuffer(50000)
    train(env, actor, critic, target_critic, actor_opt, critic_opt, memory,
          max_episodes=500, batch_size=6)
