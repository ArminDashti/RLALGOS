import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim, gymnasium as gym
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from ila.datasets.farama_minari import Dataset
from ila.utils.farama_gymnasium import evaluate_policy

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 256
DEFAULT_ENV_NAME = "D4RL/door/human-v2"
SAVE_PATH_WINDOWS = "C:/users/armin/step_aware"
SAVE_PATH_UNIX = "/home/armin/step_aware"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-8
TEMPERATURE = 1.0

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RWRPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_steps=None, embed_dim=None, hidden_sizes=[128, 128]):
        super(RWRPolicy, self).__init__()
        self.use_embedding = max_steps is not None and embed_dim is not None
        input_dim = state_dim + embed_dim if self.use_embedding else state_dim
        # input_dim = 39
        if self.use_embedding:
            self.embedding = nn.Embedding(max_steps + 1, embed_dim)
        layers = []
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU()])
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state, step=None):
        embed = self.embedding(step)
        state = torch.cat([state, embed], dim=1)

        x = self.hidden(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample_action(self, state, step=None):
        mean, std = self.forward(state, step)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

def reward_weighted_regression(policy, optimizer, states, actions, rewards, temperature=TEMPERATURE, steps=None):
    rewards = (rewards.squeeze() - rewards.mean()) / (rewards.std() + EPSILON)
    weights = torch.exp(rewards / temperature)
    weights_sum = weights.sum()
    weights /= weights_sum
    optimizer.zero_grad()
    means, stds = policy(states, steps)
    dist = torch.distributions.Normal(means, stds)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    loss = -torch.sum(weights * log_probs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

def train_actor(minari, dataset, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE, save_path=None, device=DEVICE, temperature=TEMPERATURE, max_steps=None, embed_dim=None):
    state_dim, action_dim = minari.shapes['states'][1], minari.shapes['actions'][1]
    actor = RWRPolicy(state_dim, action_dim, max_steps=max_steps, embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    rewards = torch.tensor(dataset['future_returns'], dtype=torch.float32).unsqueeze(1)
    states = torch.tensor(dataset['states'], dtype=torch.float32)
    steps = torch.tensor(dataset['steps'], dtype=torch.long)
    ds = TensorDataset(states, actions, rewards, steps) if steps is not None else TensorDataset(states, actions, rewards)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    save_path = save_path or (SAVE_PATH_WINDOWS if os.name == "nt" else SAVE_PATH_UNIX)
    os.makedirs(save_path, exist_ok=True)
    print(f"Starting actor training with Reward-Weighted Regression (RWR) at {datetime.now()}...")

    for epoch in range(1, epochs + 1):
        actor.train()
        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            reward_weighted_regression(actor, optimizer, batch[0], batch[1], batch[2], temperature, batch[3])
        env = minari.minari_instance
        env = env.recover_environment(render_mode='rgb_array')
        avg_reward = evaluate_policy(actor, device, env, save_path)
        print(f"Epoch {epoch}/{epochs} | Average Reward: {avg_reward:.4f}")

    final_model_path = os.path.join(save_path, "actor_final.pth")
    torch.save(actor.state_dict(), final_model_path)
    print(f"Training completed. Final actor model saved to {final_model_path}")

def main():
    set_seed(DEFAULT_SEED)
    minari = Dataset(DEFAULT_ENV_NAME)
    data = minari.to_dict()
    if data is None:
        raise ValueError("The processed dataset is None. Ensure `download_processed()` is implemented correctly.")
    required_keys = {'states', 'actions', 'future_returns'}
    missing = required_keys - data.keys()
    if missing:
        raise KeyError(f"The dataset is missing required keys: {missing}")
    max_steps = data['steps'].max() if 'steps' in data else None
    train_actor(minari, data, max_steps=max_steps, embed_dim=16 if max_steps else None)

if __name__ == "__main__":
    main()
