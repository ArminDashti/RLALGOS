import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo
from ila.datasets.farama_minari import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 256
DEFAULT_ENV_NAME = "D4RL/door/expert-v2"
SAVE_PATH_WINDOWS = "C:/users/armin/step_aware"
SAVE_PATH_UNIX = "/home/armin/step_aware"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.action_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        shared = self.shared_layers(state)
        action = torch.tanh(self.action_layer(shared))
        return action

class RewardPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RewardPredictor, self).__init__()
        self.state_action_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.past_return_model = nn.Linear(hidden_dim, 1)
        self.future_return_model = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        features = self.state_action_model(state_action)
        past_return = self.past_return_model(features)
        future_return = self.future_return_model(features)
        return past_return, future_return

def evaluate_policy(actor, device, env, save_path):
    seeds = [42]
    total_rewards = []
    for seed in seeds:
        env = RecordVideo(env, video_folder=save_path, episode_trigger=lambda _: True)
        actor.eval()
        state, _ = env.reset(seed=seed)
        total_reward, done = 0.0, False
        with torch.no_grad():
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = actor(state_tensor).cpu().numpy().flatten()
                state, reward, terminations, truncations, _ = env.step(action)
                total_reward += reward
                done = terminations or truncations
        total_rewards.append(total_reward)
        env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    logger.info(f"Average Reward: {avg_reward}")
    return avg_reward


def train_reward_predictor(minari, dataset, epochs, lr, batch_size, save_path, device):
    shapes = minari.dataset_shapes
    state_dim = shapes['states'][1]
    action_dim = shapes['actions'][1]

    reward_predictor = RewardPredictor(state_dim, action_dim).to(device)
    reward_optimizer = optim.Adam(reward_predictor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(dataset['states'], dtype=torch.float32),
        torch.tensor(dataset['actions'], dtype=torch.float32),
        torch.tensor(dataset['past_returns'], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(dataset['future_returns'], dtype=torch.float32).unsqueeze(-1)
    )
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    log_file = os.path.join(save_path, "reward_predictor_training_log.txt")
    with open(log_file, "a") as f:
        f.write("Starting reward predictor training...\n")

    for epoch in range(1, epochs + 1):
        reward_predictor.train()
        epoch_reward_loss = 0.0

        for batch in train_loader:
            states, actions, past_returns_batch, future_returns_batch = [x.to(device) for x in batch]

            predicted_past, predicted_future = reward_predictor(states, actions)

            reward_loss = loss_fn(predicted_future, future_returns_batch)

            reward_optimizer.zero_grad()
            reward_loss.backward()
            reward_optimizer.step()

            epoch_reward_loss += reward_loss.item()

        avg_reward_loss = epoch_reward_loss / len(train_loader)

        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        epoch_save_folder = os.path.join(save_path, f"reward_predictor_{timestamp}_epoch_{epoch}")
        os.makedirs(epoch_save_folder, exist_ok=True)

        torch.save(reward_predictor.state_dict(), os.path.join(epoch_save_folder, "reward_predictor.pth"))

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}/{epochs}, Reward Loss: {avg_reward_loss:.4f}\n")

        logger.info(f"Epoch {epoch}/{epochs} | Reward Loss: {avg_reward_loss:.4f}")

    return reward_predictor

def train_actor(minari, dataset, epochs, lr, gamma, batch_size, save_path, device, reward_predictor):
    shapes = minari.dataset_shapes
    state_dim = shapes['states'][1]
    action_dim = shapes['actions'][1]

    actor = Actor(state_dim, action_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)

    ds = TensorDataset(
        torch.tensor(dataset['states'], dtype=torch.float32),
        torch.tensor(dataset['actions'], dtype=torch.float32)
    )
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    log_file = os.path.join(save_path, "actor_training_log.txt")
    with open(log_file, "a") as f:
        f.write("Starting actor training...\n")

    for epoch in range(1, epochs + 1):
        actor.train()
        epoch_actor_loss = 0.0

        for batch in train_loader:
            states, _ = [x.to(device) for x in batch]

            new_actions = actor(states)
            predicted_past_actions, predicted_future_actions = reward_predictor(states, new_actions)

            actor_loss = -(predicted_future_actions.mean() - predicted_past_actions.mean())

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            epoch_actor_loss += actor_loss.item()

        avg_actor_loss = epoch_actor_loss / len(train_loader)

        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        epoch_save_folder = os.path.join(save_path, f"actor_{timestamp}_epoch_{epoch}")
        os.makedirs(epoch_save_folder, exist_ok=True)

        torch.save(actor.state_dict(), os.path.join(epoch_save_folder, "actor.pth"))

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}/{epochs}, Actor Loss: {avg_actor_loss:.4f}\n")

        logger.info(f"Epoch {epoch}/{epochs} | Actor Loss: {avg_actor_loss:.4f}")

def get_save_path():
    return SAVE_PATH_WINDOWS if os.name == "nt" else SAVE_PATH_UNIX

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Reward Predictor and Actor")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--save_path", type=str, default=get_save_path())
    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)

    minari = Dataset(DEFAULT_ENV_NAME)
    minari.download_processed()
    data = minari.dict_data()

    if data is None:
        raise ValueError("The processed dataset is None. Ensure `download_processed()` is implemented correctly.")

    reward_predictor = train_reward_predictor(minari, data, args.epochs, args.lr, args.batch_size, args.save_path, DEVICE)
    train_actor(minari, data, args.epochs, args.lr, args.gamma, args.batch_size, args.save_path, DEVICE, reward_predictor)

if __name__ == "__main__":
    main()
