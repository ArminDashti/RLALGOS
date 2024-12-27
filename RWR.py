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
from gymnasium.wrappers import RecordVideo
from ila.datasets.farama_minari import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 256
DEFAULT_ENV_NAME = "D4RL/door/expert-v2"
SAVE_PATH_WINDOWS = "C:/users/armin/step_aware"
SAVE_PATH_UNIX = "/home/armin/step_aware"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-8  # For numerical stability

def set_seed(seed):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Actor(nn.Module):
    """Actor network with shared layers and action probability output."""
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        shared = self.shared_layers(state)
        action_logits = self.action_layer(shared)
        action_probs = self.softmax(action_logits)
        return action_probs

def evaluate_policy(actor, device, env, save_path, seeds=None):
    """Evaluate the current policy over multiple seeds and return average reward."""
    if seeds is None:
        seeds = [42]
    total_rewards = []
    # Wrap the environment with RecordVideo once
    env = RecordVideo(env, video_folder=save_path, episode_trigger=lambda episode_id: True)
    actor.eval()
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        total_reward, done = 0.0, False
        with torch.no_grad():
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action_probs = actor(state_tensor).cpu().numpy().flatten()
                action = np.argmax(action_probs)
                state, reward, terminations, truncations, _ = env.step(action)
                total_reward += reward
                done = terminations or truncations
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    logger.info(f"Average Reward over {len(seeds)} seeds: {avg_reward}")
    return avg_reward

def reward_weighted_regression(policy, optimizer, states, actions, rewards, temperature=1.0):
    """Perform Reward-Weighted Regression update on the policy."""
    rewards = rewards.squeeze()
    # Normalize rewards to prevent large exponents
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPSILON)
    weights = torch.exp(rewards / temperature)
    weights_sum = weights.sum()
    if weights_sum.item() == 0:
        logger.warning("Sum of weights is zero. Skipping update.")
        return
    weights = weights / weights_sum

    optimizer.zero_grad()
    action_probs = policy(states)  # Shape: [batch_size, action_dim]

    # Ensure actions are of shape [batch_size, 1] for gather
    actions = actions.view(-1, 1)
    # Add EPSILON to prevent log(0)
    action_log_probs = torch.log(action_probs.gather(1, actions) + EPSILON).squeeze()

    loss = -torch.sum(weights * action_log_probs)
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

def train_actor(minari, dataset, epochs, lr, batch_size, save_path, device, temperature=1.0):
    """Train the Actor network using Reward-Weighted Regression."""
    shapes = minari.dataset_shapes
    state_dim = shapes['states'][1]
    action_dim = shapes['actions'][1]

    actor = Actor(state_dim, action_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)

    # Convert one-hot actions to indices if necessary
    actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    if actions.dim() > 1 and actions.size(1) > 1:
        actions = actions.argmax(dim=1)
    else:
        actions = actions.squeeze()
    actions = actions.long()

    # Ensure 'future_returns' exist and are appropriate
    if 'future_returns' not in dataset:
        raise KeyError("The dataset does not contain 'future_returns'.")
    rewards = torch.tensor(dataset['future_returns'], dtype=torch.float32).squeeze()
    if rewards.dim() == 0:
        rewards = rewards.unsqueeze(0)

    states = torch.tensor(dataset['states'], dtype=torch.float32)
    ds = TensorDataset(states, actions, rewards)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    log_file = os.path.join(save_path, "actor_training_log.txt")
    with open(log_file, "a") as f:
        f.write(f"Starting actor training with Reward-Weighted Regression (RWR) at {datetime.now()}...\n")

    # Define multiple seeds for evaluation
    eval_seeds = [42, 100, 2021, 7, 999]

    for epoch in range(1, epochs + 1):
        actor.train()
        for batch in train_loader:
            states_batch, actions_batch, rewards_batch = [x.to(device) for x in batch]
            reward_weighted_regression(actor, actor_optimizer, states_batch, actions_batch, rewards_batch, temperature)

        # Initialize environment for evaluation
        env = minari.env()
        env = env.recover_environment(render_mode='rgb_array')
        avg_reward = evaluate_policy(actor, device, env, save_path, seeds=eval_seeds)
        env.close()

        # Log the results
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}/{epochs}, Average Reward: {avg_reward:.4f}\n")
        logger.info(f"Epoch {epoch}/{epochs} | Average Reward: {avg_reward:.4f}")

        # Optionally, save the model at checkpoints
        if epoch % 10 == 0 or epoch == epochs:
            model_save_path = os.path.join(save_path, f"actor_epoch_{epoch}.pth")
            torch.save(actor.state_dict(), model_save_path)
            logger.info(f"Saved actor model at epoch {epoch} to {model_save_path}")

    # Save the final model
    final_model_path = os.path.join(save_path, "actor_final.pth")
    torch.save(actor.state_dict(), final_model_path)
    logger.info(f"Training completed. Final actor model saved to {final_model_path}")

def get_save_path():
    """Determine the save path based on the operating system."""
    return SAVE_PATH_WINDOWS if os.name == "nt" else SAVE_PATH_UNIX

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Actor with Reward-Weighted Regression")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME, help="Environment name")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--save_path", type=str, default=get_save_path(), help="Path to save models and logs")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for RWR")
    return parser.parse_args()

def main():
    """Main function to execute training."""
    args = parse_arguments()
    set_seed(args.seed)

    # Ensure save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    minari = Dataset(args.env_name)
    minari.download_processed()
    data = minari.dict_data()

    if data is None:
        raise ValueError("The processed dataset is None. Ensure `download_processed()` is implemented correctly.")

    # Validate dataset keys
    required_keys = {'states', 'actions', 'future_returns'}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        raise KeyError(f"The dataset is missing required keys: {missing}")

    train_actor(
        minari=minari,
        dataset=data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path,
        device=DEVICE,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
