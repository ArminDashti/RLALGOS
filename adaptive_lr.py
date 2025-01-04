import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 10
TRANSFORM_RESIZE = (16, 16)
EMA_ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.2
MIN_LR = 1e-5
MAX_LR = 1e-2

transform = transforms.Compose([
    transforms.Resize(TRANSFORM_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

criterion = nn.CrossEntropyLoss()

# Define initial learning rates for each layer
lr_conv1 = INITIAL_LEARNING_RATE
lr_conv2 = INITIAL_LEARNING_RATE * 0.1
lr_fc1 = INITIAL_LEARNING_RATE * 0.01
lr_fc2 = INITIAL_LEARNING_RATE * 0.001

# Create optimizer with separate parameter groups
optimizer = optim.SGD([
    {'params': net.conv1.parameters(), 'lr': lr_conv1},
    {'params': net.conv2.parameters(), 'lr': lr_conv2},
    {'params': net.fc1.parameters(), 'lr': lr_fc1},
    {'params': net.fc2.parameters(), 'lr': lr_fc2}
], momentum=MOMENTUM)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

def compute_ema(values, alpha=EMA_ALPHA):
    ema_values = []
    ema = 0
    for i, value in enumerate(values):
        ema = alpha * value + (1 - alpha) * ema if i > 0 else value
        ema_values.append(ema)
    return ema_values

def construct_state(current_step, total_steps, current_loss, validation_loss, learning_rates, 
                   params_norms, grad_norms, train_accuracy, val_accuracy, ema_loss, ema_val_loss):
    # Normalize learning rates
    normalized_lrs = [lr / MAX_LR for lr in learning_rates]
    state = [
        current_step / total_steps,
        current_loss,
        validation_loss,
        *normalized_lrs,
        train_accuracy,
        val_accuracy,
        ema_loss,
        ema_val_loss
    ]
    state.extend(params_norms)
    state.extend(grad_norms)
    return torch.tensor(state, dtype=torch.float32).to(device)

num_layers = 4
state_dim = 19  # Updated from 16 to 19 to match the state size
action_dim = num_layers

actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim).to(device)

optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

def ppo_train(actor, critic, optimizer_actor, optimizer_critic, states, actions, rewards, dones, old_log_probs, gamma=GAMMA, epsilon=EPSILON):
    returns = []
    discounted_sum = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_sum = 0
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    old_log_probs = torch.stack(old_log_probs).to(device)

    values = critic(states).squeeze()
    advantages = returns - values.detach()

    mean, std = actor(states)
    dist = torch.distributions.Normal(mean, std)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    ratios = torch.exp(log_probs - old_log_probs.detach())

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    critic_loss = F.mse_loss(values, returns)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

def validate_model(net, dataloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    ema_loss = 0
    ema_val_loss = 0
    total_steps = EPOCHS * len(trainloader)

    layers = [net.conv1, net.conv2, net.fc1, net.fc2]

    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    dones_buffer = []
    old_log_probs_buffer = []

    print("Starting Training with PPO-based Learning Rate Adjustment...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            current_step = epoch * len(trainloader) + i
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            grad_norms = []
            for layer in layers:
                if layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.data.norm(2).item()
                else:
                    grad_norm = 0.0
                grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

            optimizer.step()

            params_norms = []
            for layer in layers:
                param_norm = layer.weight.data.norm(2).item()
                params_norms.append(param_norm)

            ema_loss = EMA_ALPHA * loss.item() + (1 - EMA_ALPHA) * ema_loss
            validation_loss = ema_loss
            train_accuracy = 0.0  # Update with actual training accuracy if available
            val_accuracy = 0.0    # Update with actual validation accuracy if available
            ema_val_loss = EMA_ALPHA * validation_loss + (1 - EMA_ALPHA) * ema_val_loss

            state = construct_state(
                current_step=current_step,
                total_steps=total_steps,
                current_loss=loss.item(),
                validation_loss=validation_loss,
                learning_rates=[pg['lr'] for pg in optimizer.param_groups],
                params_norms=params_norms,
                grad_norms=grad_norms,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                ema_loss=ema_loss,
                ema_val_loss=ema_val_loss
            )

            mean, std = actor(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            new_lrs = torch.clamp(action, MIN_LR, MAX_LR).cpu().numpy()
            for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
                param_group['lr'] = new_lr

            reward = -validation_loss
            done = i == len(trainloader) - 1

            states_buffer.append(state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            dones_buffer.append(done)
            old_log_probs_buffer.append(log_prob.detach())

            running_loss += loss.item()
            if i % 100 == 99:
                lr_info = ", ".join([f"{pg['lr']:.6f}" for pg in optimizer.param_groups])
                print(f"Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100:.3f}, LRs {lr_info}")
                running_loss = 0.0

        ppo_train(actor, critic, optimizer_actor, optimizer_critic, states_buffer, actions_buffer, rewards_buffer, dones_buffer, old_log_probs_buffer)

        states_buffer.clear()
        actions_buffer.clear()
        rewards_buffer.clear()
        dones_buffer.clear()
        old_log_probs_buffer.clear()

        val_accuracy = validate_model(net, testloader, device)
        print(f"Epoch {epoch + 1}: Validation Accuracy: {val_accuracy:.2f}%")

    print("Finished Training with PPO-based Learning Rate Adjustment")

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    test_accuracy = validate_model(net, testloader, device)
    print(f'Final Test Accuracy: {test_accuracy:.2f}%')

if __name__ == "__main__":
    main()
