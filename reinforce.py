import numpy as np

import torch.nn as nn
import torch
from torchsummary import summary

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.shared_network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(86*86, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_network = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )


        self.log_std_network = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )


    def forward(self, state):
        shared = self.shared_network(state)

        mu = self.mu_network(shared)
        log_std = torch.log(self.log_std_network(shared) + 1)

        return mu, log_std


class REINFORCE:
    def __init__(self, action_dim):
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []

        self.policy_network = PolicyNetwork(action_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        state = np.moveaxis(state, -1, 0)
        state = torch.tensor(np.array(state), dtype=torch.float32)

        mu, log_std = self.policy_network(state)

        print(mu, log_std)

        distribution = torch.distributions.Normal(mu + self.eps, log_std.exp() + self.eps)
        action = distribution.sample()
        prob = distribution.log_prob(action)

        self.probs.append(prob)

        return action
    
    def update(self):
        self.optimizer.zero_grad()

        G = 0
        for i in reversed(range(len(self.rewards))):
            G = self.gamma * G + self.rewards[i]

        loss = 0
        for i in range(len(self.probs)):
            loss += -self.probs[i] * G

        loss = loss.mean()

        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []