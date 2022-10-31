import numpy as np
import torch
import torch.nn.functional as F


# QNet
class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNet, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        r = F.relu(self.l1(x))
        return self.l2(r)


# DQN
class DQN:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, device):

        self.action_dim = action_dim
        self.q_net = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.op = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    # 选取动作，epsilon贪婪策略
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
