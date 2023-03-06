import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state, buffer):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        weights = nn.functional.softmax(x, dim=0)
        buffer = torch.FloatTensor(buffer)
        attended_buffer = torch.sum(weights * buffer, dim=0)
        return attended_buffer

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer, batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class AER:
    def __init__(self, state_dim, action_dim, hidden_size, lr, gamma, tau, buffer_capacity, batch_size):
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.actor_target = Actor(state_dim, action_dim, hidden_size)
        self.critic = Critic(state_dim, hidden_size)
        self.critic_target = Critic(state_dim, hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.attention = Attention(state_dim, hidden_size)
        self.attention_optimizer = optim.Adam(self.attention.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(torch.cat((next_states, next_actions), dim=1))
        Q_target = rewards + self.gamma * (1 - dones) * next_Q.detach()
        Q = self.critic(torch.cat((states, actions), dim=1))
        critic_loss = nn.MSELoss()(Q, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



        """
            def update(self, batch_size):
                # 从经验缓冲区中选择一批经验
                experiences = self.memory.sample(batch_size)
                
                # 计算每个经验的注意力分数
                attention_scores = self.attention_net(experiences.state, experiences.action)
                
                # 计算每个经验的TD误差
                td_errors = self.compute_td_errors(experiences)
                
                # 计算每个经验的权重
                weights = attention_scores * td_errors
                
                # 使用加权的经验来更新智能体的参数
                self.update_params(experiences, weights)
        """