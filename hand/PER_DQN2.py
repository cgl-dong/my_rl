import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from common.buffer import PrioritizedBuffer

import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import rl_utils


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class PERAgent:

    def __init__(self, env, use_conv=False, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = PrioritizedBuffer(buffer_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_conv:
            self.model = ConvDQN(self.env.observation_space.shape[0], env.action_space.n).to(self.device)
        else:
            self.model = Qnet(self.env.observation_space.shape[0], env.action_space.n).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.0):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if (np.random.rand() > eps):
            return self.env.action_space.sample()

        return action

    def _sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def _compute_TDerror(self, batch_size):
        transitions, idxs, IS_weights = self._sample(batch_size)
        states, actions, rewards, next_states, dones = transitions

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        IS_weights = torch.FloatTensor(IS_weights).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        td_errors = torch.pow(curr_Q - expected_Q, 2) * IS_weights

        return td_errors, idxs

    def update(self, batch_size):
        td_errors, idxs = self._compute_TDerror(batch_size)

        # update model
        td_errors_mean = td_errors.mean()
        self.optimizer.zero_grad()
        td_errors_mean.backward()
        self.optimizer.step()

        # update priorities
        for idx, td_error in zip(idxs, td_errors.cpu().detach().numpy()):
            self.replay_buffer.update_priority(idx, td_error)


env_id = "CartPole-v1"
num_episodes = 500
batch_size = 64

env = gym.make(env_id)
agent = PERAgent(env, use_conv=False)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PerDQN on {}'.format(env_id))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PerDQN moving_average on {}'.format(env_id))
plt.show()