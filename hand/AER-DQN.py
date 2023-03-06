# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from collections import namedtuple
# from itertools import count
# import gym
#
#
# # 定义经验缓冲区中存储的经验
# Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
#
# # 定义DQN网络
# class DQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# # 定义注意力网络
# class Attention(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Attention, self).__init__()
#         self.fc1 = nn.Linear(state_size + action_size, 64)
#         self.fc2 = nn.Linear(64, 1)
#
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # 定义智能体
# class Agent():
#     def __init__(self, state_size, action_size, gamma=0.99, batch_size=32, lr=1e-3):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.lr = lr
#
#         # 初始化DQN网络和注意力网络
#         self.dqn_net = DQN(state_size, action_size)
#         self.attention_net = Attention(state_size, action_size)
#
#         # 定义经验缓冲区
#         self.memory = []
#
#         # 定义优化器
#         self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.lr)
#
#     # 选择行动
#     def select_action(self, state):
#         s = [[i] for i in range(self.action_size)]
#         with torch.no_grad():
#             q_values = self.dqn_net(torch.FloatTensor(state)).detach().numpy()
#             attention_scores = self.attention_net(torch.FloatTensor(state),torch.FloatTensor(s)).squeeze().detach().numpy()
#             print(attention_scores)
#             q_values = q_values + attention_scores
#             action = q_values.argmax()
#         return action
#
#     # 存储经验
#     def store_experience(self, experience):
#         self.memory.append(experience)
#
#     # 训练
#     def train(self):
#         if len(self.memory) < self.batch_size:
#             return
#
#         # 从经验缓冲区中随机选择一批经验
#         experiences = random.sample(self.memory, self.batch_size)
#
#         # 分离state、action、next_state和reward
#         states, actions, next_states, rewards = zip(*experiences)
#
#         # 将它们转换为PyTorch张量
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)
#         next_states = torch.FloatTensor(next_states)
#         rewards = torch.FloatTensor(rewards)
#
#         # 计算每个经验的TD误差
#         q_values = self.dqn_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         next_q_values = self.dqn_net(next_states).max(1)[0].detach()
#         td_errors = rewards + self.gamma * next_q_values - q_values
#
#         # 计算每个经验的注意力分数
#         attention_scores = self.attention_net(states, actions.unsqueeze(1)).squeeze(1)
#
#         # 计算每个经验的权重
#         weights = attention_scores * td_errors
#
#         # 使用加权的经验来更新DQN网络
#         loss = (weights ** 2).mean()
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # 清空经验缓冲区
#         self.memory = []
#
# # 创建一个环境
# env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# # 创建一个智能体
# agent = Agent(state_size=state_dim, action_size=action_dim)
#
# # 训练
# for i_episode in range(100):
#     state = env.reset()
#     for t in count():
#         # 选择行动
#         action = agent.select_action(state)
#
#         # 执行行动
#         next_state, reward, done, _ = env.step(action)
#
#         print(reward)
#
#         # 存储经验
#         agent.store_experience(Experience(state, action, next_state, reward))
#
#         # 更新状态
#         state = next_state
#
#         # 训练
#         agent.train()
#
#         if done:
#             break


import collections
import datetime
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

import rl_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 定义注意力网络
class AttentionNet(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(AttentionNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc1_2 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc1_2(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)

        self.attention_net = AttentionNet(state_dim,action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    """选择动作还要更新"""
    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float).to(self.device)
                shape = torch.FloatTensor([[i] for i in range(self.action_dim)]).to(self.device)
                # todo
                attention_scores = self.attention_net(state,shape).to(self.device)
                action = (self.q_net(state)+attention_scores).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标

        # 注意力
        att_score = self.attention_net(states,actions).squeeze(1)

        weight = att_score * q_targets
        dqn_loss = torch.mean(weight **2)  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


algorithm = "DQN-AER"
lr = 2e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.95
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")
print(device)

env_name = 'CartPole-v1'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    for k in range(5):
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}

                        """拿相同的数据去更新，或者从一个池子中，反复采样
                        """
                        agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


dir = "./result"
if not os.path.exists(dir):
    os.makedirs(dir)
fileName = "{}/{}_{}_{}.npy".format(dir,algorithm, env_name,
                                                    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
print(os.getcwd())
np.save(fileName, return_list)

episodes_list = list(range(len(return_list)))


mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('{} moving_average on {}'.format(algorithm,env_name))
plt.show()
