import collections
import argparse

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
                 epsilon, target_update, device,buffer_size = 10000):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.replay_buffer = rl_utils.PrioritizedReplay(buffer_size)


    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action


    def train(self,env, num_episodes, minimal_size, batch_size):
        return_list = []
        i = 0
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    i+=1
                    episode_return = 0
                    state = env.reset()
                    done = False
                    while not done:
                        action = self.take_action(state)
                        next_state, reward, done, _ = env.step(action)
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if self.replay_buffer.size() > minimal_size:
                            states, actions, rewards, next_states, dones, idx, weights = self.replay_buffer.sample(
                                batch_size)
                            transition_dict = {'states': states, 'actions': actions, 'next_states': next_states,
                                               'rewards': rewards, 'dones': dones, "idx": idx, "weights": weights}
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
            save_dict = {"q_net":self.q_net.state_dict(),"i":i}
            torch.save(save_dict,"./result/DQN.pth")
        return return_list

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
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Twin Delayed Deep Deterministic Policy Gradient')

    parser.add_argument("--policy", default="DQN-with-PER", help='Algorithm (default: LA3P_TD3)')
    parser.add_argument("--env", default="CartPole-v1", help='OpenAI Gym environment name')
    parser.add_argument("--num_episodes", default=1000,type=int, help='num_episodes')
    parser.add_argument("--seed", default=0, type=int,
                        help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')


    args = parser.parse_args()

    lr = 2e-3
    num_episodes = args.num_episodes
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

    env_name = args.env
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = agent.train(env, num_episodes, minimal_size, batch_size)

    algorithm = "DQN-with-PER"

    # dir = "./result"
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # fileName = "{}/{}_{}_{}.npy".format(dir, algorithm, env_name,
    #                                     datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # np.save(fileName, return_list)

    episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('{} on {}'.format(algorithm,env_name))
    # plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('{} moving_average on {}'.format(algorithm, env_name))
    plt.show()
