import datetime
import os
import random
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

import rl_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device,env, num_episodes, replay_buffer, minimal_size, batch_size):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络

        self.actor2 = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device) # 策略网络二号
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
            device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
            device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.env = env
        self.num_episodes = num_episodes
        self.replay_buffer = replay_buffer
        self.minimal_size = minimal_size
        self.batch_size =batch_size
        self.orflag = 0

    def take_action(self, state ,flag = True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0] if flag else self.actor2(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones,flag = True):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states) if flag else self.actor2(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update_actor1(self,states):
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return entropy

    def update_actor2(self, states):
        new_actions, log_prob = self.actor2(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor2_optimizer.zero_grad()
        actor_loss.backward()
        self.actor2_optimizer.step()
        return entropy

    def update(self, transition_dict,flag = True):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        if flag:
            self.orflag += 1
            entropy = self.update_actor1(states)
            if self.orflag %3 ==0:
                self.update_actor2(states)
        else:
            entropy = self.update_actor2(states)

        self.orflag = 0
        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


    def train(self):
        return_list = []
        flag = True
        i_d = 0
        for i in range(10):
            with tqdm(total=int(self.num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes / 10)):
                    i_d +=1
                    episode_return = 0
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = self.take_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if self.replay_buffer.size() > self.minimal_size:
                            '''
                            是否可以适当处理这个重放的次数，将其作为参数。根据回报值，加一个网络监督这个参数。
                            '''
                            for k in range(3):
                                b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(batch_size)
                                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                                   'dones': b_d}
                                if (i_d + 1) % 100 == 0:
                                    flag = False
                                    print("----actor2 learn......")
                                    self.update(transition_dict,flag)
                                else:
                                    self.update(transition_dict,flag)


                    return_list.append(episode_return)

                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))





parser = argparse.ArgumentParser()
# environment
parser.add_argument('--env_name', default='Pendulum-v1')
parser.add_argument('--algorithm', default='SACRED')
parser.add_argument('--pre_transform_image_size', default=100, type=int)

parser.add_argument('--image_size', default=84, type=int)
parser.add_argument('--action_repeat', default=1, type=int)
parser.add_argument('--frame_stack', default=3, type=int)

parser.add_argument("actor_lr",default = 3e-4)
parser.add_argument("critic_lr",default = 3e-3)
parser.add_argument("alpha_lr",default = 3e-4)
parser.add_argument("num_episodes",default = 500)
parser.add_argument("hidden_dim",default = 128)
parser.add_argument("gamma",default = 0.99)
parser.add_argument("tau",default = 0.005)
parser.add_argument("buffer_size",default = 100000)
parser.add_argument("minimal_size",default = 1000)
parser.add_argument("batch_size",default = 64)

args = parser.parse_args()


env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值

#  尝试绘制阴影曲线
for i in range(5):
    seed = i
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    algorithm = args.algorithm
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device, env, num_episodes, replay_buffer, minimal_size, batch_size)

    return_list = agent.train()

    fileName = "../my_result/{}_{}_{}.npy".format(algorithm, args.env_name,
                                                  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    np.save(fileName, return_list)







