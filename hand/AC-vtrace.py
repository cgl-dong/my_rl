import os

import matplotlib.pyplot as plt
import numpy as np

import rl_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    '''通过td-error改进critic,critic的评价会逐渐精准'''

    def update(self, transition_dict, rho_clip=1.0, c_clip=1.0):
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

        # 时序差分目标 ，使用状态代替动作
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 时序差分误差,误差越小越好 critic可以说是监督者，监督这个动作
        td_delta = td_target - self.critic(states)

        log_probs = torch.log(self.actor(states).gather(1, actions))

        rho = torch.exp(log_probs - td_delta.detach())
        rho = torch.clamp(rho, max=rho_clip)
        c = torch.clamp(rho, max=c_clip)
        v_trace = c * td_target

        # Compute policy loss
        policy_loss = -torch.mean(rho * log_probs * v_trace)

        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        """熵 鼓励探索正则项"""
        # entropy_bonus = -torch.mean(torch.distributions.Categorical(F.softmax(log_probs)).entropy())

        # Compute total loss
        # total_loss = policy_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
        total_loss = policy_loss + 0.5 * critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


from tqdm import tqdm


def v_trace_train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


"""
参数
"""
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)
algorithm = "AC"

env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

return_list = v_trace_train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('AC on {}'.format(env_name))
plt.show()
