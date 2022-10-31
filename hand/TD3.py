import argparse
import copy
import os
import shutil
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import str2bool, Reward_adapter, evaluate_policy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # 没有先提取特征
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_Agent(object):
    def __init__(
            self,
            env_with_dw,
            state_dim,
            action_dim,
            max_action,
            gamma=0.99,
            net_width=128,
            a_lr=1e-4,
            c_lr=1e-4,
            batch_size=256,
            policy_delay_freq=1
    ):

        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(state_dim, action_dim, net_width).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_dw = env_with_dw
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005
        self.batch_size = batch_size
        self.delay_counter = -1
        self.delay_freq = policy_delay_freq

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        self.delay_counter += 1
        with torch.no_grad():
            s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_a = (
                    self.actor_target(s_prime) + noise  # Noisy on target action
            ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
        target_Q = torch.min(target_Q1, target_Q2)

        '''Avoid impacts caused by reaching max episode steps'''
        if self.env_with_dw:
            target_Q = r + (1 - dw_mask) * self.gamma * target_Q  # dw: die or win
        else:
            target_Q = r + self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        if self.delay_counter == self.delay_freq:
            # Update Actor
            a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = -1

    def save(self, EnvName, episode):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, episode))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, episode))

    def load(self, EnvName, episode):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, episode)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, episode)))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, reward, next_state, dead):
        # 每次只放入一个时刻的数据
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size  # 存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device)
        )


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='PV0, Lch_Cv2, Humanv2, HCv2, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=1e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e4, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

parser.add_argument('--policy_delay_freq', type=int, default=1, help='Delay frequency of Policy Updating')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--exp_noise', type=float, default=0.15, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
print(opt)


def main():
    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'BipedalWalker-v3',
               'BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv2', 'HCv2', 'BWv3', 'BWHv3']  # Brief Environment Name.
    env_with_dw = [False, True, True, False, True, True]  # dw:die and win
    EnvIdex = opt.EnvIdex
    env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # remark: action space【-max,max】
    expl_noise = opt.exp_noise
    max_e_steps = env._max_episode_steps
    print('Env:', EnvName[EnvIdex], '  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action,
          '  min_a:', env.action_space.low[0], '  max_e_steps:', max_e_steps)

    update_after = 2 * max_e_steps  # update actor and critic after update_after steps
    start_steps = 10 * max_e_steps  # start using actor to iterate after start_steps steps

    # Random seed config:
    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "env_with_dw": env_with_dw[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": opt.gamma,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "policy_delay_freq": opt.policy_delay_freq
    }
    if not os.path.exists('model'): os.mkdir('model')
    model = TD3_Agent(**kwargs)
    if opt.Loadmodel: model.load(BrifEnvName[EnvIdex], opt.ModelIdex)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    if opt.render:
        score = evaluate_policy(env, model, opt.render, turns=10)
        print('EnvName:', BrifEnvName[EnvIdex], 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, done, steps, ep_r = env.reset(), False, 0, 0

            '''Interact & trian'''
            while not done:
                steps += 1  # steps in one episode

                if total_steps < start_steps:
                    a = env.action_space.sample()
                else:
                    a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                         ).clip(-max_action, max_action)  # explore: deterministic actions + noise
                s_prime, r, done, info = env.step(a)
                r = Reward_adapter(r, EnvIdex)

                '''Avoid impacts caused by reaching max episode steps'''
                if (done and steps != max_e_steps):
                    dw = True  # dw: dead and win
                else:
                    dw = False

                replay_buffer.add(s, a, r, s_prime, dw)
                s = s_prime
                ep_r += r

                '''train if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= update_after and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(replay_buffer)

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    expl_noise *= opt.noise_decay
                    score = evaluate_policy(eval_env, model, False)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('expl_noise', expl_noise, global_step=total_steps)
                    print('EnvName:', BrifEnvName[EnvIdex], 'steps: {}k'.format(int(total_steps / 1000)), 'score:',
                          score)
                total_steps += 1

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    model.save(BrifEnvName[EnvIdex], total_steps)
        env.close()


main()