import collections
import random

import math
import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (
                               self.pos + 1) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def size(self):
        return len(self.buffer)

"""
SumTree 用于支持下述 Actor PER
"""
class SumTree(object):
    def __init__(self, max_size):
        self.levels = [np.zeros(1)]
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority and then search the tree for the corresponding index
    def sample(self, batch_size):
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        ind = np.zeros(batch_size, dtype=int)

        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]

            is_greater = np.greater(value, left_sum)

            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater

            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            value -= left_sum * is_greater

        return ind

    def set(self, ind, new_priority):
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind, new_priority):
        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set_v2(self, ind, new_priority, t):
        max_ind_value = ind[-1]

        if len(ind) % 2 == 0:
            loop_counter = len(self.levels[::-1])

            for i in range(loop_counter):
                if i == 0:
                    self.levels[::-1][i][:len(new_priority)] = new_priority

                    max_ind_value //= 2

                else:
                    check_cond_1 = max_ind_value + 1

                    if i == 1:
                        len_priorities = len(new_priority)
                    else:
                        len_priorities = len(self.levels[::-1][i - 1][0:dummy])

                    if math.ceil(len_priorities / 2) == check_cond_1:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1] = new_priority[0:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1] = self.levels[::-1][i - 1][0:dummy][0:len_priorities:2]
                    else:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1 - 1] = new_priority[0:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1 - 1] = self.levels[::-1][i - 1][0:dummy][0:len_priorities:2]

                    if math.floor(len_priorities / 2) == check_cond_1:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1] += new_priority[1:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1] += self.levels[::-1][i - 1][0:dummy][1:len_priorities:2]
                    else:
                        if i == 1:
                            self.levels[::-1][i][:check_cond_1 - 1] += new_priority[1:len_priorities:2]
                        else:
                            self.levels[::-1][i][:check_cond_1 - 1] += self.levels[::-1][i - 1][0:dummy][1:len_priorities:2]

                    dummy = len_priorities // 2

                    if dummy == 1 or dummy == 0:
                        dummy = 2

                    max_ind_value //= 2


class ActorPrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.device = device

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.critic_tree = SumTree(max_size)

        self.max_priority_critic = 1.0

        self.new_tree = SumTree(max_size)

        self.beta_critic = 0.4

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.critic_tree.set(self.ptr, self.max_priority_critic)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_uniform(self, batch_size):
        ind = np.random.randint(self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            None
        )

    def sample_critic(self, batch_size):
        ind = self.critic_tree.sample(batch_size)

        weights = self.critic_tree.levels[-1][ind] ** -self.beta_critic
        weights /= weights.max()

        self.beta_critic = min(self.beta_critic + 2e-7, 1)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            torch.FloatTensor(weights).to(self.device).reshape(-1, 1)
        )

    def sample_actor(self, batch_size, t):
        top_value = self.critic_tree.levels[0][0]

        reversed_priorities = top_value / (self.critic_tree.levels[-1][:self.ptr] + 1e-6)

        if self.ptr != 0:
            self.new_tree.batch_set_v2(np.arange(self.ptr), reversed_priorities, t)

        ind = self.new_tree.sample(batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            torch.FloatTensor(reversed_priorities[ind]).to(self.device).reshape(-1, 1)
        )

    def update_priority_critic(self, ind, priority):
        self.max_priority_critic = max(priority.max(), self.max_priority_critic)
        self.critic_tree.batch_set(ind, priority)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
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


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
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
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
