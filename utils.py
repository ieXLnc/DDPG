import numpy as np
from collections import deque
import random
import pickle
import torch
import gym

np.random.default_rng(14)
random.seed(14)
torch.manual_seed(14)


class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.max_size)

    def memorize(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))

    def get_sample(self):
        random_sample = random.sample(self.buffer, self.batch_size)

        # states, actions, rewards, dones, next_states = map(np.stack, zip(*random_sample))
        states = np.zeros((self.batch_size, 3), dtype=np.float)
        actions = np.zeros((self.batch_size, 1), dtype=np.float)
        rewards = np.zeros((self.batch_size, 1), dtype=np.float)
        dones = np.zeros((self.batch_size, 1), dtype=np.float)
        next_states = np.zeros((self.batch_size, 3), dtype=np.float)
        for (state, action, reward, done, next_state), (t) in zip(random_sample, range(self.batch_size)):
            states[t] = state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            next_states[t] = next_state

        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), \
              torch.FloatTensor(dones), torch.FloatTensor(next_states)

    def __len__(self):
        return len(self.buffer)

    def save_memory(self, name):
        with open('./Models/' + 'memory_' + name, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Ornstein-Ulhenbeck Process
# Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
