import numpy as np
from collections import deque
import random
import pickle
import torch  # Torch version :1.9.0+cpu
import gym
import matplotlib.pyplot as plt

np.random.default_rng(14)
random.seed(14)
torch.manual_seed(14)

cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")


class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.max_size)

    def memorize(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))

    def get_sample(self):
        random_sample = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, dones, next_states = map(np.stack, zip(*random_sample))

        return torch.FloatTensor(states).to(device), torch.FloatTensor(actions).to(device),\
               torch.FloatTensor(rewards).unsqueeze(1).to(device), torch.FloatTensor(dones).unsqueeze(1).to(device),\
               torch.FloatTensor(next_states).to(device)

    def len_(self):
        return len(self.buffer)

    def save_memory(self, name):
        with open('./Models/' + 'memory_' + name, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_memory(self, name):
        with open('./Models/' + 'memory_' + name, 'rb') as handle:
            self.buffer = pickle.load(handle)


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
        return action + ou_state


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


def plot(rewards_list, avg_reward_list, test_rewards, actor_loss_list, critic_loss_list, name_task):

    # Create two plots: one for the loss value, one for the accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 10))

    # Plot accuracy values
    ax1.plot(rewards_list, label='Rewards', color='red', alpha=0.3)
    ax1.plot(avg_reward_list, label='Average rewards', color='red')
    ax1.plot(test_rewards, label='Test rewards', color='green')
    ax1.set_title('Rewards for the {} task'.format(name_task))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Rewards')
    ax1.legend()

    # Plot accuracy values
    ax2.plot(actor_loss_list, label='Actor Losses', color='blue')
    ax2.set_title('Actor Network Losses for the {} task'.format(name_task))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Losses')
    ax2.legend()

    ax3.plot(critic_loss_list, label='Critic Losses', color='black')
    ax3.set_title('Critic Network Losses for the {} task'.format(name_task))
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Losses')
    ax3.legend()

    number = random.randint(1, 100000)
    plt.savefig("./Plots/" + name_task + '_' + str(number) + '_' + '_plot.png')
    plt.show()


def gif(images, name, address="./Recordings/"):
    images[0].save(address + name, save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)
