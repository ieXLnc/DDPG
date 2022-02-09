import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(14)
np.random.default_rng(14)


def weight_init__uniform_net(m):
    classname = m.__class__.__name__
    # for every linear layer in the model
    if classname.find('Linear') != -1:
        # get the number of inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


class Critic(nn.Module):
    def __init__(self, in_dims, fc1, fc2, out_dims):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(in_dims, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, out_dims)

        self.apply(weight_init__uniform_net)

    def forward(self, state, action):

        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x


class Actor(nn.Module):
    def __init__(self, in_dims, fc1, fc2, out_dims):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(in_dims, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, out_dims)

        self.apply(weight_init__uniform_net)

    def forward(self, state):

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))

        return x
