import numpy as np
import torch # Torch version :1.9.0+cpu
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
        self.layer2 = nn.Linear(fc1+out_dims, fc2)
        self.layer3 = nn.Linear(fc2, 1)

        # Initialization and batch norm ideas from
        # https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/lunar-lander/pytorch
        f1 = 1./np.sqrt(self.layer1.weight.data.size()[0])
        nn.init.uniform_(self.layer1.weight.data, -f1, f1)
        nn.init.uniform_(self.layer1.bias.data, -f1, f1)

        f2 = 1./np.sqrt(self.layer2.weight.data.size()[0])
        nn.init.uniform_(self.layer2.weight.data, -f2, f2)
        nn.init.uniform_(self.layer2.bias.data, -f2, f2)

        f3 = 0.003  # specified in the paper
        nn.init.uniform_(self.layer3.weight.data, -f3, f3)
        nn.init.uniform_(self.layer3.bias.data, -f3, f3)

        # self.layer1 = nn.Linear(in_dims, fc1)
        # self.layer2 = nn.Linear(fc1+out_dims, fc2)
        # self.layer3 = nn.Linear(fc2, 1)
        #
        # self.apply(weight_init__uniform_net)

    def forward(self, state, action):

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(torch.cat([x, action], 1)))
        x = self.layer3(x)
        return x


class Actor(nn.Module):
    def __init__(self, in_dims, fc1, fc2, out_dims, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space

        self.layer1 = nn.Linear(in_dims, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, out_dims)

        # Initialization and batch norm ideas from
        # https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/lunar-lander/pytorch
        f1 = 1./np.sqrt(self.layer1.weight.data.size()[0])
        nn.init.uniform_(self.layer1.weight.data, -f1, f1)
        nn.init.uniform_(self.layer1.bias.data, -f1, f1)

        f2 = 1./np.sqrt(self.layer2.weight.data.size()[0])
        nn.init.uniform_(self.layer2.weight.data, -f2, f2)
        nn.init.uniform_(self.layer2.bias.data, -f2, f2)

        f3 = 0.003  # specified in the paper
        nn.init.uniform_(self.layer3.weight.data, -f3, f3)
        nn.init.uniform_(self.layer3.bias.data, -f3, f3)

        # self.layer1 = nn.Linear(in_dims, fc1)
        # self.layer2 = nn.Linear(fc1, fc2)
        # self.layer3 = nn.Linear(fc2, out_dims)
        #
        # self.apply(weight_init__uniform_net)

    def forward(self, state):

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))

        x = x * torch.from_numpy(self.action_space.high).float()

        return x
