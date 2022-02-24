import numpy as np
import torch # Torch version :1.9.0+cpu
from torch import nn
from torch.nn import functional as F

torch.manual_seed(14)
np.random.default_rng(14)

cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class Critic(nn.Module):
    def __init__(self, in_dims, fc1, fc2, out_dims, layer_norm=False):
        super(Critic, self).__init__()
        self.layer_norm = layer_norm

        self.layer1 = nn.Linear(in_dims, fc1)
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(fc1)
        self.layer2 = nn.Linear(fc1+out_dims, fc2)
        if self.layer_norm:
            self.ln2 = nn.LayerNorm(fc2)
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

    def forward(self, state, action):

        x = self.layer1(state)
        if self.layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.layer2(torch.cat([x, action], 1))
        if self.layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        x = self.layer3(x)

        return x


class Actor(nn.Module):
    def __init__(self, in_dims, fc1, fc2, out_dims, action_space, layer_norm=False):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.layer_norm = layer_norm

        self.layer1 = nn.Linear(in_dims, fc1)
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        if self.layer_norm:
            self.ln2 = nn.LayerNorm(fc2)
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

    def forward(self, state):

        x = self.layer1(state)
        if self.layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.layer2(x)
        if self.layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        x = torch.tanh(self.layer3(x))

        x = x * torch.from_numpy(self.action_space.high).float().to(device)

        return x
