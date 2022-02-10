import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from networks import *
from utils import ReplayBuffer
import pickle

torch.manual_seed(14)
# set GPU for faster training
cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class DDPGAgent:
    def __init__(self, env, fc1=256, fc2=128, batch_size=128, early_stop_val=-200):

        self.env = env
        self.n_obs = env.observation_space.shape[0]
        self.n_acts = env.action_space.shape[0]
        self.early_stop_val = early_stop_val
        self.early_stop = False

        # hyperparams
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.001

        # Create the networks
        self.fc1 = fc1
        self.fc2 = fc2

        self.actor = Actor(self.n_obs, self.fc1, self.fc2, self.n_acts).to(device)
        self.actor_target = Actor(self.n_obs, self.fc1, self.fc2, self.n_acts).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(self.n_obs, self.fc1, self.fc2, self.n_acts).to(device)        # here out dims to process actions
        self.critic_target = Critic(self.n_obs, self.fc1, self.fc2, self.n_acts).to(device)  # here out dims to process actions
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

        # setup weights
        for target_params, params in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_params.data.copy_(params.data)
        for target_params, params in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_params.data.copy_(params.data)

        # Memory
        self.memory_size = 100_000
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.memory_size, self.batch_size)

        # saving modalities
        self.name_env = env.unwrapped.spec.id
        self.name = self.name_env + '_model.pth'

        # Create log
        self.log = {
            'rewards': [],
            'rewards_ep': -np.inf,
            'mean_rewards': [],
            'best_score': [],
            'actor_loss': [0],
            'critic_loss': [0],
            'episode': 0,
            'batch_size': self.batch_size,
            'test_rew':[]
        }

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        action = self.actor(state)

        return action.detach().numpy()

    def update_models(self):

        if self.replay_buffer.len_() < self.batch_size:
            return

        states, actions, rewards, dones, next_states = self.replay_buffer.get_sample()

        # -------- Calculate losses --------
        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q

        critic_loss = nn.MSELoss()(Qvals, Qprime)
        self.log['critic_loss'].append(critic_loss.detach().numpy())

        # Actor loss
        policy_loss = - self.critic.forward(states, self.actor.forward(states)).mean()
        self.log['actor_loss'].append(policy_loss.detach().numpy())

        # -------- Update networks --------
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Soft update of nets --------
        for target_params, params in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)
        for target_params, params in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)

    def save_models(self):
        torch.save(self.actor.state_dict(), './Models/' + 'actor_' + self.name)
        torch.save(self.actor_target.state_dict(), './Models/' + 'actor_target_' + self.name)
        torch.save(self.critic.state_dict(), './Models/' + 'critic_' + self.name)
        torch.save(self.critic_target.state_dict(), './Models/' + 'critic_target_' + self.name)
        self.replay_buffer.save_memory(self.name)

    def test_agent(self, render=False, n_test=1, test_train=True):
        rewards_ep = []
        for i in range(n_test):
            state = self.env.reset()
            rewards = 0
            d = False
            while not d:
                if render: self.env.render()
                action = self.get_action(state)
                new_state, rew, d, _ = self.env.step(action)
                rewards += rew
                if d:
                    break
                state = new_state

            rewards_ep.append(rewards)
        if test_train:
            return rewards
        else:
            for i in range(n_test):
                print(f'Tests episodes {i} with rewards: {rewards_ep[i]}')

    def summary(self):

        if len(self.log['rewards']) == 0:
            best_current_score = -np.inf
        else:
            best_current_score = self.log['rewards'][np.argmax(self.log['rewards'])]

        last_val = self.log['rewards_ep']

        self.log['rewards'].append(self.log['rewards_ep'])
        self.log['mean_rewards'].append(np.mean(self.log['rewards'][-10:]))
        mean_early_stop = np.mean(self.log['test_rew'][-100:])

        print(f'-------------------------------------------------')
        print(f'----------- Episode #{self.log["episode"]}-------------------')
        print(f'Rewards for the episode: {self.log["rewards_ep"]}')
        print(f'Mean value for last 10 {self.log["mean_rewards"][-1]}')
        if last_val > best_current_score:
            self.save_models()
            with open('./Models/logger_' + self.name_env + '.pkl', 'wb') as handle:
                pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'New models saved with {last_val}')
        else:
            print(f'Best model: {best_current_score}')

        print(f'test rewards: {self.log["test_rew"][-1]}')
        print(f'mean early stop is currently: {mean_early_stop}')
        print(f'Actor loss: {self.log["actor_loss"][-1]}')
        print(f'Critic loss: {self.log["critic_loss"][-1]}')
        print(f'With Batch size of {self.log["batch_size"]}')

        if mean_early_stop > self.early_stop_val:
            self.early_stop = True
            self.save_models()
            with open('./Models/logger_' + self.name_env + '.pkl', 'wb') as handle:
                pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Early stop activated with score {last_val} at episode {self.log["episode"]}')
        print(f'-------------------------------------------------')
