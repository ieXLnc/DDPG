import numpy as np
import torch # Torch version :1.9.0+cpu
from torch import nn
from torch.optim import Adam
from networks import *
from utils import *
import pickle
import matplotlib.pyplot as plt

torch.manual_seed(14)
# set GPU for faster training
cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class DDPGAgent:
    def __init__(self, env, fc1=400, fc2=300, gamma=0.99, tau=0.01, actor_lr=0.01, critic_lr=0.001,
                 batch_size=64, early_stop_val=-200, normalize=False, noise=None,
                 noise_std=0.3, layer_norm=True, interval_adapt=5, early_stop_timesteps=2e6):

        self.env = env
        self.n_obs = env.observation_space.shape[0]
        self.n_acts = env.action_space.shape[0]
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.normalize = normalize

        self.early_stop_val = early_stop_val
        self.early_stop = False
        self.early_stop_timesteps = early_stop_timesteps

        # hyperparams
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.min = -np.inf
        self.max = np.inf

        # Create the networks
        self.fc1 = fc1
        self.fc2 = fc2
        self.layer_norm = layer_norm

        self.actor = Actor(self.n_obs, self.fc1, self.fc2, self.n_acts, self.env.action_space, self.layer_norm).to(device)
        self.actor_target = Actor(self.n_obs, self.fc1, self.fc2, self.n_acts, self.env.action_space, self.layer_norm).to(device)
        self.actor_perturbed = Actor(self.n_obs, self.fc1, self.fc2, self.n_acts, self.env.action_space, self.layer_norm).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(self.n_obs, self.fc1, self.fc2, self.n_acts, self.layer_norm).to(device)        # here out dims to process actions
        self.critic_target = Critic(self.n_obs, self.fc1, self.fc2, self.n_acts, self.layer_norm).to(device)  # here out dims to process actions
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

        # Noise
        self.noise_type = noise
        self.noise_std = noise_std
        self.initial_stddev = 0.05   # initial std
        # self.desired_action_stddev = 0.3
        self.adoption_coefficient = 1.01
        if self.noise_type is not None:
            if self.noise_type == 'ou':
                self.OU_noise = OUNoise(self.env.action_space, max_sigma=self.noise_std)
                print('Ou noise used')
            if self.noise_type == 'param':
                self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=self.initial_stddev, desired_action_stddev=self.noise_std)
                self.interval_adapt = interval_adapt
                self.distance = []
                self.stddev = []
                print('Adaptive noise used')
            if self.noise_type == 'normal':
                self.Gauss_noise = GaussianStrategy(self.env.action_space, max_sigma=self.noise_std)
                print('Gaussian noise used')
        if self.noise_type is None:
            print('No noise added.')

        # saving modalities
        self.name_env = env.unwrapped.spec.id
        self.name = self.name_env + '_model_' + str(self.noise_type) + '_' + str(self.noise_std) + '_normalize_' \
                    + str(self.normalize) + '_ln_' + str(self.layer_norm) + '.pth'

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
            'test_rew': [],
            'timesteps': 0
        }

    def get_perturbed_actor(self):
        # make a hardcopy of the actor
        for target_params, params in zip(self.actor_perturbed.parameters(), self.actor.parameters()):
            target_params.data.copy_(params.data)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param.data += torch.normal(mean=torch.zeros(param.shape),
                                       std=self.param_noise.current_stddev).to(device)

    def ddpg_distance_metric(self, actions1, actions2):
        """
        Compute "distance" between actions taken by two policies at the same states
        Expects numpy arrays
        """
        diff = actions1 - actions2
        mean_diff = np.mean(np.square(diff), axis=0)
        dist = np.sqrt(np.mean(mean_diff))
        return dist

    def get_action(self, state, step, evaluate=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)

        action = self.actor(state).detach().cpu().numpy()

        if not evaluate or self.noise_type is not None:              # if eval no noise, if noise=None same

            if self.noise_type == 'ou':
                action = self.OU_noise.get_action(action, step)      # removed the clipped to put it here

            if self.noise_type == 'param':
                action = self.actor_perturbed(state).detach().cpu().numpy()

            if self.noise_type == 'normal':
                action = self.Gauss_noise.get_action(action, step)

        return np.clip(action, self.low, self.high)

    def update_models(self):

        if self.replay_buffer.len_() < self.batch_size:
            return

        states, actions, rewards, dones, next_states = self.replay_buffer.get_sample()

        # -------- Calculate losses --------
        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + (1.0 - dones) * self.gamma * next_Q
        # Qprime = torch.clamp(Qprime, self.min, self.max)

        critic_loss = nn.MSELoss()(Qvals, Qprime)
        self.log['critic_loss'].append(critic_loss.detach().cpu().numpy())

        # Actor loss
        policy_loss = - self.critic.forward(states, self.actor.forward(states)).mean()
        self.log['actor_loss'].append(policy_loss.detach().cpu().numpy())

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

    def train(self):
        ep = 1
        while not self.early_stop:
            if self.normalize:
                self.env = NormalizedEnv(self.env)
            state = self.env.reset()
            if self.noise_type == 'ou':
                self.OU_noise.reset()
            if self.noise_type == 'param' and ep % self.interval_adapt == 0:
                self.get_perturbed_actor()

            rewards = 0
            action_noise = []
            states_noise = []

            for step in range(1000):
                action = self.get_action(state, step)  # np array with one val .detach()
                new_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.memorize(state, action, reward, done, new_state)
                self.update_models()

                rewards += reward
                state = new_state

                action_noise.append(action)
                states_noise.append(state)

                if done:
                    break

            self.log['rewards_ep'] = rewards
            self.log['episode'] = ep
            reward_ep = np.mean([self.test_agent() for _ in range(1)])
            self.log['test_rew'].append(reward_ep)
            self.log['timesteps'] += step

            self.summary()

            if self.noise_type == 'param' and ep % self.interval_adapt == 0:
                states_noise = torch.tensor(states_noise, dtype=torch.float).to(device)
                unperturbed_actions = self.actor(states_noise).detach().cpu().numpy()
                perturbed_actions = np.asarray(action_noise)
                distance = self.ddpg_distance_metric(perturbed_actions, unperturbed_actions)
                self.param_noise.adapt(distance)
                print('distance = ', distance)
                self.distance.append(distance)
                print('current stddev = ', self.param_noise.current_stddev)
                self.stddev.append(self.param_noise.current_stddev)

            if self.early_stop:
                break

            ep += 1

        if self.noise_type =='param':
            plt.plot(self.distance)
            plt.savefig('./Plots/distance_' + str(self.noise_type))
            plt.close()
            plt.plot(self.stddev)
            plt.savefig('./Plots/stddev' + str(self.noise_type))
            plt.close()


        plot(self.log['rewards'], self.log['mean_rewards'], self.log['test_rew'], self.log['actor_loss'],
             self.log['critic_loss'], self.name_env, self.name)

    def test_agent(self, render=False, n_test=1, test_train=True):
        rewards_ep = []
        for i in range(n_test):
            state = self.env.reset()
            rewards = 0
            step = 0
            d = False
            while not d:
                if render: self.env.render()
                action = self.get_action(state, step, True)
                new_state, rew, d, _ = self.env.step(action)
                rewards += rew
                if d:
                    break
                state = new_state
                step += 1

            rewards_ep.append(rewards)
        if test_train:      # get the reward of the ep to test the model on an iteration basis
            return rewards
        else:               # just test the model after its done converging
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
        print(f'Timesteps: {self.log["timesteps"]}')

        if mean_early_stop > self.early_stop_val or self.log["timesteps"] >= 100_000:
            self.early_stop = True
            self.save_models()
            with open('./Models/logger_' + self.name_env + '.pkl', 'wb') as handle:
                pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Early stop activated with score {last_val} at episode {self.log["episode"]}')
        print(f'-------------------------------------------------')
