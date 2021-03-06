import os
os.environ['PATH'] += r";C:\Users\xavier\.mujoco\mjpro150\bin"
os.add_dll_directory("C://Users//xavier//.mujoco//mjpro150//bin")
import gym
from utils import *
from ddpg import DDPGAgent
from networks import Actor

print('Importing M...')
import mujoco_py
print('Mujoco Imported...')


if __name__ == '__main__':

    '''
    params:
    fc1: nodes first layer
    fc2: nodes second layer
    name_env: name of the env
    early_stop_val: mean of the last 100 test reward score
    batch_size: batch size taken in the learn function at each step (def 64)
    noise: noise implemented {None, 'ou', 'param', 'normal'}
    noise_std: std of the noise
    actor_lr: def 0.0001
    critic_lr: def 0.001
    tau: tau def 0.001
    gamma: def 0.99
    normalize: normalize the obs (not in testing)
    '''

    TRAIN = True
    test_epochs = 5

    fc1 = 400
    fc2 = 300

    environments = {
        # 'pendulum': {
        #     'name_env': 'Pendulum-v1',
        #     'early_stop_val': -200,
        #     'batch_size': 128,
        #     'noise': 'ou',
        #     'noise_std': 0.2,
        #     'normalize': True,
        #     'layer_norm': False,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99},
        # 'mountain': {
        #     'name_env': 'MountainCarContinuous-v0',
        #     'early_stop_val': 90,
        #     'batch_size': 128,
        #     'noise': 'ou',
        #     'noise_std': 0.3,
        #     'normalize': True,
        #     'layer_norm': False,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99},
        # 'lunar': {
        #     'name_env': 'LunarLanderContinuous-v2',
        #     'early_stop_val': 200,
        #     'batch_size': 128,
        #     'noise': 'normal',
        #     'noise_std': 0.1,
        #     'normalize': True,
        #     'layer_norm': False,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99},
        'biped': {
            'name_env': 'BipedalWalker-v3',
            'early_stop_val': 300,
            'batch_size': 128,
            'noise': 'param',
            'noise_std': 0.2,
            'normalize': True,
            'layer_norm': False,
            'actor_lr': 0.0001,
            'critic_lr': 0.001,
            'tau': 0.001,
            'gamma': 0.99},
        # 'pendulum_muj': {
        #     'name_env': 'InvertedPendulum-v2',
        #     'early_stop_val': 900,
        #     'batch_size': 64,
        #     'noise': 'ou',
        #     'noise_std': 0.3,
        #     'normalize': False,
        #     'layer_norm': False,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99},
        # 'cheetah': {
        #     'name_env': 'HalfCheetah-v2',
        #     'early_stop_val': np.inf,
        #     'batch_size': 64,
        #     'noise': 'param',
        #     'layer_norm': False,
        #     'noise_std': 0.2,
        #     'normalize': True,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99},
        # 'ant': {
        #     'name_env': 'Ant-v2',
        #     'early_stop_val': np.inf,
        #     'batch_size': 64,
        #     'noise': 'param',
        #     'layer_norm': True,
        #     'noise_std': 0.2,
        #     'normalize': True,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99},
        # 'human': {
        #     'name_env': 'Humanoid-v2',
        #     'early_stop_val': 1500,
        #     'batch_size': 64,
        #     'noise': 'normal',
        #     'noise_std': 0.1,
        #     'normalize': True,
        #     'actor_lr': 0.0001,
        #     'critic_lr': 0.001,
        #     'tau': 0.001,
        #     'gamma': 0.99}
                   }

    if TRAIN:
        for key, params in environments.items():

            name_env = params['name_env']

            # create env
            env = gym.make(name_env)
            env.action_space.seed(14)

            agent = DDPGAgent(env, fc1=fc1, fc2=fc2,
                              early_stop_val=params['early_stop_val'],
                              batch_size=params['batch_size'],
                              actor_lr=params['actor_lr'],
                              critic_lr=params['critic_lr'],
                              tau=params['tau'],
                              gamma=params['gamma'],
                              noise=params['noise'],
                              noise_std=params['noise_std'],
                              normalize=params['normalize'],
                              layer_norm=params['layer_norm'],
                              )
            agent.train()
            agent.test_agent(True, 5, False)  # render, num of iter (default=1), True to test during training

    else:
        for key, info_env in environments.items():
            name_env = info_env['name_env']
            # create env
            env = gym.make(name_env)
            env.action_space.seed(14)

            n_obs = env.observation_space.shape[0]
            n_acts = env.action_space.shape[0]

            model = Actor(n_obs, fc1, fc2, n_acts, env.action_space)
            model.load_state_dict(torch.load('./Models/' + 'actor_' + name_env + '_model.pth'))

            for i in range(5):
                obs = env.reset()
                rewards = 0
                d = False
                images = []
                for steps in range(50000):
                    env.render()
                    obs = torch.tensor(obs, dtype=torch.float)
                    act = model(obs).detach().numpy()
                    obs_, rew, d, _ = env.step(act)
                    rewards += rew

                    if d:
                        break
                    obs = obs_

                print(f'Episode: {i} | Rewards: {rewards} | steps: {steps}')

            env.close()