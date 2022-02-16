from utils import *
from ddpg import DDPGAgent
from networks import Actor
from PIL import Image

if __name__ == '__main__':

    TRAIN = False
    RECORD = False
    test_epochs = 5

    fc1 = 400
    fc2 = 300

    environments = {
        'pendulum': {
            'name_env': 'Pendulum-v0',
            'early_stop_val': -200,
            'batch_size': 64},
        'mountain': {
            'name_env': 'MountainCarContinuous-v0',
            'early_stop_val': 90,
            'batch_size': 64},
        'lunar': {
            'name_env': 'LunarLanderContinuous-v2',
            'early_stop_val': 200,
            'batch_size': 64},
        'biped': {
            'name_env': 'BipedalWalker-v3',
            'early_stop_val': 250,
            'batch_size': 64}
                   }

    if TRAIN:

        for key, info_env in environments.items():

            name_env = info_env['name_env']
            early_stop__val = info_env['early_stop_val']
            batch_size = info_env['batch_size']

            # create env
            env = NormalizedEnv(gym.make(name_env))
            env.action_space.seed(14)

            agent = DDPGAgent(env, fc1=fc1, fc2=fc2, early_stop_val=early_stop__val, batch_size=batch_size)
            agent.train()
            agent.test_agent(True, 5, False)  # render, num of iter (default=1), True to test during training

    else:

        for key, info_env in environments.items():
            name_env = info_env['name_env']

            # create env
            env = NormalizedEnv(gym.make(name_env))
            env.action_space.seed(14)

            n_obs = env.observation_space.shape[0]
            n_acts = env.action_space.shape[0]

            model = Actor(n_obs, fc1, fc2, n_acts, env.action_space)
            model.load_state_dict(torch.load('./Models/' + 'actor_' + name_env + '_model.pth'))

            for i in range(test_epochs):
                obs = env.reset()
                rewards = 0
                d = False
                images = []
                for steps in range(1000):
                    if i % 2 == 0 and RECORD:
                        # Render to frames buffer
                        image = (env.render(mode="rgb_array"))
                        image = Image.fromarray(image)
                        images.append(image)

                    env.render()
                    obs = torch.tensor(obs, dtype=torch.float)
                    act = model(obs).detach().numpy()
                    obs_, rew, d, _ = env.step(act)
                    rewards += rew

                    if d:
                        break
                    obs = obs_

                print(f'Episode: {i} | Rewards: {rewards} | steps: {steps}')

                if i % 2 == 0 and RECORD:  # Record
                    num = random.randint(0, 100000)
                    gif(images, 'gif_ppo_mod_' + name_env + "_" + str(num) + 'trained.gif')

            env.close()
