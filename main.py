import matplotlib.pyplot as plt
from utils import *
from ddpg import DDPGAgent
from networks import Actor
from PIL import Image


if __name__ == '__main__':

    # select env and
    pendulum = 'Pendulum-v0'
    lunar = 'LunarLanderContinuous-v2'
    name_env = pendulum
    # Create env
    env = NormalizedEnv(gym.make(name_env))
    env.action_space.seed(14)

    TESTING = False
    RECORD = False

    if TESTING:

        name = './Models/actor_' + name_env + '_model.pth'

        n_obs = env.observation_space.shape[0]
        fc1 = 256
        fc2 = 128
        n_acts = env.action_space.shape[0]

        print('using model:', name)
        model = Actor(n_obs, fc1, fc2, n_acts)
        model.load_state_dict(torch.load(name))

        TEST_MOD = 5
        max_steps = 1000
        for i in range(TEST_MOD):
            obs = env.reset()
            rewards = 0
            done = False
            images = []
            for step in range(max_steps):
                if i % 2 == 0 and RECORD:
                    # Render to frames buffer
                    image = (env.render(mode="rgb_array"))
                    image = Image.fromarray(image)
                    images.append(image)

                env.render()
                obs = torch.tensor(obs, dtype=torch.float)
                act = model(obs).detach().numpy()
                obs_, rew, done, _ = env.step(act)
                rewards += rew
                if done:
                    break
                step += 1
                obs = obs_

            print(f'Episode: {i} | Rewards: {rewards} | Steps taken: {step}')

            if i % 2 == 0 and RECORD:  # Record
                num = random.randint(0, 100000)
                gif(images, 'gif_ppo_mod_' + name_env + "_" + str(num) + 'trained.gif')

        env.close()

    else:
        # Create agent
        agent = DDPGAgent(env, batch_size=128, early_stop_val=-200)
        noise = OUNoise(env.action_space)

        TRAINING_EP = 100_000

        all_rewards = []
        mean_rewards = []
        best_score = -np.inf

        for ep in range(TRAINING_EP):
            state = env.reset()
            noise.reset()
            rewards = 0

            for step in range(1000):

                action = agent.get_action(state)                 # np array with one val .detach()
                action_noise = noise.get_action(action, step)
                new_state, reward, done, _ = env.step(action_noise)

                agent.replay_buffer.memorize(state, action, reward, done, new_state)

                rewards += reward

                agent.update_models()

                state = new_state

                if done:
                    agent.log['rewards_ep'] = rewards
                    agent.log['episode'] = ep
                    reward_ep = np.mean([agent.test_agent() for _ in range(1)])
                    agent.log['test_rew'].append(reward_ep)
                    break

            agent.summary()

            if agent.early_stop:
                break

        plot(agent.log['rewards'], agent.log['mean_rewards'], agent.log['test_rew'], agent.log['actor_loss'],
             agent.log['critic_loss'], name_env)

        agent.test_agent(True, 5, False)

