import matplotlib.pyplot as plt
from utils import *
from ddpg import DDPGAgent


if __name__ == '__main__':

    # select env and
    pendulum = 'Pendulum-v0'
    lunar = 'LunarLanderContinuous-v2'
    name_env = pendulum
    # Create env
    env = NormalizedEnv(gym.make(name_env))
    env.action_space.seed(14)

    # Create agent
    agent = DDPGAgent(env, batch_size=512, early_stop_val=-150)
    noise = OUNoise(env.action_space)

    TRAINING_EP = 100_000

    all_rewards = []
    mean_rewards = []
    best_score = -np.inf

    for ep in range(500):
        state = env.reset()
        noise.reset()
        rewards = 0

        for step in range(200):

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
                reward_ep = np.mean([agent.test_agent() for _ in range(3)])
                agent.log['test_rew'].append(reward_ep)
                break

        agent.summary()

        if agent.early_stop:
            break

    plt.plot(agent.log['rewards'])
    plt.plot(agent.log['mean_rewards'])
    plt.plot(agent.log['test_rew'])
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./Plots/' + name_env)
    plt.show()

    agent.test_agent(True, 5, False)







