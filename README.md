# DDPG

Model in training!

### Hyperparams

Actor Net:
- in_dims: n_obs
- fc1: 400 (Relu activation)
- fc2: 300 (Relu activation)
- lr: 0.00005
- out_dims: n_acts (Tanh activation) * action_space.high
- uniform_ init of the weights

Critic:
- in_dims: n_obs
- fc1: 400 (Relu activation)
- fc2: 300 + n_acts (Relu activation)
- lr: 0.0005
- out_dims: 1
- uniform_ init of the weights

Agent:
- gamma: 0.99
- tau: 0.001
- batch_size = 64
- reward: rewards for the ep
- mean reward: mean of last 10 rewards
- test reward: test episode played
- mean early val: mean of last 100 test ep


![Pendulum-v0_88528__plot_Trained_perf](https://user-images.githubusercontent.com/63811972/153418561-4d6565fb-b815-4ede-9ead-1b54c21b4a18.png)


