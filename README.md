# DDPG

Model in training!

### Hyperparams

Actor Net:
- in_dims: n_obs
- fc1: 256 (Relu activation)
- fc2: 128 (Relu activation)
- lr: 0.00005
- out_dims: n_acts (Tanh activation) * action_space.high
- uniform_ init of the weights

Critic:
- in_dims: n_obs
- fc1: 256 (Relu activation)
- fc2: 128 + n_acts (Relu activation)
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

![Pendulum-v0_62871__plot](https://user-images.githubusercontent.com/63811972/154078904-558c3b06-4ac5-42f9-b88a-b9398896c4b9.png)



