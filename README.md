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

![Pendulum-v0_6117__plot](https://user-images.githubusercontent.com/63811972/154112501-d7b8a3a8-df96-412b-be30-b9d859d2df05.png)

![gif_ppo_mod_Pendulum-v0_92089trained](https://user-images.githubusercontent.com/63811972/154115630-8c58345d-1986-43be-8995-6f896ad25ed6.gif)





