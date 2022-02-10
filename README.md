# DDPG

Model in training!

### Hyperparams

Actor Net:
- in_dims: n_obs
- fc1: 256
- fc2: 128
- lr: 0.0001
- out_dims: n_acts

Critic:
- in_dims: n_obs
- fc1: 256
- fc2: 128 + n_acts
- lr: 0.001
- out_dims: 1

Agent:
- gamma: 0.99
- tau: 0.01
- batch_size = 128
- reward: rewards for the ep
- test reward: test episode played
- mean reward: mean of last ten 
- mean early val: mean of last x test ep

![Pendulum-v0_4625__plot_Trained_100rewards](https://user-images.githubusercontent.com/63811972/153404523-33a91f85-7572-4a58-9c01-d241b717abe6.png)
