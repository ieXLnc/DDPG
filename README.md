# DDPG

Model in training!

### Hyperparams

Actor Net:
- in_dims: n_obs
- fc1: 256 (Relu activation)
- fc2: 128 (Relu activation)
- lr: 0.0001
- out_dims: n_acts (Tanh activation)
- uniform_ init of the weights

Critic:
- in_dims: n_obs
- fc1: 256 (Relu activation)
- fc2: 128 + n_acts (Relu activation)
- lr: 0.001
- out_dims: 1
- uniform_ init of the weights

Agent:
- gamma: 0.99
- tau: 0.01
- batch_size = 128
- reward: rewards for the ep
- test reward: test episode played
- mean reward: mean of last ten 
- mean early val: mean of last 100 test ep


![Pendulum-v0_88528__plot_Trained_perf](https://user-images.githubusercontent.com/63811972/153418561-4d6565fb-b815-4ede-9ead-1b54c21b4a18.png)


