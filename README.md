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


#### Pedulum-v0

![Pendulum-v0_6117__plot](https://user-images.githubusercontent.com/63811972/154112501-d7b8a3a8-df96-412b-be30-b9d859d2df05.png)

![gif_ppo_mod_Pendulum-v0_92089trained](https://user-images.githubusercontent.com/63811972/154115630-8c58345d-1986-43be-8995-6f896ad25ed6.gif)


#### MountainCarContinuous-v0 

![MountainCarContinuous-v0_50795__plot](https://user-images.githubusercontent.com/63811972/154244675-cbdba7a3-b2d3-4514-9791-eef3c034cdd0.png)

![gif_ppo_mod_MountainCarContinuous-v0_92089_BEST](https://user-images.githubusercontent.com/63811972/154248112-eb6d55ae-a715-40a7-be11-7fe90864a3c1.gif)


#### LunarLanderContinuous-v2 

![LunarLanderContinuous-v2_66734__plot](https://user-images.githubusercontent.com/63811972/154249188-ec5e1922-6a63-41bf-adad-c6cbcd213157.png)


![gif_ppo_mod_LunarLanderContinuous-v2_98986trained_bEST](https://user-images.githubusercontent.com/63811972/154248248-7d2e4531-9a0f-40e4-8d58-1d3ec3024f70.gif)


#### BipedalWalker-v3

![gif_ppo_mod_BipedalWalker-v3_96355trained](https://user-images.githubusercontent.com/63811972/154254064-532ddc4f-cda2-4e65-a163-7019fe1d6345.gif)














