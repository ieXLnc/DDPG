# DDPG


Pytorch implementation of the a Deep Deterministic Policy Gradient agent following the [continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) paper by Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.

The agent is implemented to solve multiple gym environments  (pendulum, mountaincar, lunarlander, bipedwalker) and mujoco environment (halfcheetah, ant, humanoid). 

I tried to implement several noise options:
- Ornstein-Ulhenbeck noise
- Gaussian noise
- Adaptive noise

### The model takes the following hyperparameters by default:
- fc1=400 (nodes on the first dense layer)
- fc2=300 (nodes on the second dense layer)
- gamma=0.99
- tau=0.001
- actor_lr=0.0001
- critic_lr=0.001
- batch_size=64
- normalize=False (normalize environment with gym wrapper)
- noise=None
- noise_std=0.3
- layer_norm=False (LayerNormalisation on dense layers if set to True)
- interval_adapt=5 (if param noise, update the stddev every five episodes)
- early_stop_val=-200 (if want to reach a maximum score, based on the last 100 test rewards score)
- early_stop_timesteps=2e6

### The two nets

Actor Net:
- in_dims: n_obs
- fc1: 400 (Relu activation)
- fc2: 300 (Relu activation)
- out_dims: n_acts (Tanh activation) * action_space.high
- uniform_ init of the weights

Critic:
- in_dims: n_obs
- fc1: 400 (Relu activation)
- fc2: 300 + n_acts (Relu activation)
- out_dims: 1
- uniform_ init of the weights


#### Pedulum-v0

![Pendulum-v1_model_ou_0 2_normalize_True_ln_False pth_81266__plot](https://user-images.githubusercontent.com/63811972/155567682-20b01367-ab8f-4a98-922f-cbfd1bb83168.png)


![gif_ppo_mod_Pendulum-v0_92089trained](https://user-images.githubusercontent.com/63811972/154115630-8c58345d-1986-43be-8995-6f896ad25ed6.gif)


#### MountainCarContinuous-v0 

![MountainCarContinuous-v0_model_ou_0 3_normalize_True_ln_False pth_50145__plot](https://user-images.githubusercontent.com/63811972/155578130-d66c51de-e7bd-47c8-a899-8bf1442844e1.png)

![gif_ppo_mod_MountainCarContinuous-v0_92089_BEST](https://user-images.githubusercontent.com/63811972/154248112-eb6d55ae-a715-40a7-be11-7fe90864a3c1.gif)


#### LunarLanderContinuous-v2 

![LunarLanderContinuous-v2_model_normal_0 1_normalize_True_ln_False pth_73736__plot](https://user-images.githubusercontent.com/63811972/155745710-40844708-5bcb-4085-a947-b595d943b659.png)

![gif_ppo_mod_LunarLanderContinuous-v2_98986trained_bEST](https://user-images.githubusercontent.com/63811972/154248248-7d2e4531-9a0f-40e4-8d58-1d3ec3024f70.gif)


#### BipedalWalker-v3

![gif_ppo_mod_BipedalWalker-v3_96355trained](https://user-images.githubusercontent.com/63811972/154254064-532ddc4f-cda2-4e65-a163-7019fe1d6345.gif)


#### InvertedPendulum-v2

![InvertedPendulum-v2_96925__plot](https://user-images.githubusercontent.com/63811972/154674444-274a71f5-eafd-4d43-a1c0-b75a50106148.png)



#### HalfCheetah-v2

![HalfCheetah-v2_model_param_0 2_normalize_True_ln_False pth_6163__plot](https://user-images.githubusercontent.com/63811972/155986547-2ea61c82-1ba9-4114-80c3-9dc31c538345.png)


https://user-images.githubusercontent.com/63811972/155771505-aa285672-da71-43d9-afa6-fad2122d00bc.mp4


#### Ant-v2


![Ant-v2_model_param_0 2_normalize_True_ln_True pth_29968__plot](https://user-images.githubusercontent.com/63811972/155986521-28bba9a2-23d7-4f22-9b14-a384278f992c.png)


https://user-images.githubusercontent.com/63811972/155986483-c4215b9f-46df-48ad-ac1f-f260e3a1e447.mp4















