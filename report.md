# Report


## Learning Algorithm - Policy based methods  - DDPG



## Algorithm



**Improvements**:


## Hyper parameters and Other Changes


I modified the architecture of the Actor and Critic Neural networks.


* Activation function  - Leaky Relu for all layers except the last one.  Using torch.tanh for the final one
* I'm also using Batchnorm1D which required a hack for the states dimension for single agent case
* I'm using dropout in the Critic
* I had the **batch_size as 512**


Tuning other hyperparameters might help converge even faster

* BUFFER_SIZE = int(1e6)  # replay buffer size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 2e-4         # learning rate of the actor
* LR_CRITIC = 3e-4        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay
* HIDDEN_LAYERS=(512,256) # Modified the architecture to include an additional hidden layers with 512 and 256 units
* UPDATE_EVERY = 20       # 20 for 20 agents case and 4 for single
* DROPOUT =0.2



## Plot of Rewards


 * **Single agent**:



        Episode 100	Average Score: 2.64
        Episode 200	Average Score: 20.53
        Episode 254	Average Score: 30.05
        Environment solved in 254 episodes!	Average Score: 30.05


![alt text](https://github.com/snknitin/Navigation-RL/blob/master/curve-single.PNG)

* **20 Agents**


## Ideas for Future Work



*