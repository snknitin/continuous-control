# Report


## Learning Algorithm - Policy based methods  - DDPG


We can find the optimal policy through interactions without having to worry about first figuring out the optimal value function.Policy-based methods are a class of algorithms that search directly for the optimal policy, without simultaneously maintaining value function estimates. Policy gradient methods are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent

**Actor Critic methods** are at the intersection of value based and policy based methods like dqn or reinforce respectively. If a DRL agent uses a NN to approximate a value function, it is value-based, if it approximates a policy, then it is policy based. We can a NN to approximate a value function and use it as a baseline to reduce the variance of policy based methods.

## DDPG - Deep Deterministic Policy Gradient, Continuous Action-space

In the DDPG paper, they introduced this algorithm as an "Actor-Critic" method. Though, some researchers think DDPG is best classified as a DQN method for continuous action spaces (along with NAF). DDPG is a different kind of AC. The critic in DDPG is used to approximate the maximizer over the Q values of the next state, and not as a learned baseline. One of the limitations of the DQN agent is that it is difficult to use it in continuous action spaces. It is easy to do a max over different actions from the action-value function in DQN because it is a discrete space. Even if it is a high dimensional action space with many actions, it would still be doable. But if the action space is continuous, DDPG solves this using AC (2 NN)


## Algorithm

The actor here is used to approximate an optimal policy deterministically μ(s;θμ),ie., always output the best believed action for any given state which is unlike the stochastic policy given by π(a|s;θπ) where we want to learn a probability distribution over the actions. In DDPG we want the believed best action every single time we query the actor network. That is a deterministic policy where the actor is basically learning argmax_a Q(s,a) and the critic learns to evaluate the optimal action value function using the actor's best believed action. So instead of V(s;θ) we learn Q(s,μ(s;θμ);θQ). Again, we use this actor which is an approximate maximizer to calculate a new target value for training the action value function, much like DQN.


**Improvements**:

2 other intersting aspects of DDPG are -
* Use of a replay buffer
* Soft updates to target network

In DQN we have 2 copies of NN weights - local and target. Local is the one that is most up to date because we are training, while target is the one we use for prediction to stabilize the training. After every C=10000 steps ,we copy the local network weights to the target. The target network is fixed for 10000 time steps and then gets a big update. **In DDPG we have local and target for both, actor and critic**- 4 sets of weights, but the target networks are updated using a soft update strategy - Slowly blend your regular weights with your target weights. **So every step mix in 0.01% of regular network weights with target weights.** We get faster convergence by using this update strategy.


## Hyper parameters and Other Changes


I modified the architecture of the Actor and Critic Neural networks. I tried tuning several hyperparameters like the learning rates, batchsize and the weight decay


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
            * DROPOUT = 0.2
            * num_update = 10



## Plot of Rewards


 * **Single agent**:



        Episode 100	Average Score: 2.64
        Episode 200	Average Score: 20.53
        Episode 254	Average Score: 30.05
        Environment solved in 254 episodes!	Average Score: 30.05


![alt text](https://github.com/snknitin/continuous-control/blob/master/curve-single.PNG)

* **20 Agents**

        Episode 96	Average Score: 30.02
        Environment solved in 96 episodes!	Average Score: 30.02


![alt text](https://github.com/snknitin/continuous-control/blob/master/curve-twenty.PNG)

## Ideas for Future Work

* Try optimizing further using shared learning from the single agent case which trained relatively fast, and use those weights as the starting point for each of the 20 agents
* I would like to try out Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG) since literature suggests it should achieve better performance.
* Even Proximal Policy Optimization (PPO), as it is known to give good performance with continuous control tasks.

From the note mentioned in the project page, i came across Distributed Distributional Deterministic Policy Gradients (D4PG) algorithm as another method for adapting DDPG for continuous control, which seems interesting.