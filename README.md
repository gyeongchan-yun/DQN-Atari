Playing Atari with Deep Reinforcement Learning
==============================================  

**Abstract**
   
A model is a *convolutional neural network*, trained with a variant of *Q-learning*, whose input is *raw pixels* and whose output is a *value function*.  
  
**Introduction**  

Most successful RL applications have relied on *hand-crafted features* combined with linear value functions or policy representations. Recent advances in deep learning have made it possible to extract high-level features from raw sensory data.  
 
several challenges from a deep learning perspective:  
1. large amounts of handlabelled training data.
2. The *delay* between actions and resulting rewards.
3. RL encounters sequences of highly *correlated* states.  
  
Use an **experience replay** mechanism  which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors to alleviate the problems of correlated data and non-stationary distributions.  
  
**Background**  
  
Define environment E - Atari emulator<br>  
At each time-step, the agent selects an action at from the set of legal game actions, A = {1, . . . , K}.
The action is passed to the emulator and modifies its internal state and the game score.<br>  
The agent observes an *image* ![equation](https://latex.codecogs.com/gif.latex?x_%7Bt%7D%20%5Cin%20R%5E%7B_%7Bd%7D%7D) from the emulator, which is a vector of raw pixel values representing the current screen and *reward* ![equation](https://latex.codecogs.com/gif.latex?r_%7Bt%7D) representing the change in game score instead of internal state.<br>  
Since it is impossible to fully understand the current situation
from only the current screen, they consider sequences of actions and observations, ![equation](https://latex.codecogs.com/gif.latex?s_%7Bt%7D%20%3D%20x_%7B1%7D%2C%20a_%7B1%7D%2C%20x_%7B2%7D%2C%20a_%7B2%7D%2C%20...%2Ca_%7Bt-1%7D%2Cx_%7Bt%7D) and learn game strategies that depend upon these sequences. As a result, they can apply standard reinforcement learning methods for MDP(Markov Decision Process) where each sequence is a distinct state.

The goal of the agent: Maximize future rewards.  
Make assumption: future rewards are discounted by a factor of ![equation](https://latex.codecogs.com/gif.latex?%5Cgamma) per time-step.  
Define the future reward at time t as ![image](https://user-images.githubusercontent.com/30262658/50749953-9027f980-1285-11e9-81b3-9a4a5a5910a9.png) where T is the terminal time.  
Define optimal action-value function ![equation](https://latex.codecogs.com/gif.latex?Q%5E%7B*%7D%28s%2Ca%29%20%3D%20max_%7B%5Cpi%20%7DE%5BR_%7Bt%7D%7Cs_%7Bt%7D%3Ds%2C%20a_%7Bt%7D%3Da%2C%5Cpi%5D) where ![equation](https://latex.codecogs.com/gif.latex?%5Cpi) is a policy mapping sequences to actions (distributions over actions)  
The basic idea behind RL algorithm is to estimate the action-value function, by using *Bellan equation* as an iterative update, ![equation](https://latex.codecogs.com/gif.latex?Q_%7Bi&plus;1%7D%28s%2Ca%29%20%3D%20E%5Br&plus;%5Cgamma%20max_%7Ba%5E%7B%27%7D%7DQ_%7Bi%7D%28s%5E%7B%27%7D%2Ca%5E%7B%27%7D%29%7C%20s%2Ca%5D) which is called *value-iteration algorithm.*  
However, this approach is impractical because action-value function is estimated separately[독립적으로] for each sequence. (각각의 sequence (고차원의 data)에 대해서 함수를 estimate하기에는 시간과 메모리 문제) Therefore, it's common to use a function approximator (sequence의 경향성(parameter)을 통해 함수화 시켜놓는 것), ![image](https://user-images.githubusercontent.com/30262658/50748471-f0666d80-127c-11e9-9b6b-f66a978a600a.png)  
They refer to a neural network function approximator with weights ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) as a *Q-network.* A Q-network can be trained by minimising a sequence of loss functions ![equation](https://latex.codecogs.com/gif.latex?L_%7Bi%7D%28%5Ctheta%20_%7Bi%7D%29) that changes at each
iteration i, ![image](https://user-images.githubusercontent.com/30262658/50748550-5eab3000-127d-11e9-9bdf-b0c12086a612.png)  
The parameters from the previous iteration ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta%20_%7Bi-1%7D) are held fixed when optimising the loss function. (target의 값이 θ의 값에 민감하게 영향을 받기 때문에 stable한 learning을 위하여 θ값을 고정하는 것이다)  Differentiating the loss function
with respect to the weights we arrive at the following gradient, ![image](https://user-images.githubusercontent.com/30262658/50748660-d2e5d380-127d-11e9-836d-0bee5b942a04.png)

**Deep Reinforcement Learning**
- **Experience replay**
  - store the agent’s experiences at each time-step ![equation](https://latex.codecogs.com/gif.latex?e_%7Bt%7D%20%3D%20%28s_%7Bt%7D%2C%20a_%7Bt%7D%2C%20r_%7Bt%7D%2Cs_%7Bt&plus;1%7D%29) in data-set ![equation](https://latex.codecogs.com/gif.latex?D%3D%28e_%7B1%7D%2C%20e_%7B2%7D%2C...%2Ce_%7BN%7D%29) pooled over many episodes into a *replay memory.*  
  - In practice, the algorithm in the paper only stores the last N experiences and samples uniformly at random form D.
-  After performing experience replay,
the agent selects and executes an action according to an ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon)-greedy policy.
-  our Q-function instead works on fixed length representation of histories produced by a function ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)  

The full algorithm, which we call deep Q-learning, is presented in Algorithm 1.
![image](https://user-images.githubusercontent.com/30262658/50749271-2b6aa000-1281-11e9-86ee-b53877658367.png)  
- Advantage:
1. each step of experience is potentially used in many weight updates, which allows for greater data efficiency.
2. randomizing the samples breaks these correlations and therefore reduces the variance of the updates.
3. when learning on-policy the current parameters determine the next
data sample that the parameters are trained on.
- The algorithm is *model-free* and *off-policy*
  - model-free: Agent learns Trial-and-Error not planning(model-based)
  - off-policy: Divide a policy which is refered to action and one which is updated. (on-policy uses the same policy on two parts.)  

**Pre-processing and Model Architecture**
- raw Atari frames: 210 x 160 pixel images with 128 color palette.
- gray-scale and down-sampling it to a 110 x 84 image.
- crop 84 x 84 image.
- the function ![equation](https://latex.codecogs.com/gif.latex?%5Cphi) implies pre-processing to the last 4 frames of a history and stacks them to produce the input to the Q-function.
-  use an architecture in which there is a separate output unit for each possible action, and only the state representation is an input to the neural network.

**Experiments**  
- Since the scale of scores varies greatly from game to game, we
fixed all positive rewards to be 1 and all negative rewards to be −1, leaving 0 rewards unchanged.
- used the *RMSProp* algorithm with minibatches of size 32.
- use a simple *frame-skipping technique*. More precisely, the agent sees and selects actions on every kth frame instead of every frame.
