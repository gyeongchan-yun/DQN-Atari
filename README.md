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

