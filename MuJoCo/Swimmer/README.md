# The Swimmer
## Overview
The Swimmer is a classic control environment in the OpenAI gym, where the goal is to make a 2D swimmer move forward as quickly as possible by controlling its body movements. The environment is continuous, meaning the agent has to select a continuous action at each time step to control the swimmer. The observation space consists of the swimmer's position, velocity, and body angle, and the agent receives a small reward at each time step for making the swimmer move forward.

## Solution
We trained an agent to solve the Swimmer environment using the Deep Deterministic Policy Gradient (DDPG) algorithm. To improve the agent's performance, we made the following adjustments:

We added parameter noise to the DDPG algorithm, which was introduced in the paper 'Better Exploration with Parameter Noise' by Plappert et al. (2017). This technique adds noise to the neural network weights during training to encourage exploration. We used a Gaussian noise with a mean of 0 and standard deviation of 0.1, which was gradually reduced to 0.01 after half of the training steps.
To deal with the issue of small rewards per environment step, we performed 5 environment steps for each action calculated by the network. However, we only used 1 environment step per action calculation during testing. This allowed the agent to learn faster during training.
To handle the large differences in the scale of observations, we normalized them so that their maximum absolute value is 1. This helped the agent learn more efficiently.
## Results
Our solution successfully trained an agent that can swim forward for over 1000 time steps, which is when the environment terminates. The agent's performance is shown in the following animation:

<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/MuJoCo/Swimmer/doc/screen.gif" width="400">
</p>
In conclusion, our approach using DDPG and the adjustments we made allowed us to successfully train an agent that can solve the Swimmer environment.
