# The Reacher
## Overview
The Reacher environment is a part of the OpenAI Gym toolkit, which provides a variety of environments for reinforcement learning (RL) research. The Reacher environment involves a double-jointed arm that must be controlled to move its hand to a particular target location. The goal is to train an agent to move the arm to the target location and maintain it there for as long as possible.

## Solution
In our solution, we used the Deep Deterministic Policy Gradient (DDPG) algorithm to train the agent to solve the Inverted Pendulum environment. To improve the agent's performance, we made the following adjustments:

We added normally distributed noise to the network's actions during training and linearly decreased the variance of the noise from 1 to 0.1. This helped the agent explore the state space more effectively.
To deal with the issue of small rewards per environment step, we performed 5 environment steps for each action calculated by the network. However, we only used 1 environment step per action calculation during testing. This allowed the agent to learn faster during training.
To handle the large differences in the scale of observations, we normalized them so that their maximum absolute value is 1. This helped the agent learn more efficiently.
## Results
Our solution successfully trained an agent that can hold up the pendulum for over 1000 time steps, which is when the environment terminates. The agent's performance is shown in the following animation:

<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/MuJoCo/Reacher/doc/screen.gif" width="400">
</p>
In conclusion, our approach using DDPG and the adjustments we made allowed us to successfully train an agent that can solve the Reacher environment.
