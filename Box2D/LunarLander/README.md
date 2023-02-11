# The Lunar Lander Environment
## Overview
The lunar lander environment is a classic control problem from the OpenAI gym library. The goal is to control a lunar lander to land safely on the moon by applying actions to its engines.

## Our Solution
We used a Deep Deterministic Policy Gradient (DDPG) algorithm with normal distributed noise added to the network's actions during training. The variance of the noise was decreased linearly from 1 to 0.1 over the course of the training.

To overcome several challenges in the environment, we made the following adjustments to our solution:

To avoid problems with the added noise during landing, we shut off all engines when at least one leg of the lander reported contact with the moon.
To address the issue of small rewards per environment step, we calculated the action using the network and performed 5 environment steps on the calculated action.
To handle the large differences in the scale of observations, we scaled them so that their maximum absolute value is 1.
To address the issue of large rewards, we divided the rewards by 250.
<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/Box2D/LunarLander/doc/screen.gif" width="400">
</p>
