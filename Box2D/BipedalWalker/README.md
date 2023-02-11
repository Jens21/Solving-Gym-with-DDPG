# The Bipedal Walker Environment
## Overview
This repository contains a solution for the BipedalWalker-v3 environment from the OpenAI Gym library. The goal of the environment is to make a two-legged robot walk as far as possible without falling.

## Solution
We used the Deep Deterministic Policy Gradient (DDPG) algorithm to train an agent to solve the BipedalWalker-v3 environment. During training, we added normally distributed noise to the network's actions, and linearly decreased the variance of the noise from 1 to 0.1.

To address some of the challenges in the environment, we made the following adjustments:

To deal with the issue of small rewards per environment step, we performed 5 environment steps for each action calculated by the network.
To handle the large differences in the scale of observations, we normalized them so that their maximum absolute value is 1.
To address the issue of large rewards, we divided the rewards by 20.
## Results
Our solution was able to achieve a score of [insert score here] on the BipedalWalker-v3 environment.

Here's an animated gif showing the agent walking in the environment:

<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/Box2D/BipedalWalker/doc/screen.gif" width="400">
</p>
