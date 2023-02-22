# The Inverted Double Pendulum
## Overview
The Inverted Double Pendulum is a classic problem in control theory and robotics that involves balancing a two-link pendulum in an inverted position. The task is challenging due to its high-dimensional observation and action spaces, nonlinear dynamics, and unstable equilibrium point. The goal of this environment is to develop an agent that can hold up the pendulum for as long as possible.

## Solution
We used the Deep Deterministic Policy Gradient (DDPG) algorithm to train an agent to solve the Inverted Double Pendulum environment. During training, we added normally distributed noise to the network's actions, and linearly decreased the variance of the noise from 1 to 0.1.

To address some of the challenges in the environment, we made the following adjustments:

To deal with the issue of small rewards per environment step, we performed 2 environment steps for each action calculated by the network.
To handle the large differences in the scale of observations, we normalized them so that their maximum absolute value is 1.
To address the issue of large rewards, we divided the rewards by 200.
Results
Our solution successfully holds up the pendulum for over 1000 time steps, which is when the environment terminates. The agent in the environment can be seen in the following animation:

<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/MuJoCo/InvertedDoublePendulum/doc/screen.gif" width="400">
</p>
