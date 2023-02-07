# The Pendulum Environment
## Overview
The pendulum environment is a classic control problem from the OpenAI gym library. It involves controlling a pendulum to maintain its balance without falling over by applying actions to it. The goal is to maintain the balance of the pendulum for as long as possible.

## Our Solution
Our solution uses a normal distributed noise added to the network output during training. The standard deviation of the noise is set to 0.3.

<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/Classic%20Control/Pendulum/doc/screen.gif" width="400">
</p>
