# The Continuous Mountain Car Environment
## Overview

The continuous mountain car environment is a classic control problem from the OpenAI gym library. It involves a car that needs to drive up a mountain by applying actions to its engine. The goal is to reach the flag at the top of the mountain within a limited number of steps.

# Our Solution
Our solution uses an algorithm with 5 superimposed sine waves as random noise. Compared to random gaussian noise, sine waves were found to be more effective in driving the car up the mountain. The critic is trained to understand the effects of different actions in different states, even if the sine waves don't succeed in driving the car up the mountain.

<p align="center">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/Classic%20Control/Mountain_Car/doc/screen.gif" width="400">
  <img src="https://github.com/Jens21/Solving-Gym-with-DDPG/blob/main/Classic%20Control/Mountain_Car/doc/random%20sine%20samples.png">
</p>
