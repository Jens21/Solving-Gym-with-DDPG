import torch as th
import numpy as np

th.manual_seed(12345)
np.random.seed(54329)

import gym

from network import Network
from evaluater import Evaluater

divider = np.array([1.5, 1.5, 5., 5., 3.14, 5., 1., 1.])
def scale_observation(obs):
    obs /= divider
    return obs

env = gym.make('LunarLander-v2', continuous=True, render_mode='human')
seed = int(np.random.randint(0,1e8,1))
obs, _ = env.reset(seed=seed)
obs = scale_observation(obs)

network = Network(observation_size=8, action_size=2)
network.actor_policy.load_state_dict(th.load('actor.pth'))
evaluater = Evaluater(network, action_size=2)

current_episode_reward = 0
steps_since_last_reset = 0

if __name__ == '__main__':
    rounds = 0
    while rounds < 11:
        action = evaluater.get_network_action(obs)
        reward = 0
        for i in range(5):
            obs, r, done, _, _ = env.step(action)
            obs = scale_observation(obs)
            reward += r
            if done:
                break
        current_episode_reward += reward

        steps_since_last_reset += 5

        if done or steps_since_last_reset > 2_000:
            seed = int(np.random.randint(0, 1e8, 1))
            env.reset(seed = seed)
            print('Reward: {}\tTimesteps: {}'.format(current_episode_reward, steps_since_last_reset))
            current_episode_reward = 0
            steps_since_last_reset = 0
            rounds += 1

env.close()