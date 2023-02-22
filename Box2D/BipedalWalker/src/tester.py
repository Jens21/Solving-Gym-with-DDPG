import gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory
from ddpg_network import Actor, Critic

GAMMA = 0.95
BATCH_SIZE = 32
N_TOTAL = 1_000_000
BUFFER_SIZE = 300_000
WARMUP = 1_000

def edit_observation(obs):
    obs = th.from_numpy(obs)
    ranges = th.Tensor([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    obs /= ranges

    return obs

def main():
    actor = Actor(input_size=24, action_size=4)
    actor.load_state_dict(th.load('actor.pth'))

    env = gym.make('BipedalWalker-v3', render_mode='human')
    obs, _ = env.reset()
    obs = edit_observation(obs)

    reward_per_round = 0
    itt_since_reset = 0
    for itt in range(N_TOTAL):
        # action, action_idx = sample_action(obs, actor_policy, itt)
        action = actor(obs[None])[0].detach().numpy()
        reward = 0
        for _ in range(5):
            obs, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break
        reward_per_round += reward
        obs = edit_observation(obs)
        itt_since_reset += 5

        if done:
            obs, _ = env.reset()
            obs = edit_observation(obs)
            print('Iteration: {}\tIterations since last reset: {}\tReward: {}'.format(itt, itt_since_reset, reward_per_round))
            reward_per_round = 0
            itt_since_reset = 0

    env.close()

if __name__ == '__main__':
    main()