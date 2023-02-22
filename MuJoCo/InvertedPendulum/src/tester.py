import gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory
from ddpg_network import Actor, Critic

GAMMA = 0.95
BATCH_SIZE = 64
N_TOTAL = 1_000_000
BUFFER_SIZE = 300_000
WARMUP = 1_000

def edit_observation(obs):
    obs = th.from_numpy(obs).float()
    ranges = th.Tensor([1., 0.4, 3.8, 8.])
    obs /= ranges

    return obs

def main():
    actor = Actor(input_size=4, action_size=1)
    actor.load_state_dict(th.load('actor.pth'))

    env = gym.make('InvertedPendulum-v4', render_mode='human')
    obs, _ = env.reset()
    obs = edit_observation(obs)

    reward_per_round = 0
    itt_since_reset = 0
    for itt in range(N_TOTAL):
        # action, action_idx = sample_action(obs, actor_policy, itt)
        action = actor(obs[None])[0].detach().numpy()
        reward = 0
        for _ in range(2):
            obs, r, done, _, _ = env.step(action * 3)
            reward += r
            if done:
                break
        reward_per_round += reward
        obs = edit_observation(obs)
        itt_since_reset += 2

        if done or itt_since_reset>150:
            obs, _ = env.reset()
            obs = edit_observation(obs)
            print('Iteration: {}\tIterations since last reset: {}\tReward: {}'.format(itt, itt_since_reset, reward_per_round))
            reward_per_round = 0
            itt_since_reset = 0

    env.close()

if __name__ == '__main__':
    main()