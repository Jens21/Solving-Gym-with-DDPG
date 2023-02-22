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
    ranges = th.Tensor([1., 1., 1., 1., 0.2, 0.2, 60., 38., 0.4, 0.4, 1.])
    obs /= ranges

    return obs

def sample_action(obs, net_policy, itt):
    threshold = max(0.1, 1 - itt / 50_000)
    obs = obs[None]
    action = net_policy(obs)[0].detach().numpy()
    action += np.random.uniform(-threshold, threshold, action.shape)
    action = np.clip(-1, 1, action)

    return action, 0

def main():
    actor = Actor(input_size=11, action_size=2)
    actor.load_state_dict(th.load('actor.pth'))

    env = gym.make('Reacher-v4', render_mode='human')
    obs, _ = env.reset()
    obs = edit_observation(obs)

    reward_per_round = 0
    itt_since_reset = 0
    for itt in range(N_TOTAL):
        # action, action_idx = sample_action(obs, actor, 100_000)
        action = actor(obs[None])[0].detach().numpy()
        reward = 0
        for _ in range(1):
            obs, r, done, _, _ = env.step(action * 0.15)
            reward += r
            if done:
                break
        print(reward)
        reward_per_round += reward
        obs = edit_observation(obs)
        itt_since_reset += 1

        if done or itt_since_reset>150:
            obs, _ = env.reset()
            obs = edit_observation(obs)
            print('Iteration: {}\tIterations since last reset: {}\tReward: {}'.format(itt, itt_since_reset, reward_per_round))
            reward_per_round = 0
            itt_since_reset = 0

    env.close()

if __name__ == '__main__':
    main()