import gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory
from ddpg_network import Actor, Critic
from sine_sampler import SineSampler

GAMMA = 0.95
BATCH_SIZE = 64
N_TOTAL = 1_000_000
BUFFER_SIZE = 300_000
WARMUP = 1_000

def edit_observation(obs):
    obs = th.from_numpy(obs).float()
    ranges = th.Tensor([2., 2., 2., 2., 3., 3., 6., 6.])
    obs /= ranges

    return obs

def sample_action(obs, net_policy, itt, sine_sampler):
    # threshold = max(0.1, 1 - itt / 100_000)
    obs = obs[None]
    action = net_policy(obs)[0].detach().numpy()
    # action += np.random.uniform(-threshold, threshold, action.shape)
    # print(action)
    # action = 0.6 * sine_sampler.sample()
    # action = np.clip(action, -1, 1)

    return action, 0

def main():
    sine_sampler = SineSampler(steps_till_reset=5, dims=2)
    actor = Actor(input_size=8, action_size=2)
    actor.load_state_dict(th.load('actor.pth'))
    actor.eval()

    env = gym.make('Swimmer-v4', render_mode='human')
    obs, _ = env.reset()
    obs = edit_observation(obs)

    reward_per_round = 0
    itt_since_reset = 0
    for itt in range(N_TOTAL):
        action, action_idx = sample_action(obs, actor, 100_000, sine_sampler)
        # action = actor(obs[None])[0].detach().numpy()
        reward = 0
        for _ in range(5):
            obs, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break
        reward_per_round += reward
        obs = edit_observation(obs)
        itt_since_reset += 5
        actor.sample_noise()

        if done or itt_since_reset > 500:
            obs, _ = env.reset()
            obs = edit_observation(obs)
            print('Iteration: {}\tIterations since last reset: {}\tReward: {}'.format(itt, itt_since_reset, reward_per_round))
            reward_per_round = 0
            itt_since_reset = 0

    env.close()

if __name__ == '__main__':
    main()