import torch as th
import numpy as np

th.manual_seed(12345)
np.random.seed(54321)

import gym
import matplotlib.pyplot as plt

from network import Network
from replay_buffer import ReplayBuffer
from trainer import Trainer
from evaluater import Evaluater

N_TOTAL = 45_000
BATCH_SIZE = 64
GAMMA = 0.9
# GAMMA = 0.8
TAU = 0.01

divider = np.array([1.5, 1.5, 5., 5., 3.14, 5., 1., 1.])
def scale_observation(obs):
    obs /= divider
    return obs

env = gym.make('LunarLander-v2', continuous=True)
seed = int(np.random.randint(0, 1e8, 1))
obs, _ = env.reset(seed=seed)
obs = scale_observation(obs)
replay_buffer = ReplayBuffer(buffer_size=100_000, observation_size=8, action_size=2)

network = Network(observation_size=8, action_size=2)

trainer = Trainer(network, replay_buffer, BATCH_SIZE, GAMMA, TAU)
evaluater = Evaluater(network, action_size=2)

rewards = []
current_episode_reward = 0
steps_since_last_reset = 0

def add_sample_to_replay_buffer(obs, action, reward, done, next_obs):
    state = th.from_numpy(obs).clone()
    action = th.from_numpy(action).clone()
    reward = th.FloatTensor([reward]).clone()
    done = th.FloatTensor([done]).clone()
    next_state = th.from_numpy(next_obs).clone()

    replay_buffer.push(state, action, reward, done, next_state)

def plot_rewards():
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(rewards)), rewards)
    plt.hlines(y=0, xmin=0, xmax=len(rewards), colors='black')
    plt.ylim(-800, 400)
    plt.savefig('rewards.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    for itt in range(N_TOTAL):
        action = evaluater.get_network_action(obs)
        reward = 0
        for i in range(5):
            next_obs, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break
        next_obs = scale_observation(next_obs)
        current_episode_reward += reward
        # reward = np.clip(reward, -10, 10)
        reward = reward/250
        add_sample_to_replay_buffer(obs, action, reward, done, next_obs)
        obs = next_obs

        steps_since_last_reset += 5

        trainer.train()

        if done or steps_since_last_reset > 1_000:
            seed = int(np.random.randint(0, 1e8, 1))
            obs, _ = env.reset(seed = seed)

            obs = scale_observation(obs)
            rewards.append(current_episode_reward)
            current_episode_reward = 0
            steps_since_last_reset = 0

        if itt % 1_000 == 0 and itt > 0:
            print('Iteration: {}'.format(itt))
            plot_rewards()
            trainer.save_losses()
            network.save_actor()

env.close()