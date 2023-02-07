import gym
import matplotlib.pyplot as plt
import numpy as np

class Environment:
    rewards = []

    def __init__(self, render_mode=None, n_envs=1):
        self.n_envs = n_envs
        self.env = gym.vector.make('Pendulum-v1', render_mode=render_mode, num_envs=n_envs)

    def reset(self):
        obs, _ = self.env.reset()
        obs = self.edit_observations(obs)
        self.reward_per_round = np.zeros(self.n_envs)

        return obs

    def edit_observations(self, obs):
        obs[:, 2] /= 8

        return obs

    def edit_reward(self, reward):
        # max absolute reward is 16.2736044
        # every episode is truncated to 200 steps
        reward *= 255 / (16.2736044 * 200)

        return reward

    def step(self, actions):
        actions *= 2
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        obs = self.edit_observations(obs)
        self.reward_per_round += reward
        reward = self.edit_reward(reward)

        if terminated.any() or truncated.any():
            self.rewards.append(np.mean(self.reward_per_round))
            self.reset()

        return obs, reward, truncated

    def close(self):
        self.env.close()

    def plot_rewards(self):
        fig = plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.rewards)), self.rewards)
        plt.savefig('rewards.png')
        plt.clf()
        plt.close(fig)