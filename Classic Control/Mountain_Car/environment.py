import gym
import matplotlib.pyplot as plt
import numpy as np

class Environment:
    rewards = []

    def __init__(self, render_mode=None, n_envs=1):
        self.n_envs = n_envs
        self.env = gym.vector.make('MountainCarContinuous-v0', render_mode=render_mode, num_envs=n_envs)

    def reset(self):
        obs, _ = self.env.reset()
        obs = self.edit_observations(obs)
        self.reward_per_round = np.zeros(self.n_envs)
        self.largest_x = -np.ones(self.n_envs)

        return obs

    def edit_observations(self, obs):
        obs[:, 0] = (obs[:, 0] + 0.3) / 0.9
        obs[:, 1] = obs[:, 1] / 0.07

        return obs

    def edit_reward(self, reward, obs):
        # max absolute reward is 16.2736044
        # every episode is truncated to 200 steps
        # larger = obs[:, 0] > self.largest_x
        # reward = larger * (obs[:, 0] - self.largest_x)
        # self.largest_x = np.maximum(self.largest_x, obs[:, 0])

        reward /= 10

        return reward

    def edit_actions(self, actions):
        return actions

    def step(self, actions):
        actions = self.edit_actions(actions)
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        obs = self.edit_observations(obs)
        self.reward_per_round += reward
        reward = self.edit_reward(reward, obs)

        if terminated.any() or truncated.any():
            self.rewards.append(np.mean(self.reward_per_round))
            self.reset()

        return obs, reward, (terminated | truncated)

    def close(self):
        self.env.close()

    def plot_rewards(self):
        fig = plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.rewards)), self.rewards)
        plt.savefig('rewards.png')
        plt.clf()
        plt.close(fig)

# if __name__ == '__main__':
#     env = Environment(render_mode='human', n_envs=3)
#     env.reset()
#
#     action = np.random.normal(0, 1, (3,1))
#     for i in range(100):
#         env.step(action)
#
#     env.close()