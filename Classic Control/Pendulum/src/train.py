from pyvirtualdisplay import Display
import argparse
import numpy as np
import torch as th

np.random.seed(12345)
th.manual_seed(54321)

from network import Network
from replaybuffer import ReplayBuffer
from environment import Environment

BUFFER_SIZE = 100_000
GAMMA = 0.9
BATCH_SIZE = 64
N_TOTAL = 300 * 200
N_WARMUP = 1_000
TAU = 0.002
N_ENVS = 1

def add_samples_to_buffer(replay_buffer, obs, actions, rewards, truncated, next_obs):
    obs = th.from_numpy(np.array(obs)).float()
    actions = th.from_numpy(np.array(actions)).float()
    rewards = th.from_numpy(np.array(rewards)).float()
    truncated = th.from_numpy(np.array(truncated)).float()
    next_obs = th.from_numpy(np.array(next_obs)).float()

    for i in range(obs.shape[0]):
        replay_buffer.push([obs[i], actions[i], rewards[i], truncated[i], next_obs[i]])

def main():
    env = Environment(n_envs=N_ENVS)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    # replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
    network = Network(n_envs=N_ENVS)

    obs = env.reset()
    for itt in range(N_TOTAL):
        actions = network.get_action(obs)
        next_obs, rewards, truncated = env.step(actions)
        add_samples_to_buffer(replay_buffer, obs, actions, rewards, truncated, next_obs)
        obs = next_obs

        if itt >= N_WARMUP:
            network.train(replay_buffer, BATCH_SIZE, TAU, GAMMA)

        if itt % 10000 == 0 or itt == N_TOTAL-1:
            print('Iteration: {}'.format(itt))
            network.plot_losses()
            env.plot_rewards()
            network.save_networks()

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--display', action="store_true", help='a flag indicating whether training runs in a virtual environment')

    args = parser.parse_args()

    if args.display:
        display = Display(visible=0, size=(800, 600))
        display.start()
        print('Display started')

    main()

    if args.display:
        display.stop()