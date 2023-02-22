import gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory
from ddpg_network import Actor, Critic

GAMMA = 0.95
BATCH_SIZE = 64
N_TOTAL = 400_000
BUFFER_SIZE = 300_000
WARMUP = 1_000

def edit_observation(obs):
    obs = th.from_numpy(obs).float()
    ranges = th.Tensor([1.,  1.1,  2.9,  2.7,  1.5,  2.6,  2.7, 1.6,  5.4,  7.9, 10., 10., 10., 10., 10., 10., 10.])
    obs /= ranges

    return obs

def add_to_replay_buffer(obs, action, reward, done, next_obs, replay_mem):
    action, reward, done = th.tensor(action), th.tensor(reward), th.tensor(done)
    replay_mem.push_sample(obs, action, reward, done, next_obs)

def soft_update(critic_target, critic_policy, actor_target, actor_policy):
    tau = 0.1
    for target_param, policy_param in zip(critic_target.parameters(), critic_policy.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    tau = 1
    for target_param, policy_param in zip(actor_target.parameters(), actor_policy.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def train(replay_mem, optimizer_critic, optimizer_actor, actor_target, actor_policy, critic_policy, critic_target):
    obs, actions, rewards, dones, next_obs = replay_mem.get_sample(BATCH_SIZE)

    # Compute Q-values for current state and next state
    q_values = critic_policy(obs, actions).flatten()
    next_action = actor_target(next_obs)
    next_q_values = critic_target(next_obs, next_action).detach().flatten()

    targets = rewards + GAMMA * next_q_values * (1 - dones)

    # Compute loss and update network parameters
    loss_critic = th.nn.functional.mse_loss(q_values, targets)
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # train the actor
    action = actor_policy(obs)
    q_values = critic_policy(obs, action)
    loss_actor = -q_values.mean()
    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()

    soft_update(critic_target, critic_policy, actor_target, actor_policy)

    return loss_critic.item() + loss_actor.item()

def edit_reward(reward):
    reward /= 50
    reward = np.clip(reward, -1, 1)

    return reward

def make_plots(rewards, losses):
    plt.figure(figsize = (7, 7))
    plt.plot(np.arange(len(rewards)), rewards)
    plt.savefig('rewards.png')
    plt.clf()
    plt.close()

    plt.figure(figsize = (7, 7))
    plt.plot(np.arange(len(losses)), losses)
    plt.savefig('losses.png')
    plt.clf()
    plt.close()

def sample_action(obs, net_policy, itt):
    threshold = max(0.3, 1 - itt / 100_000)
    obs = obs[None]
    action = net_policy(obs)[0].detach().numpy()
    action += np.random.uniform(-threshold, threshold, action.shape)
    action = np.clip(-1, 1, action)

    return action, 0

def main():
    replay_mem = ReplayMemory(buffer_size=BUFFER_SIZE)

    actor_policy, actor_target = Actor(input_size=17, action_size=6), Actor(input_size=17, action_size=6)
    actor_target.load_state_dict(actor_policy.state_dict())
    critic_policy, critic_target = Critic(input_size=17, action_size=6), Critic(input_size=17, action_size=6)
    critic_target.load_state_dict(critic_policy.state_dict())
    optimizer_actor = th.optim.AdamW(actor_policy.parameters(), lr=8e-4, weight_decay=1e-2)
    optimizer_critic = th.optim.AdamW(critic_policy.parameters(), lr=4e-4, weight_decay=1e-2)

    env = gym.make('Walker2d-v4')
    obs, _ = env.reset()
    obs = edit_observation(obs)

    rewards = []
    reward_per_round = 0
    losses = []
    itt_since_reset = 0
    for itt in range(N_TOTAL):
        action, action_idx = sample_action(obs, actor_policy, itt)
        reward = 0
        for _ in range(2):
            next_obs, r, done, _, _ = env.step(action)
            reward += r
            if done:
                break
        reward_per_round += reward
        next_obs = edit_observation(next_obs)
        reward = edit_reward(reward)
        add_to_replay_buffer(obs, action, reward, done, next_obs, replay_mem)
        obs = next_obs
        itt_since_reset += 2

        if itt >= WARMUP:
            loss = train(replay_mem, optimizer_critic, optimizer_actor, actor_target, actor_policy, critic_policy, critic_target)
            losses.append(loss)

        if done or itt_since_reset >= 10_000:
            obs, _ = env.reset()
            obs = edit_observation(obs)
            print('Iteration: {}\tReward: {}'.format(itt, reward_per_round))
            rewards.append(reward_per_round)
            reward_per_round = 0
            itt_since_reset = 0

        if itt % 10_000 == 0:
            make_plots(rewards, losses)
            th.save(critic_policy.state_dict(), 'critic.pth')
            th.save(actor_policy.state_dict(), 'actor.pth')

    env.close()

if __name__ == '__main__':
    main()