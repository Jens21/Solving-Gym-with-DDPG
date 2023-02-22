import gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory
from ddpg_network import Actor, Critic

GAMMA = 0.9
BATCH_SIZE = 32
N_TOTAL = 200_000
BUFFER_SIZE = 300_000
WARMUP = 1_000

def edit_observation(obs):
    obs = th.from_numpy(obs).float()
    ranges = th.Tensor([1., 1., 1., 1., 0.2, 0.2, 60., 38., 0.4, 0.4, 1.])
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
    # reward /= 2
    # reward = np.clip(reward, -2, 10)

    return reward

def make_plots(rewards, losses):
    plt.figure(figsize = (7, 7))
    plt.plot(np.arange(len(rewards)), rewards)
    plt.hlines(xmin=0, xmax=len(rewards), y=0, colors='black')
    plt.savefig('rewards.png')
    plt.clf()
    plt.close()

    plt.figure(figsize = (7, 7))
    plt.plot(np.arange(len(losses)), losses)
    plt.savefig('losses.png')
    plt.clf()
    plt.close()

def sample_action(obs, net_policy, itt):
    threshold = max(0.1, 1 - itt / 100_000)
    obs = obs[None]
    action = net_policy(obs)[0].detach().numpy()
    action += np.random.uniform(-threshold, threshold, action.shape)
    action = np.clip(-1, 1, action)

    return action, 0

def main():
    replay_mem = ReplayMemory(buffer_size=BUFFER_SIZE)

    actor_policy, actor_target = Actor(input_size=11, action_size=2), Actor(input_size=11, action_size=2)
    actor_target.load_state_dict(actor_policy.state_dict())
    critic_policy, critic_target = Critic(input_size=11, action_size=2), Critic(input_size=11, action_size=2)
    critic_target.load_state_dict(critic_policy.state_dict())
    optimizer_actor = th.optim.AdamW(actor_policy.parameters(), lr=2e-3, weight_decay=1e-2)
    optimizer_critic = th.optim.AdamW(critic_policy.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler_actor = th.optim.lr_scheduler.CosineAnnealingLR(optimizer_actor, T_max=N_TOTAL - WARMUP)
    scheduler_critic = th.optim.lr_scheduler.CosineAnnealingLR(optimizer_critic, T_max=N_TOTAL - WARMUP)

    env = gym.make('Reacher-v4')
    obs, _ = env.reset()
    obs = edit_observation(obs)

    rewards = []
    reward_per_round = 0
    losses = []
    itt_since_reset = 0
    for itt in range(N_TOTAL):
        action, action_idx = sample_action(obs, actor_policy, itt)
        reward = 0
        for _ in range(3):
            next_obs, r, done, _, _ = env.step(action * 0.15)
            reward += r
            if done:
                break
        reward_per_round += reward
        next_obs = edit_observation(next_obs)
        reward = edit_reward(reward)
        add_to_replay_buffer(obs, action, reward, done, next_obs, replay_mem)
        obs = next_obs
        itt_since_reset += 3

        if itt >= WARMUP:
            loss = train(replay_mem, optimizer_critic, optimizer_actor, actor_target, actor_policy, critic_policy, critic_target)
            losses.append(loss)

            scheduler_actor.step()
            scheduler_critic.step()

        if done or itt_since_reset >= 150:
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