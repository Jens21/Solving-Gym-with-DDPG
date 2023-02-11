import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Trainer():

    losses_actor = []
    losses_critic = []

    def __init__(self, network, replay_buffer, batch_size, gamma, tau):
        self.network = network
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.optimizer_actor = th.optim.AdamW(network.actor_policy.parameters(), lr=0.001, weight_decay=1e-2)
        self.optimizer_critic = th.optim.AdamW(network.critic_policy.parameters(), lr=0.0005, weight_decay=1e-2)

    def save_losses(self):
        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.losses_actor)), self.losses_actor)
        plt.savefig('actor losses.png')
        plt.clf()
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.losses_critic)), self.losses_critic)
        plt.savefig('critic losses.png')
        plt.clf()
        plt.close()

    def train_critic(self, states, actions, rewards, dones, next_states):
        self.optimizer_critic.zero_grad()
        qvalues = self.network.critic_policy(states, actions).flatten()
        with th.no_grad():
            actor_out = self.network.actor_target(next_states)
            qtargets = rewards + self.gamma * self.network.critic_target(next_states, actor_out).flatten() * (1 - dones)

        loss_critic = F.mse_loss(qvalues, qtargets)

        loss_critic.backward()
        self.optimizer_critic.step()

        return loss_critic.item()

    def train_actor(self, states, actions, rewards, dones, next_states):
        self.optimizer_actor.zero_grad()
        actor_out = self.network.actor_policy(states)
        critic_out = self.network.critic_policy(states, actor_out)
        loss_actor = -critic_out.mean()

        loss_actor.backward()
        self.optimizer_actor.step()

        return loss_actor.item()

    def soft_update(self):
        self.tau = 0.5
        for target_param, policy_param in zip(self.network.critic_target.parameters(), self.network.critic_policy.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        self.tau = 1
        for target_param, policy_param in zip(self.network.actor_target.parameters(), self.network.actor_policy.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train(self):
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size)

        loss_critic = self.train_critic(states, actions, rewards, dones, next_states)
        loss_actor = self.train_actor(states, actions, rewards, dones, next_states)

        self.losses_actor.append(loss_actor)
        self.losses_critic.append(loss_critic)

        self.soft_update()