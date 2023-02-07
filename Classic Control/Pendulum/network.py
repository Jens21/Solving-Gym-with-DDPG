import torch as th
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from actor import Actor
from critic import Critic
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class Network():
    actor_losses = []
    critic_losses = []

    def __init__(self, load=False, n_envs=1):
        self.n_envs = n_envs
        # self.sampler = OrnsteinUhlenbeckProcess(n_envs=n_envs, size=1, mu=0, theta=0.1, sigma=0.1)

        self.actor_target = Actor().to(device)
        self.actor_policy = Actor().to(device)
        self.actor_target.eval()

        self.critic_target = Critic().to(device)
        self.critic_policy = Critic().to(device)
        self.critic_target.eval()

        if load:
            self.load_dicts()

        self.actor_optimizer = th.optim.AdamW(self.actor_policy.parameters(), lr=1e-4, weight_decay=1e-2)
        self.critic_optimizer = th.optim.AdamW(self.critic_policy.parameters(), lr=1e-3, weight_decay=1e-2)

    def load_dicts(self):
        actor_dict = th.load('actor.pth', map_location='cpu')
        critic_dict = th.load('critic.pth', map_location='cpu')

        self.actor_target.load_state_dict(actor_dict)
        self.actor_policy.load_state_dict(actor_dict)

        self.critic_target.load_state_dict(critic_dict)
        self.critic_policy.load_state_dict(critic_dict)

    def get_network_action(self, obs):
        self.actor_policy.eval()
        with th.no_grad():
            inp = th.from_numpy(obs).float().to(device)
            out = self.actor_policy(inp)
            out = out.cpu().detach().numpy()

        return out

    def get_random_action(self):
        return np.random.uniform(-1, 1, 1)

    def get_action(self, obs):
        action = self.get_network_action(obs)

        # action += self.sampler.sample()
        action += np.random.normal(0, 0.3, (self.n_envs, 1))
        action = np.clip(action, -1, 1)

        return action

    def soft_update(self, tau):
        for target_param, policy_param in zip(self.critic_target.parameters(), self.critic_policy.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

        for target_param, policy_param in zip(self.actor_target.parameters(), self.actor_policy.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def train(self, replay_buffer, batch_size, tau, gamma):
        samples = replay_buffer.sample(batch_size)
        # samples, indices, _ = replay_buffer.sample(batch_size)

        obs = th.stack([samp[0] for samp in samples], dim=0).to(device)
        actions = th.stack([samp[1] for samp in samples], dim=0).to(device)
        rewards = th.hstack([samp[2] for samp in samples]).to(device)
        truncated = th.hstack([samp[3] for samp in samples]).to(device)
        next_obs = th.stack([samp[4] for samp in samples], dim=0).to(device)

        # train the critic
        self.critic_optimizer.zero_grad()
        q_values = self.critic_policy(obs, actions).flatten()
        with th.no_grad():
            actor_out = self.actor_target(next_obs)
            next_q_values = self.critic_target(next_obs, actor_out).flatten()
            trg = rewards + gamma * next_q_values * (1 - truncated)
        # priorities = F.mse_loss(q_values, trg, reduction='none')
        # replay_buffer.update_priorities(indices=indices, priorities=priorities)
        # loss_critic = priorities.mean()
        loss_critic = F.mse_loss(q_values, trg)

        loss_critic.backward()
        self.critic_optimizer.step()

        # train the actor
        self.critic_policy.eval()
        self.actor_optimizer.zero_grad()
        actor_out = self.actor_policy(obs)
        critic_out = self.critic_policy(obs, actor_out).flatten()
        loss_actor = -critic_out.mean()

        loss_actor.backward()
        self.actor_optimizer.step()

        self.soft_update(tau)

        self.actor_losses.append(loss_actor.item())
        self.critic_losses.append(loss_critic.item())

    def plot_losses(self):
        fig = plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.actor_losses)), self.actor_losses)
        plt.savefig('actor losses.png')
        plt.clf()
        plt.close(fig)

        fig = plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.critic_losses)), self.critic_losses)
        plt.savefig('critic losses.png')
        plt.clf()
        plt.close(fig)

    def save_networks(self):
        th.save(self.actor_policy.state_dict(), 'actor.pth')
        th.save(self.critic_policy.state_dict(), 'critic.pth')