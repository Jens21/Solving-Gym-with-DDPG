import torch as th
import torch.nn.functional as F

class Network():
    def __init__(self, observation_size, action_size):
        self.critic_policy = Critic(observation_size, action_size)
        self.critic_target = Critic(observation_size, action_size)
        self.critic_target.load_state_dict(self.critic_policy.state_dict())

        self.actor_policy = Actor(observation_size, action_size)
        self.actor_target = Actor(observation_size, action_size)
        self.actor_target.load_state_dict(self.actor_policy.state_dict())

    def save_actor(self):
        th.save(self.actor_policy.state_dict(), 'actor.pth')

class Actor(th.nn.Module):
    def __init__(self, observation_size, action_size):
        super(Actor, self).__init__()

        self.lin1 = th.nn.Linear(observation_size, 64)
        self.lin2 = th.nn.Linear(64, 64)
        self.lin3 = th.nn.Linear(64, action_size)

        for param in self.parameters():
            th.nn.init.uniform_(param.data, -1e-3, 1e-3)

    def forward(self, state):
        out1 = th.relu(self.lin1(state))
        out2 = F.relu(self.lin2(out1))
        # out3 = th.tanh(self.lin3(out1 + out2))
        actions = th.empty((state.shape[0],2))
        actions[:,0]=out2[:,:32].mean(dim=1)
        actions[:,1]=out2[:,32:].mean(dim=1)

        return actions, out2
        # return self.model(state)

class Critic(th.nn.Module):
    def __init__(self, observation_size, action_size):
        super(Critic, self).__init__()

        self.lin1 = th.nn.Linear(observation_size + action_size + 64, 64)
        self.lin2 = th.nn.Linear(64, 64)
        self.lin3 = th.nn.Linear(64, 1)

        for param in self.parameters():
            th.nn.init.uniform_(param.data, -1e-3, 1e-3)

    def forward(self, state, action, hidden_state):
        inp = th.concat([state, action, hidden_state], dim=1)
        out1 = F.leaky_relu(self.lin1(inp))
        out2 = F.leaky_relu(self.lin2(out1))
        out3 = self.lin3(out1 + out2)

        return out3
        # return self.model(inp)