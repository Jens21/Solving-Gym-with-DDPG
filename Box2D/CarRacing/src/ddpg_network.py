import torch as th

class Actor(th.nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()

        self.model = th.nn.Sequential(
            th.nn.Linear(input_size, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, action_size),
            th.nn.Tanh()
        )
    def forward(self, obs):
        return self.model(obs)

class Critic(th.nn.Module):
    def __init__(self, input_size, action_size):
        super(Critic, self).__init__()

        self.model = th.nn.Sequential(
            th.nn.Linear(input_size + action_size, 64),
            th.nn.LeakyReLU(),
            th.nn.Linear(64, 64),
            th.nn.LeakyReLU(),
            th.nn.Linear(64, 1),
        )
    def forward(self, obs, action):
        inp = th.concat([obs, action], dim=1)
        return self.model(inp)
