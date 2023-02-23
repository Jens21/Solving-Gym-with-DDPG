import torch as th

from noisy_linear import NoisyLinear

class Actor(th.nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()

        self.model = th.nn.Sequential(
            NoisyLinear(input_size, 64),
            th.nn.ReLU(),
            # th.nn.Linear(64, 64),
            # th.nn.ReLU(),
            NoisyLinear(64, action_size),
            th.nn.Tanh()
        )
    def forward(self, obs):
        return self.model(obs)

    def sample_noise(self):
        self.model[0].sample_noise()
        self.model[2].sample_noise()

    def set_std(self, std):
        self.model[0].std = std
        self.model[2].std = std

class Critic(th.nn.Module):
    def __init__(self, input_size, action_size):
        super(Critic, self).__init__()

        self.model = th.nn.Sequential(
            th.nn.Linear(input_size + action_size, 64),
            th.nn.LeakyReLU(),
            # th.nn.Linear(64, 64),
            # th.nn.LeakyReLU(),
            th.nn.Linear(64, 1),
        )
    def forward(self, obs, action):
        inp = th.concat([obs, action], dim=1)
        return self.model(inp)
