import torch as th

class Critic(th.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = th.nn.Sequential(
            th.nn.Linear(3 + 1, 32),
            th.nn.BatchNorm1d(32),
            th.nn.ReLU(),
            # th.nn.Linear(32, 32),
            # th.nn.BatchNorm1d(32),
            # th.nn.ReLU(),

            th.nn.Linear(32, 1)
        )

    def forward(self, x, action):
        inp = th.concat([x.flatten(1), action], dim=1)
        out = self.model(inp)

        return out