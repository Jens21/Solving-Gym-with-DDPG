import torch as th

class Actor(th.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.model = th.nn.Sequential(
            th.nn.Flatten(1),
            th.nn.Linear(3, 32),
            th.nn.BatchNorm1d(32),
            th.nn.ReLU(),
            # th.nn.Linear(32, 32),
            # th.nn.BatchNorm1d(32),
            # th.nn.ReLU(),

            th.nn.Linear(32, 1),
            th.nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)

        return out