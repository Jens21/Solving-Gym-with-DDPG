import numpy as np

class SineSampler():
    def __init__(self, steps_till_reset, dims):
        self.i = 0
        self.target = np.ones(dims)
        self.a = np.ones(dims)
        self.steps = steps_till_reset
        self.b = np.repeat(np.arcsin(0), dims)
        self.dims = dims

    def sample(self):
        y = self.a * np.sin(self.i / self.steps * np.pi / 2) + self.b
        self.i += 1

        if self.i == self.steps:
            self.b = self.target.copy()
            self.target = np.random.uniform(-1, 1, self.dims)
            self.a = (self.target - self.b)
            self.i = 0

        return y