import numpy as np

class SineSampler():
    def __init__(self, n_envs=1, n_length=1, time_steps=200, n_waves=5):
        self.n_envs = n_envs
        self.n_length = n_length
        self.step = 0
        self.n_waves = n_waves
        self.time_steps = time_steps

    def generate_parameters(self):
        p = 1 / np.logspace(0.1, 100, self.time_steps)[:, np.newaxis, np.newaxis]
        p = np.tile(p, (1, self.n_envs, self.n_length))
        p += np.random.normal(0, 1, p.shape)
        a = np.random.uniform(0.1, 1, p.shape)
        b = np.random.uniform(0.1, 10, p.shape)

        x = np.linspace(0, 30, self.time_steps)[:, np.newaxis, np.newaxis]
        x = np.tile(x, (1, self.n_envs, self.n_length))
        self.noise = np.zeros_like(p)

        for i in range(self.n_waves):
            self.noise += a[i] * np.sin(p[i] * x + b[i])

        self.step = 0

    def sample(self):
        if self.step % self.time_steps == 0:
            self.generate_parameters()

        res = self.noise[self.step]
        self.step += 1

        return res