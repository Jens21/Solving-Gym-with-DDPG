import numpy as np

class OrnsteinUhlenbeckProcess:
    def __init__(self, n_envs, size, mu=0, theta=0.15, sigma=0.2):
        self.n_envs = n_envs
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones((n_envs, self.size)) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.n_envs, self.size)
        self.x += dx
        return self.x

if __name__ == '__main__':
    sampler = OrnsteinUhlenbeckProcess(2)
    print(sampler.sample())
