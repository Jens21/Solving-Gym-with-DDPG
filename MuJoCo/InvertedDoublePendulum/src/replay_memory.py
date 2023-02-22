import torch as th

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = th.empty((self.buffer_size, 11+1+1+1+11))
        self.idx = 0
        self.current_buffer_size = 0

    def get_sample(self, batch_size):
        indices = th.randint(0, self.current_buffer_size, (batch_size,))
        samples = self.buffer[indices]
        obs = samples[:, :11]
        action = samples[:, 11]
        reward = samples[:, 12]
        done = samples[:, 13]
        next_obs = samples[:, 14:]

        return obs, action, reward, done, next_obs

    def push_sample(self, obs, action, reward, done, next_obs):
        sample = th.cat([obs, action, reward[None], done[None], next_obs])
        self.buffer[self.idx] = sample
        self.idx = (self.idx + 1) % self.buffer_size
        self.current_buffer_size = min(self.buffer_size, self.current_buffer_size + 1)

def main():
    memory = ReplayMemory(10)

    # Push some samples
    for i in range(10):
        obs = th.ones(24) * i
        action = th.tensor(0) * i
        reward = th.tensor(i)
        done = th.tensor(0)
        next_obs = th.ones(24) * (i + 1)
        memory.push_sample(obs, action, reward, done, next_obs)

    # Get a sample
    obs, action, reward, done, next_obs = memory.get_sample(1)
    print("Sample:")
    print(f"obs: {obs}")
    print(f"action: {action}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"next_obs: {next_obs}")

if __name__ == '__main__':
    main()
