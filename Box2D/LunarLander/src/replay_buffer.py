import torch as th

class ReplayBuffer:

    idx = 0
    current_mem_size = 0

    def __init__(self, buffer_size, observation_size, action_size):
        self.buffer_size = buffer_size

        self.states = th.empty((buffer_size, observation_size))
        self.actions = th.empty((buffer_size, action_size))
        self.rewards = th.empty(buffer_size)
        self.dones = th.empty(buffer_size)
        self.next_states = th.empty((buffer_size, observation_size))
        self.hidden_states = th.empty((buffer_size, 64))

    def push(self, state, action, reward, done, next_state, hidden_state):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.next_states[self.idx] = next_state
        self.hidden_states[self.idx] = hidden_state

        self.idx = (self.idx + 1) % self.buffer_size

        self.current_mem_size = min(self.buffer_size, self.current_mem_size + 1)

    def sample(self, batch_size):
        indices = th.randint(0, self.current_mem_size, (batch_size,))

        return self.states[indices], self.actions[indices], self.rewards[indices], self.dones[indices], self.next_states[indices], self.hidden_states[indices]