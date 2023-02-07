import torch as th
import numpy as np

class Evaluater():
    def __init__(self, network, action_size):
        self.network = network
        self.action_size = action_size

    def get_network_action(self, state):
        state = th.from_numpy(state)[None]
        out = self.network.actor_policy(state)

        if state[0, 6] or state[0, 7]:
            out[:] = 0

        return out.cpu().detach().numpy()[0]

    def get_action(self, state, itt):
        # if itt < 50_000:
        #     std = min(0.1, 1 - itt/100_000)
        #     noise = np.random.normal(0, std, self.action_size)
        #     action = np.clip(self.get_network_action(state) + noise, -1, 1)
        threshold = min(0.1, 1 - itt/50_000)
        if np.random.uniform(0, 1, 1) < threshold:
            action = np.random.uniform(-1, 1, 2)
        else:
            action = self.get_network_action(state)

        return action