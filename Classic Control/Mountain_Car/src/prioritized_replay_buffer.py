import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=1, beta=1):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def push(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)
        self.priorities[len(self.buffer) - 1] = max_prio

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None

        priorities = self.priorities ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs[:len(self.buffer)])
        experiences = [self.buffer[idx] for idx in indices]
        sample_probs = probs[indices]
        importance_weights = (len(self.buffer) * sample_probs) ** -self.beta
        return experiences, indices, importance_weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


if __name__ == '__main__':
    prio = PrioritizedReplayBuffer(max_size=10)
    for i in range(5):
        prio.push(i)

    experiences, indices, probabilities = prio.sample(16)
    prio.update_priorities([0,1,2,3,4,5,6,7], [0.1,0.1,0,0,0,0,0,0])
    experiences, indices, probabilities = prio.sample(32)
    print(experiences)
    bins = np.bincount(experiences)
    print(bins)