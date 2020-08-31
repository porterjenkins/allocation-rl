import numpy as np

class EvalQueue(object):

    def __init__(self):
        self.storage = {}


    def __getitem__(self, state):
        s_string = self.stringify(state)
        return self.storage[s_string]


    def stringify(self, arr):
        return ",".join(arr.astype(str))

    def add(self, state):
        s_string = self.stringify(state)

        self.storage[s_string] = []

    def is_in(self, state):
        s_string = self.stringify(state)
        if s_string in self.storage:
            return True
        else:
            return False


    def permute(self):
        for k, v in self.storage.items():

            n_items = len(v)
            ran_idx = np.random.permutation(range(n_items))
            randomized = []

            for i in range(n_items):
                idx = ran_idx[i]
                randomized.append(v[idx])

            self.storage[k] = randomized




