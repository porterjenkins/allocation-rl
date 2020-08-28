import pickle
import numpy as np

from evaluators.queue import Queue

class Policy(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, state):

        p = np.ones(self.n_actions) / self.n_actions
        return np.random.dirichlet(p, 1)




class PSRS(object):

    """Per-state Rejection Sampling evaluator"""

    def __init__(self, buffer_path, policy, env_policy):

        self.buffer_path = buffer_path
        self.policy = policy
        self.env_policy = env_policy

        self.buffer = self.get_buffer(self.buffer_path)
        self.queue = self.build_queue(self.buffer)



    def get_buffer(self, fpath):

        with open(fpath, 'rb') as f:
            buffer = pickle.load(f)

        return buffer


    def build_queue(self, buffer):

        queue = Queue()


        for i in range(len(buffer.storage)):
            s, a, r, s_prime, _ = buffer.storage[i]

            if not queue.is_in(s):
                queue.add(s)

            queue[s].append((s, a, r, s_prime))

        queue.permute()

        return queue








if __name__ == "__main__":
    policy = Policy(n_actions=10)
    env_policy = Policy(n_actions=10)

    psrs = PSRS("../data/random-buffer.p", policy, env_policy)