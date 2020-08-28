import pickle
import numpy as np
import random

from evaluators.queue import Queue

class Policy(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, state):

        p = np.ones(self.n_actions) / self.n_actions
        return p




class PSRS(object):

    """Per-state Rejection Sampling evaluator"""

    def __init__(self, buffer_path, policy, env_policy, n_actions, n_episodes):

        self.buffer_path = buffer_path
        self.policy = policy
        self.env_policy = env_policy
        self.n_actions = n_actions
        self.n_episodes = n_episodes

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

    def get_m(self, state):

        prob_policy = self.policy.predict(state)
        prob_env = self.env_policy.predict(state)

        M = -1

        for a in range(self.n_actions):

            M_prime = prob_policy[a] /  prob_env[a]

            if M_prime > M:
                M = M_prime

        return np.exp(M)

    def evaluate(self):

        rewards = []
        for i in range(self.n_episodes):

            r_i = 0
            state, _, _, _, _ = self.buffer.sample(batch_size=1)
            state = state[0]

            while True:

                M = self.get_m(state)

                try:
                    _, a, r, s_prime = self.queue[state].pop()
                except IndexError:
                    break

                alpha = random.random()

                prob_policy = self.policy.predict(state)[a]
                prob_env = self.env_policy.predict(state)[a]

                rejection_tol = (1/M) * prob_policy/prob_env

                if alpha > rejection_tol:
                    continue
                else:
                    r_i += r
                    state = s_prime

            rewards.append(r_i)


        return rewards





if __name__ == "__main__":
    A  = 361
    policy = Policy(n_actions=A)
    env_policy = Policy(n_actions=A)

    psrs = PSRS("../data/random-buffer.p", policy, env_policy, A, 10)
    psrs.evaluate()