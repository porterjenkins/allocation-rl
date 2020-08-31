import pickle
import numpy as np
import random

from evaluators.eval_queue import EvalQueue

from sklearn.neural_network import MLPClassifier

def get_buffer(fpath):
    with open(fpath, 'rb') as f:
        buffer = pickle.load(f)

    return buffer


class Policy(object):
    def __init__(self, n_actions, buffer_path):
        self.buffer_path = buffer_path
        self.n_actions = n_actions
        self.buffer = get_buffer(buffer_path)
        self.neural_net = MLPClassifier(hidden_layer_sizes=(256, 64))

        self.X_train, self.y_train = self.get_train_data()


    def get_train_data(self):

        s, a, r, s_prime, _ = self.buffer.storage[0]
        state_space = len(s)
        n_samples = len(self.buffer.storage)

        X = np.zeros((n_samples, state_space))
        y = np.zeros((n_samples, 1))

        for i in range(len(self.buffer.storage)):
            s, a, r, s_prime, _ = self.buffer.storage[i]

            X[i, :] = s
            y[i, 0] = a

        return X, y

    def train(self):
        self.neural_net.fit(self.X_train, self.y_train)


    def predict(self, state):

        a_hat = self.neural_net.predict(state.reshape(1, -1))

        return a_hat

    def predict_proba(self, state):
        a_hat = self.neural_net.predict_proba(state.reshape(1, -1))[0]
        return a_hat



class PSRS(object):

    """Per-state Rejection Sampling evaluator"""

    def __init__(self, buffer_path, policy, env_policy, n_actions, n_episodes):

        self.buffer_path = buffer_path
        self.policy = policy
        self.env_policy = env_policy
        self.n_actions = n_actions
        self.n_episodes = n_episodes

        self.buffer = get_buffer(self.buffer_path)
        self.queue = self.build_queue(self.buffer)






    def build_queue(self, buffer):

        queue = EvalQueue()


        for i in range(len(buffer.storage)):
            s, a, r, s_prime, _ = buffer.storage[i]

            if not queue.is_in(s):
                queue.add(s)

            queue[s].append((s, a, r, s_prime))

        queue.permute()

        return queue

    def get_m(self, state):

        prob_policy = self.policy.predict_proba(state)
        prob_env = self.env_policy.predict_proba(state)

        M = -1

        for a in range(self.n_actions):

            M_prime = prob_policy[a] /  prob_env[a]

            if M_prime > M:
                M = M_prime

        return M

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

                prob_policy = self.policy.predict_proba(state)[a]
                prob_env = self.env_policy.predict_proba(state)[a]

                rejection_tol = (1/M) * prob_policy/prob_env

                if alpha > rejection_tol:
                    continue
                else:
                    r_i += r
                    state = s_prime

            rewards.append(r_i)


        return rewards





if __name__ == "__main__":
    A = 361
    buffer_path = "../data/store-2-buffer-r.p"

    policy = Policy(A, buffer_path)
    policy.train()

    env_policy = Policy(A, buffer_path)
    env_policy.train()

    psrs = PSRS(buffer_path, policy, env_policy, A, 10)
    psrs.evaluate()