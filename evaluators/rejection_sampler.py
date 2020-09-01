import pickle
import numpy as np
import random
import torch

import config.config as cfg

from evaluators.eval_queue import EvalQueue
from evaluators.mlp_env_model import MLPClassifer

from policies.deepq.dqn import DQN
from envs.allocation_env import AllocationEnv
from envs.state import State

def get_buffer(fpath):
    with open(fpath, 'rb') as f:
        buffer = pickle.load(f)

    return buffer



class Policy(object):
    def __init__(self, n_actions, buffer_path, model_path):
        self.buffer_path = buffer_path
        self.n_actions = n_actions
        self.neural_net = self.load_policy(model_path)
        self.neural_net.eval()


    def load_policy(self, fpath):
        model = torch.load(fpath)
        return model


    def predict_proba(self, state):
        state = torch.from_numpy(state.astype(np.float32))
        a_hat = self.neural_net.forward(state).data.numpy()
        return a_hat



class PSRS(object):

    """Per-state Rejection Sampling evaluator"""

    def __init__(self, buffer_path, policy, env_policy, n_actions, n_regions, n_products, n_episodes):

        self.buffer_path = buffer_path
        self.policy = policy
        self.env_policy = env_policy
        self.n_actions = n_actions
        self.n_regions = n_regions
        self.n_products = n_products
        self.n_episodes = n_episodes

        self.buffer = get_buffer(self.buffer_path)
        self.queue = self.build_queue(self.buffer)


    def reduced_state(self, s):
        return s[7:]




    def build_queue(self, buffer):

        queue = EvalQueue()


        for i in range(len(buffer.storage)):
            s, a, r, s_prime, _ = buffer.storage[i]

            if not queue.is_in(s):
                queue.add(s)

            queue[s].append((s, a, r, s_prime))

        queue.permute()

        return queue

    def get_m(self, state, mask):


        prob_policy = self.policy.proba_step(state.reshape(1, -1), mask=mask)[0]
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
                board_cfg = State.get_board_config_from_vec(state,
                                                            n_regions=self.n_regions,
                                                            n_products=self.n_products
                                                            )
                feasible_actions = AllocationEnv.get_feasible_actions(board_cfg)
                action_mask = AllocationEnv.get_action_mask(feasible_actions, self.n_actions)

                M = self.get_m(state, action_mask)

                try:
                    _, a, r, s_prime = self.queue[state].pop()
                except IndexError:
                    break

                alpha = random.random()

                prob_policy = self.policy.proba_step(state.reshape(1, -1), mask=action_mask)[0][a]
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
    model_path = "../data/env_policy.pt"

    mopo_dqn = DQN.load(f"../experiments/models/mopo-policy.p")
    env_policy = Policy(A, buffer_path, model_path)


    psrs = PSRS(buffer_path, mopo_dqn, env_policy, A, cfg.vals["n_regions"], cfg.vals["n_products"], 10)
    r = psrs.evaluate()
    mean = np.mean(r)
    sigma = np.std(r)

    print(mean, sigma)