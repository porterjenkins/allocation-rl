from stable_baselines.deepq.replay_buffer import ReplayBuffer
import config.config as cfg
from envs.state import State
from envs.allocation_env import AllocationEnv
from envs.prior import Prior
import numpy as np

class Mopo(object):

    def __init__(self, policy, env_model, epochs, rollout, n_actions, lmbda, buffer_size=50000):
        self.epochs = epochs
        self.env_model = env_model
        self.rollout = rollout
        self.policy = policy
        self.n_actions = n_actions
        self.lmbda = lmbda
        self.buffer = ReplayBuffer(buffer_size)


    def get_penalized_reward(self, r, lmbda):
        variance = np.var(r)
        return r - lmbda*variance

    def learn(self):

        results = {'rewards': [0.0]}

        for i in range(self.epochs):
            state = State.init_state(config=cfg.vals)

            for h in range(self.rollout):
                #feasible_actions = AllocationEnv.get_feasible_actions(state["board_config"])
                #action_mask = AllocationEnv.get_action_mask(feasible_actions, self.n_actions)

                # sample action a_j ~ pi(s_j)
                #action, _states = self.policy.predict(state, mask=action_mask)
                action = env.action_space.sample()
                #action = AllocationEnv.check_action(state['board_config'], action)

                # compute dynamics from env model
                new_state, r_hat, dones, info = self.env_model.step([action])

                reward = self.get_penalized_reward(r_hat, self.lmbda)

                results['rewards'].append(reward + results['rewards'][-1])

                # TODO: add (s, a, r, s') to buffer

                state = new_state

            # TODO: update policy with samples from D_env and D_model


if __name__ == "__main__":
    prior = Prior(config=cfg.vals)
    env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True, full_posterior=True)

    mopo = Mopo(policy=None,
                env_model=env,
                epochs=10,
                rollout=5,
                n_actions = env.n_actions,
                lmbda=.1
    )

    mopo.learn()