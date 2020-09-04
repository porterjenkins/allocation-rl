import numpy as np
from tqdm import tqdm
import pickle
import random

from stable_baselines.deepq.replay_buffer import ReplayBuffer
import config.config as cfg
from policies.deepq.policies import MlpPolicy
from envs.allocation_env import AllocationEnv
from envs.prior import Prior
from policies.deepq.dqn import DQN
from envs.state import State


class Mopo(object):

    def __init__(self,
                 policy,
                 env_model=None,
                 rollout_batch_size=None,
                 buffer_path=None,
                 epochs=None,
                 rollout=None,
                 n_actions=None,
                 lmbda=None,
                 buffer_size=50000,
                 eps=.3):


        self.epochs = epochs
        self.env_model = env_model
        self.rollout_batch_size = rollout_batch_size
        self.buffer_path = buffer_path
        self.rollout = rollout
        self.policy = policy
        self.n_actions = n_actions
        self.lmbda = lmbda
        self.eps = eps
        self.buffer_env, self.buffer_model = self.init_buffer(self.buffer_path, buffer_size)
        self.n_regions = env_model.n_regions
        self.n_products = env_model.n_products




    def init_buffer(self, fpath=None, buffer_size=None):

        with open(fpath, 'rb') as f:
            buffer_env = pickle.load(f)

        buffer_model = ReplayBuffer(buffer_size)

        print("Copying environment buffer: ")
        for i in tqdm(range(len(buffer_env))):
            obs_t, action, reward, obs_tp1, done = buffer_env._storage[i]
            buffer_model.add(obs_t, action, reward, obs_tp1, done)

        return buffer_env, buffer_model


    def save_buffer(self, fpath="../data/mopo-buffer.p"):

        with open(fpath, 'wb') as f:
            pickle.dump(self.buffer_model, f)



    def get_penalized_reward(self, r, lmbda):
        variance = np.var(r)
        mean = r.mean()
        return mean - lmbda*variance

    def learn(self):


        for i in range(self.epochs):
            print(f"Epoch {i}/{self.epochs}")
            pbar = tqdm(range(self.rollout_batch_size))
            for b in pbar:
                #state = self.buffer_env.sample(batch_size=1)[0][0]
                state = self.env_model.reset()
                state = State.get_vec_observation(state)



                for h in range(self.rollout):
                    pbar.set_description(f"batch: {b} rollout: {h}")
                    board_cfg = State.get_board_config_from_vec(state,
                                                                n_regions=self.n_regions,
                                                                n_products=self.n_products
                                                                )

                    feasible_actions = AllocationEnv.get_feasible_actions(board_cfg)
                    #feasible_actions = AllocationEnv.get_feasible_actions(state["board_config"])
                    action_mask = AllocationEnv.get_action_mask(feasible_actions, self.n_actions)

                    # sample action a_j ~ pi(s_j)
                    alpha = random.random()

                    if alpha < self.eps:
                        action = self.env_model.action_space.sample()
                    else:
                        action, _states = self.policy.predict(state.reshape(1, -1), mask=action_mask)

                    # compute dynamics from env model
                    new_state, r_hat, dones, info = self.env_model.step(action)
                    new_state = State.get_vec_observation(new_state)

                    reward = self.get_penalized_reward(r_hat, self.lmbda)


                    # add (s, a, r, s') to buffer
                    self.buffer_model.add(obs_t=state,
                                          action=action,
                                          reward=reward,
                                          obs_tp1=new_state,
                                          done=float(dones))

                    state = new_state



                # update policy with samples from D_env and D_model
                self.policy.update_weights(self.buffer_model)
        self.save_buffer()


if __name__ == "__main__":
    prior = Prior(config=cfg.vals)
    env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True, full_posterior=True)
    policy = DQN(MlpPolicy, env, batch_size=32)

    mopo = Mopo(policy=policy,
                env_model=env,
                rollout_batch_size=10,
                epochs=100,
                rollout=10,
                n_actions = env.n_actions,
                lmbda=1e-3,
                buffer_path="../data/random-buffer.p"

    )

    mopo.learn()