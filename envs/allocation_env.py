import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import pandas as pd
from envs.prior import Prior
import config.config as cfg
from envs.features import Features
from envs.state import State
import gym
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import datetime
from envs.models import LinearModel, HierarchicalModel
import pickle
import matplotlib.pyplot as plt
from utils import get_store_id

class AllocationEnv(gym.Env):
    """Environment model for training Reinforcement Learning agent"""
    metadata = {'render.modes': ['allocation'],
                'max_cnt_reward_not_reduce_round': cfg.vals['episode_len']}

    def __init__(self, config, prior, full_posterior=False, load_model=True, posterior_samples=25):
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.n_temporal_features = config['n_temporal_features']
        self.adj_mtx = config['adj_mtx']
        self.model_type = config['model_type']
        self.prior = prior
        self.full_posterior = full_posterior
        self.env_model = None
        self.trace = None
        self.posterior_samples = posterior_samples
        self.max_rollouts = 3 # 90 day rollouts
        self.sales = []
        self.seed()
        self.viewer = None
        self.state = None
        self.n_actions = 1 + self.n_regions*self.n_products*2
        self.cost = config["cost"]
        self.log_linear = config["log_linear"]

        self._load_data(config['model_path'], config['train_data'], load_model)
        self.sample_index = np.arange(self.feature_shape[0])

        self.cnt_reward_not_reduce_round = 0
        self.time_step_cntr = 0
        self.max_cnt_reward_not_reduce_round = self.metadata['max_cnt_reward_not_reduce_round']

        observation_shape = list(self.feature_shape)
        observation_shape[-1] = observation_shape[-1] + 1
        observation_shape = tuple(observation_shape)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({"day_vec": gym.spaces.MultiBinary(7),
                                              "board_config": spaces.Box(low=-2, high=1, shape=(self.n_regions, self.n_products),
                                                          dtype=np.int8),
                                              "prev_sales": spaces.Box(low=0, high=5, shape=(1, 6), dtype=np.int8)}
                                              )
        self.action_map = self.build_action_map()
        self.sales_quantiles = self.get_sales_quantiles(get_store_id(config["train_data"]))

    def get_sales_quantiles(self, store_id):

        q = {}
        fpath = f"../data/{store_id}-quantiles.txt"

        with open(fpath, "r") as f:
            for i, line in enumerate(f):

                q[i] = float(line.strip())

        return q



    def set_state(self, input_state):
        self.state = copy.copy(input_state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        '''
        take one simulation step given action
        :param action: tuple, action to take
        :return:
                ob: np.array, agent observation
                reward: float, reward of the passed action
                episode_over: boolean, indicates whether it is the episode end
                info: additional info for the
        '''

        action, is_valid_action = self.map_agent_action(action)
        _, posterior = self._take_action(action)

        reward = self._get_reward(is_valid_action, posterior)

        ob = self._get_state()

        episode_over = self._check_episode_over(reward)

        info = {}
        return ob, reward, episode_over, info


    def _build_env_model(self):
        ts = datetime.datetime.now()
        print("Building environment model: {}".format(ts))

        if self.model_type == 'linear':

            model = LinearModel(prior=self.prior,
                                     n_regions=self.n_regions,
                                     n_products=self.n_products,
                                     n_temporal_features=self.n_temporal_features,
                                     X_region=self.X_region,
                                     X_product=self.X_product,
                                     X_lagged=self.X_lagged,
                                     X_temporal=self.X_temporal,
                                     y=self.y,
                                     time_stamps=self.time_stamps, log_linear=self.log_linear)
        else:

            model = HierarchicalModel(prior=self.prior,
                                     n_regions=self.n_regions,
                                     n_products=self.n_products,
                                     n_temporal_features=self.n_temporal_features,
                                     X_region=self.X_region,
                                     X_product=self.X_product,
                                     X_lagged=self.X_lagged,
                                     X_temporal=self.X_temporal,
                                     y=self.y,
                                     time_stamps=self.time_stamps,
                                     product_idx=self.product_idx, log_linear=self.log_linear)

        return model.build()

    def __check_model(self):
        if self.trace is not None:
            return None
        else:
            raise ValueError("Environment model has not been loaded or trained. Try re-training or set load_model=True")

    def train(self, n_iter, n_samples, fname='model.trace', debug=False):
        print("Beginning training job - iterations: {} samples: {}".format(n_iter,n_samples))
        if debug:
            for RV in self.env_model.basic_RVs:
                print(RV.name, RV.logp(self.env_model.test_point))
        with self.env_model:
            #inference = pm.ADVI()
            #approx = pm.fit(n=n_iter, method=inference, total_grad_norm_constraint=10)
            #self.trace = approx.sample(draws=n_samples)
            self.trace = pm.sample(n_samples, tune=n_iter, init='advi+adapt_diag')
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=1000)
        #plt.plot(inference.hist)
        #plt.ylabel('ELBO')
        #plt.xlabel('iteration')
        #plt.show()
        #pm.save_trace(self.trace, directory=fname, overwrite=True)

        with open(fname, "wb") as f:
            pickle.dump(self.trace, f)

        return posterior_pred


    def _predict(self, features=None, n_samples=500):
        self.__check_model()

        self.__update_features(features)
        with self.env_model:
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples, progressbar=False)
        sales = self.__get_sales(posterior_pred['quantity_ij'], prices=features.prices)
        # clip estimated sales
        sales_hat = State.clip_val(sales, self.state.sales_bound)
        return sales_hat


    def __update_features(self, features):

        self.X_region.set_value(features.region)
        self.X_product.set_value(features.product)
        self.X_temporal.set_value(features.temporal)
        self.X_lagged.set_value(features.lagged)
        self.time_stamps.set_value(features.time_stamps)
        self.product_idx.set_value(features.product_idx)


    def __get_sales(self, q_ij, prices):
        sales = q_ij * prices
        return sales


    def reset(self):
        self.state = copy.copy(self.init_state)
        self.cnt_reward_not_reduce_round = 0
        self.viewer = None
        print("*************************Resetting Environment")
        return self._get_state()

    def render(self, action, mode='allocation'):
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _check_episode_over(self, reward):
        '''
                The episode end is to be decided
                :param reward:
                :return:
                '''
        self.cnt_reward_not_reduce_round += 1
        if self.cnt_reward_not_reduce_round > self.max_cnt_reward_not_reduce_round:
            return True
        else:
            return False

    def _get_state(self):
        return Features.featurize_state_saperate(self.state, self.sales_quantiles)

    def _get_cost(self):
        is_region_allocated = np.where(self.state.board_config == 1)[0]
        n_region_allocated = len(np.unique(is_region_allocated))
        total_cost = n_region_allocated*self.cost
        return total_cost

    def _get_reward(self, is_valid_action, posterior):
        if is_valid_action:
            sales_hat = posterior.mean(axis=0)
            if self.full_posterior:
                r = posterior.sum(axis=1) - self._get_cost()
            else:
                #r = self.state.prev_sales.sum() - self._get_cost()
                r = sales_hat.sum() - self._get_cost()
        else:
            r = -1.0
        return r

    def _take_action(self, action):
        self.state.update_board(a=action)
        state_features = Features.featurize_state(self.state)
        sales_posterior = self._predict(state_features, n_samples=self.posterior_samples)
        sales_hat = sales_posterior.mean(axis=0)
        self.state.advance(sales_hat)

        return self._get_state(), sales_posterior

    def _load_data(self, model_path, train_data_path, load_model):
        train_data = pd.read_csv(train_data_path)
        # Remove zero quantity samples from training data
        train_data = train_data[train_data['quantity'] > 0]
        train_features = Features.feature_extraction(train_data, y_col='quantity')

        self.X_region = theano.shared(train_features.region)
        self.X_product = theano.shared(train_features.product)
        self.X_temporal = theano.shared(train_features.temporal)
        self.X_lagged = theano.shared(train_features.lagged)
        self.time_stamps = theano.shared(train_features.time_stamps)
        self.product_idx = theano.shared(train_features.product_idx)
        self.y = theano.shared(train_features.y)
        self.prices = train_features.prices

        self.init_state = State.init_state(cfg.vals)
        init_features = Features.featurize_state(self.init_state).toarray()
        self.init_state_len = init_features.shape[1]
        self.feature_shape = init_features.shape

        self.init_state_dimension = len(self.feature_shape)
        self.env_model = self._build_env_model()

        if load_model:

            if not os.path.exists(model_path):
                raise Exception("Model file {} does not exist. Run train.py and try again".format(model_path))

            with open(model_path, 'rb') as f:
                self.trace = pickle.load(f)

            #with self.env_model:
                #self.trace = pm.load_trace(model_path)
            ts = datetime.datetime.now()
            print("Environment model read from disk: {}".format(ts))

    def _convert_to_categorical(self, action):
        action = action.astype(int)
        num_class = self.n_regions*self.n_products*3
        return np.eye(num_class)[action]

    def build_action_map(self):
        m = {-1: ((), 0),
             0: ((), 0)}

        idx = 1


        for a in [-1, 1]:
            for i in range(self.n_regions):
                for j in range(self.n_products):

                    m[idx] = ((i,j), a)

                    idx += 1

        return m

    def map_agent_action(self, action):

        a_mtx = np.zeros((self.n_regions, self.n_products))
        idx, val = self.action_map[action]
        a_mtx[idx] = val
        if action == -1:
            is_valid_action = False
        else:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            is_valid_action = True

        return a_mtx, is_valid_action

    @staticmethod
    def get_feasible_actions(board_config):
        curr_board = board_config.flatten()
        board_positive = curr_board + 1
        board_negative = curr_board - 1

        # add one to account for null action: idx = 0
        valid_neg_moves = set(np.where(board_negative > -1)[0] + 1)
        # to get correct action indices add one plus number of possible negative moves (n_regions * n_products)
        valid_pos_moves = set(np.where(board_positive < 2)[0] + curr_board.shape[0] + 1)
        feasible_actions = valid_pos_moves.union(valid_neg_moves)

        # null action: do nothing:
        feasible_actions = feasible_actions.union(set([0]))

        return feasible_actions

    @staticmethod
    def get_action_mask(actions, n_actions):
        mask = np.zeros(n_actions)
        for a in actions:
            mask[a] = 1.0
        return mask


    @staticmethod
    def check_action(board_config, action):
        feasible_actions = AllocationEnv.get_feasible_actions(board_config)
        if action in feasible_actions:
            return action
        else:
            return -1


if __name__ == "__main__":
    import gym

    from policies.deepq.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from policies.deepq.dqn import DQN

    prior = Prior(config=cfg.vals)

    env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)

    n_actions = env.n_actions
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


    model = DQN(MlpPolicy, env, verbose=2,learning_starts=500,exploration_fraction=.75)
    model.learn(total_timesteps=1000)

    obs = env.reset()
    for i in range(10):
        action, _states = model.predict(obs)
        # TODO: add check for feasible action space
        action = 2
        action = AllocationEnv.check_action(obs['board_config'], action)
        obs, rewards, dones, info = env.step([action])

