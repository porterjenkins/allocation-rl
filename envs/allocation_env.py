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

from gym import error, spaces, utils
from gym.utils import seeding
import datetime
from envs.models import LinearModel, HierarchicalModel
import pickle

class AllocationEnv(gym.Env):
    """Environment model for training Reinforcement Learning agent"""
    metadata = {'render.modes': ['allocation'],
                'max_cnt_reward_not_reduce_round': 100}

    def __init__(self, config, prior, load_model=True):
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.n_temporal_features = config['n_temporal_features']
        self.adj_mtx = config['adj_mtx']
        self.model_type = config['model_type']
        self.prior = prior
        self.env_model = None
        self.trace = None
        self.posterior_samples = 25
        self.sales = []
        self.seed()
        self.viewer = None
        self.state = None
        self.n_actions = 1 + self.n_regions*self.n_products*2

        self._load_data(config['model_path'], config['train_data'], load_model)
        self.sample_index = np.arange(self.feature_shape[0])

        self.cnt_reward_not_reduce_round = 0
        self.max_cnt_reward_not_reduce_round = self.metadata['max_cnt_reward_not_reduce_round']

        observation_shape = list(self.feature_shape)
        observation_shape[-1] = observation_shape[-1] + 1
        observation_shape = tuple(observation_shape)

        # todo modify the action space and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({"day_vec": gym.spaces.MultiBinary(7),
                                              "board_config": spaces.Box(low=-2, high=1, shape=(self.n_regions, self.n_products),
                                                          dtype=np.int8),
                                              "prev_sales": spaces.Box(low=0, high=np.inf, shape=(self.n_regions, self.n_products),
                                                          dtype=np.float32)}
                                              )


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
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action = AllocationEnv.map_agent_action(action)
        self._take_action(action)

        reward = self._get_reward()

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
                                     time_stamps=self.time_stamps)
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
                                     product_idx=self.product_idx)

        return model.build()

    def __check_model(self):
        if self.trace is not None:
            return None
        else:
            raise ValueError("Environment model has not been loaded or trained. Try re-training or set load_model=True")

    def train(self, n_iter, n_samples, fname='model.trace'):
        self.__check_model()
        print("Beginning training job - iterations: {} samples: {}".format(n_iter,n_samples))
        with self.env_model:
            inference = pm.ADVI()
            approx = pm.fit(n=n_iter, method=inference)
            self.trace = approx.sample(draws=100)
            #self.trace = pm.sample(n_samples, tune=tune, init='advi+adapt_diag')
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples)
        #pm.save_trace(self.trace, directory=fname, overwrite=True)

        with open(fname, "wb") as f:
            pickle.dump(self.trace, f)

        return posterior_pred


    def _predict(self, features, n_samples):
        self.__check_model()

        self.__update_features(features)
        with self.env_model:
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples, progressbar=False)
        sales = self.__get_sales(posterior_pred['quantity_ij'], prices=features.prices)
        return sales


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
        self.state = self.init_state
        self.cnt_reward_not_reduce_round = 0
        self.viewer = None

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
        return Features.featurize_state_saperate(self.state)

    def _get_reward(self):
        r = self.state.prev_sales.sum()
        return r

    def _take_action(self, action):
        self.state.update_board(a=action)
        state_features = Features.featurize_state(self.state)
        sales_posterior = self._predict(state_features, n_samples=self.posterior_samples)
        sales_hat = sales_posterior.mean(axis=0)
        self.state.advance(sales_hat)

        return self._get_state()

    def _load_data(self, model_path, train_data_path, load_model):
        train_data = pd.read_csv(train_data_path, index_col=0)
        train_features = Features.feature_extraction(train_data, prices=cfg.vals['prices'], y_col='quantity')

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

    @staticmethod
    def map_agent_action(action):
        n_r = cfg.vals['n_regions']
        n_p = cfg.vals['n_products']
        n_actions = 1 + n_r * n_p * 2

        if isinstance(action, np.int64) or isinstance(action, np.int32) or isinstance(action, np.int8) or isinstance(action, int):
            action = np.array([action])
        a_idx = action.astype(int)[0] - 1

        action_vec = np.zeros(n_actions - 1)

        if a_idx >= 0:
            # if action is valid action
            action_vec[a_idx] = 1.0
            action_mtx = action_vec.reshape((n_r, n_p, 2))
            action_space_arr = np.zeros((n_r, n_p, 2))
            action_space_arr[:, :, 0] = -1.0
            action_space_arr[:, :, 1] = 1.0
            a_sparse = action_mtx * action_space_arr

        else:
            # if action is the null action (do nothing, i == 0)
            a_sparse = action_vec.reshape((n_r, n_p, 2))

        return a_sparse.sum(axis=2)

    @staticmethod
    def get_feasible_actions(board_config):
        curr_board = board_config.flatten()
        board_positive = curr_board + 1
        board_negative = curr_board - 1

        valid_pos_moves = set(np.where(board_positive <= 1)[0])
        valid_neg_moves = set(np.where(board_negative >= -1)[0])
        feasible_action = valid_pos_moves.intersection(valid_neg_moves)

        return feasible_action

    @staticmethod
    def is_feasible_action(board_config, action):
        feasible_actions = AllocationEnv.get_feasible_actions(board_config)
        if action in feasible_actions:
            return True
        else:
            return False

    @staticmethod
    def check_action(board_config, action):
        is_feasible = AllocationEnv.is_feasible_action(board_config, action)
        if is_feasible:
            return action
        else:
            return 0


if __name__ == "__main__":
    import gym

    from policies.deepq.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from policies.deepq.dqn import DQN

    prior = Prior(config=cfg.vals)

    env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)

    n_actions = env.n_actions
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


    model = DQN(MlpPolicy, env, verbose=2,learning_starts=20)
    model.learn(total_timesteps=100)

    obs = env.reset()
    for i in range(10):
        action, _states = model.predict(obs)
        # TODO: add check for feasible action space
        action = 2
        action = AllocationEnv.check_action(obs['board_config'], action)
        obs, rewards, dones, info = env.step([action])
        print(obs)

