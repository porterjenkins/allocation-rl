import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import pandas as pd
import json
from plot import plot_posterior_predictive_check
from envs.prior import Prior
import config.config as cfg
from envs.features import Features
from envs.state import State
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn import metrics
from itertools import cycle, islice


class AllocationEnv(gym.Env):
    """Environment model for training Reinforcement Learning agent"""
    metadata = {'render.modes': ['allocation'],
                'max_cnt_reward_not_reduce_round': 100}

    def __init__(self, config, prior, data_model_path):
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.n_temporal_features = config['n_temporal_features']
        self.adj_mtx = config['adj_mtx']
        self.prior = prior
        self.env_model = None
        self.trace = None
        self.posterior_samples = 25
        self.sales = []
        self.seed()
        self.viewer = None
        self.state = State.init_state(cfg.vals)

        self._load_data(data_model_path)
        self.sample_index = np.arange(self.feature_shape[0])

        self.cnt_reward_not_reduce_round = 0
        self.max_cnt_reward_not_reduce_round = self.metadata['max_cnt_reward_not_reduce_round']

        observation_shape = list(self.feature_shape)
        observation_shape[-1] = observation_shape[-1] + 1
        observation_shape = tuple(observation_shape)

        # todo modify the action space and observation space
        self.action_space = spaces.MultiDiscrete([self.n_regions, self.n_products, 2])
        self.observation_space = spaces.Box(low=-2, high=2, shape=observation_shape)






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

        self._take_action(action)

        reward = self._get_reward()

        ob = self._get_state()

        episode_over = self._check_episode_over(reward)

        info = {}
        return ob, reward, episode_over, info


    def build_env_model(self):


        with pm.Model() as env_model:

            # Generate region weights
            w_r = pm.MvNormal('w_r', mu=self.prior.loc_w_r, cov=self.prior.scale_w_r,
                              shape=self.n_regions)

            # Generate Product weights
            w_p = pm.MvNormal('w_p', mu=self.prior.loc_w_p, cov=self.prior.scale_w_p,
                              shape=self.n_products)

            # Prior for customer weight
            w_c = pm.Normal('w_c', mu=self.prior.loc_w_c, sigma=self.prior.scale_w_c)

            # Generate customer weight
            w_s = pm.Gamma('w_s', mu=self.prior.loc_w_s, sigma=self.prior.scale_w_s)

            # Generate temporal weights
            w_t = pm.MvNormal('w_t', mu=self.prior.loc_w_t, cov=self.prior.scale_w_t,
                              shape=self.n_temporal_features)
            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)
            c_t = pm.Poisson("customer_t", mu=lambda_c_t, shape=self.X_temporal.shape.eval()[0])

            c_all = c_t[self.time_stamps] * w_c

            lambda_q = pm.math.dot(self.X_region, w_r.T) + pm.math.dot(self.X_product, w_p.T) + c_all + w_s * self.X_lagged

            q_ij = pm.Poisson('quantity_ij', mu=lambda_q, observed=self.y)

        return env_model

    def __check_model(self):
        if self.env_model is not None:
            return None
        else:
            raise ValueError("environment model has not been built. run build_env_mode()")

    def train(self, n_samples, tune):
        self.__check_model()

        with self.env_model:
            self.trace = pm.sample(n_samples, tune=tune, init='advi+adapt_diag')
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples)


    def predict(self, features, n_samples):
        self.__check_model()

        self.__update_features(features)
        with self.env_model:
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples)
        sales = self.__get_sales(posterior_pred['quantity_ij'], prices=features.prices)
        return sales


    def __update_features(self, features):

        self.X_region.set_value(features.region)
        self.X_product.set_value(features.product)
        self.X_temporal.set_value(features.temporal)
        self.X_lagged.set_value(features.lagged)
        self.time_stamps.set_value(features.time_stamps)


    def __get_sales(self, q_ij, prices):
        sales = q_ij * prices
        return sales


    def reset(self):
        self.state = self.init_features
        self.cnt_reward_not_reduce_round = 0
        self.viewer = None

        return self.state

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
        state = Features.featurize_state(self.state).toarray()
        return state

    def _get_reward(self):
        r = self.state.prev_sales.sum()
        return r

    def _take_action(self, action):
        self.state.update_board(a=action)
        state_features = Features.featurize_state(self.state)
        sales_posterior = self.predict(state_features, n_samples=self.posterior_samples)
        sales_hat = sales_posterior.mean(axis=0)
        self.state.advance(sales_hat)

        return self._get_state()

    def _load_data(self, file_path, train=True):
        train_data = pd.read_csv(file_path, index_col=0)
        train_features = Features.feature_extraction(train_data, prices=cfg.vals['prices'], y_col='quantity')

        self.X_region = theano.shared(train_features.region)
        self.X_product = theano.shared(train_features.product)
        self.X_temporal = theano.shared(train_features.temporal)
        self.X_lagged = theano.shared(train_features.lagged)
        self.time_stamps = theano.shared(train_features.time_stamps)
        self.y = theano.shared(train_features.y)
        self.prices = train_features.prices

        self.init_features = State.init_state(cfg.vals)
        init_state = Features.featurize_state(self.init_features).toarray()
        self.init_state_len = init_state.shape[1]
        self.feature_shape = init_state.shape
        self.init_state_dimension = len(self.feature_shape)
        self.env_model = self.build_env_model()  # to be implemented

        if train:
            self.train(n_samples=100, tune=100)





if __name__ == "__main__":
    prior = Prior(config=cfg.vals,
                  fname='prior.json')


    env = AllocationEnv(config=cfg.vals,prior=prior,data_model_path='../train-data-simple.csv')

    a = np.zeros((4, 4))
    a[3, 3] = 1.0
    ob = env._take_action(a)
    print(env._get_reward())

    a = np.zeros((4, 4))
    a[3, 0] = 1.0
    ob = env._take_action(a)
    print("reward: {}".format(env._get_reward()))
