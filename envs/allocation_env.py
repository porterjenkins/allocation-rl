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

class AllocationEnv(object):
    """Environment model for training Reinforcement Learning agent"""

    def __init__(self, config, prior, train_features):
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.n_temporal_features = config['n_temporal_features']
        self.adj_mtx = config['adj_mtx']
        self.prior = prior
        self.X_region = theano.shared(train_features.region)
        self.X_product = theano.shared(train_features.product)
        self.X_temporal = theano.shared(train_features.temporal)
        self.X_lagged = theano.shared(train_features.lagged)
        self.time_stamps = theano.shared(train_features.time_stamps)
        self.y = theano.shared(train_features.y)
        self.prices = train_features.prices
        self.model = None
        self.trace = None
        self.state = State
        self.posterior_samples = 25

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

        self.model = env_model

    def __check_model(self):
        if self.model is not None:
            return None
        else:
            raise ValueError("environment model has not been built. run build_env_mode()")

    def train(self, n_samples, tune):
        self.__check_model()

        with self.model:
            self.trace = pm.sample(n_samples, tune=tune, init='advi+adapt_diag')
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples)


    def predict(self, features, n_samples):
        self.__check_model()

        self.__update_features(features)
        with self.model:
            posterior_pred = pm.sample_posterior_predictive(self.trace, samples=n_samples)
        sales = self.__get_sales(posterior_pred['quantity_ij'], prices=features.prices)
        return sales


    def __update_features(self, features):

        self.X_region.set_value(features.region)
        self.X_product.set_value(features.product)
        self.X_temporal.set_value(features.temporal)
        self.X_lagged.set_value(features.lagged)
        self.time_stamps.set_value(features.time_stamps)
        self.y.set_value(features.y)


    def __get_sales(self, q_ij, prices):
        sales = q_ij * prices
        return sales


    def reset(self):
        self.state = State.init_state(cfg.vals)
        return self.state

    def _get_state(self):
        state = Features.featurize_state(self.state)
        return state

    def _take_action(self, action):
        self.state.update_board(action)
        state_features = Features.featurize_state(self.state)
        sales_posterior = self.predict(state_features, n_samples=self.posterior_samples)
        sales_hat = sales_posterior.mean(axis=0)
        print(sales_hat)
        self.state.advance(sales_hat)

        return self.state





if __name__ == "__main__":
    prior = Prior(config=cfg.vals,
                  fname='prior.json')
    train_data = pd.read_csv('../train-data-simple.csv', index_col=0)
    test_data = pd.read_csv('../test-data-simple.csv', index_col=0)

    train_features = Features.feature_extraction(train_data, prices=cfg.vals['prices'], y_col='quantity')
    test_features = Features.feature_extraction(test_data, prices=cfg.vals['prices'])



    env = AllocationEnv(config=cfg.vals, prior=prior, train_features=train_features)
    env.build_env_model()
    env.train(n_samples=100, tune=100)
    env.reset()
    a = np.zeros((4, 4))
    a[3, 3] = 1.0
    ob = env._take_action(a)
    print(ob)
